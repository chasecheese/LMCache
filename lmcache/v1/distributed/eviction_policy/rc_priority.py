# SPDX-License-Identifier: Apache-2.0
"""
rc-testbed: priority-driven L1 eviction policy (the D4 "keep how long" surface).

Victim order: lowest external priority first; within equal priority, LRU order
(least recently used first). Priorities are attached per chunk by the external
KV-store scheduling policy at store time (``lmcache.v1.store_policy_hook``,
loaded via ``LMCACHE_STORE_POLICY_REF``); keys the external policy never saw
(hook inactive, prefetch loads, ...) default to priority 0.0, so with the hook
inactive this policy degrades to plain LRU.
"""

# Standard
from collections import OrderedDict
from collections.abc import Callable
import threading

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction import EvictionPolicy
from lmcache.v1.distributed.internal_api import (
    EvictionAction,
    EvictionDestination,
)


class RCPriorityEvictionPolicy(EvictionPolicy):
    """
    Priority-driven eviction policy for the rc-testbed D4 decision surface.

    Structurally a peer of
    :class:`~lmcache.v1.distributed.eviction_policy.lru.LRUEvictionPolicy`:
    the same recency bookkeeping (including the reversed per-request insertion
    so that a request's LATER chunks are evicted before its earlier ones —
    prefix-hit friendly), but victim selection sorts eligible keys by external
    priority (ascending) before recency, using a stable sort so equal
    priorities keep pure LRU order.

    Thread Safety:
        Thread-safe; all operations are protected by one lock. Priority
        lookups go through ``store_policy_hook`` (its own lock).
    """

    def __init__(
        self,
        default_destination: EvictionDestination = EvictionDestination.DISCARD,
    ):
        """
        Initialize the policy.

        Args:
            default_destination: The default destination for evicted objects.
                Defaults to DISCARD.
        """
        self._lock = threading.Lock()
        self._order: OrderedDict[ObjectKey, None] = OrderedDict()
        self._destinations: list[EvictionDestination] = []
        self._default_destination = default_destination

    def register_eviction_destination(self, destination: EvictionDestination):
        """
        Register an eviction destination for the eviction policy to use.

        Args:
            destination (EvictionDestination): The eviction destination to register
        """
        with self._lock:
            if destination not in self._destinations:
                self._destinations.append(destination)

    def on_keys_created(self, keys: list[ObjectKey]):
        """
        Track new keys as most recently used (reversed, as in LRU: within one
        request the later chunks must be evicted first to preserve prefix hits).

        Args:
            keys (list[ObjectKey]): The keys that have been created
        """
        if not keys:
            return
        with self._lock:
            for key in reversed(keys):
                if key in self._order:
                    self._order.move_to_end(key)
                else:
                    self._order[key] = None

    def on_keys_touched(self, keys: list[ObjectKey]):
        """
        Move accessed keys to the most recently used position.

        Args:
            keys (list[ObjectKey]): The keys that have been accessed
        """
        if not keys:
            return
        with self._lock:
            for key in reversed(keys):
                if key in self._order:
                    self._order.move_to_end(key)

    def on_keys_removed(self, keys: list[ObjectKey]):
        """
        Stop tracking deleted keys and forget their external priorities.

        Args:
            keys (list[ObjectKey]): The keys that have been deleted
        """
        if not keys:
            return
        # First Party
        from lmcache.v1.store_policy_hook import mp_forget_priorities

        with self._lock:
            for key in keys:
                if key in self._order:
                    del self._order[key]
        mp_forget_priorities(keys)

    def get_eviction_actions(
        self,
        expected_ratio: float,
        key_eligible_filter: Callable[[ObjectKey], bool] | None = None,
        cache_salt: str | None = None,
    ) -> list[EvictionAction]:
        """
        Select victims: lowest external priority first, LRU order within ties.

        Args:
            expected_ratio (float): Approximate fraction of tracked keys to
                evict, in [0.0, 1.0].
            key_eligible_filter: Optional callable; keys for which it returns
                False (e.g. locked keys) are skipped.
            cache_salt: Ignored (not user-level).

        Returns:
            list[EvictionAction]: The eviction actions to perform.
        """
        # First Party
        from lmcache.v1.store_policy_hook import mp_priority_of

        with self._lock:
            if not self._order:
                return []

            expected_ratio = max(0.0, min(1.0, expected_ratio))
            target_count = int(len(self._order) * expected_ratio)
            if expected_ratio > 0 and target_count == 0 and len(self._order) > 0:
                target_count = 1
            if target_count == 0:
                return []

            # Stable sort by priority over the LRU iteration order: equal
            # priorities keep least-recently-used-first order.
            candidates = sorted(self._order, key=mp_priority_of)

            keys_to_evict: list[ObjectKey] = []
            for key in candidates:
                if key_eligible_filter is not None and not key_eligible_filter(key):
                    continue
                keys_to_evict.append(key)
                if len(keys_to_evict) >= target_count:
                    break

            if not keys_to_evict:
                return []

            destination = self._default_destination
            if self._destinations:
                destination = self._destinations[0]

            return [EvictionAction(keys=keys_to_evict, destination=destination)]
