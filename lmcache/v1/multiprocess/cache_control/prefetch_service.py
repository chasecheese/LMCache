# SPDX-License-Identifier: Apache-2.0
"""Node-local warm-prefetch operations (submit + status).

:class:`PrefetchService` resolves a token sequence to per-rank keys and submits
a **warm** load (retained, unlocked) from L2 into L1, returning a ``request_id``
the caller polls. It validates its own inputs and raises transport-agnostic
domain errors (see :mod:`cache_control.errors`); the HTTP layer maps those to
status codes. It owns the node's :class:`WarmPrefetchJobs` table.
"""

# Standard
from typing import Any

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.tiers import Tier
from lmcache.v1.multiprocess.cache_control.errors import (
    InvalidRequest,
    NotFound,
    Unavailable,
)
from lmcache.v1.multiprocess.cache_control.key_resolver import (
    MAX_TOKEN_IDS,
    resolve_l1_keys,
)
from lmcache.v1.multiprocess.warm_prefetch import (
    COMPLETED,
    UNKNOWN,
    WarmPrefetchJobs,
)

# Warm prefetch loads from L2 into L1; other directions are rejected.
_SOURCE_TIER = Tier.L2
_TARGET_TIER = Tier.L1

# Page size used when scanning an adapter's full key inventory.
_SCAN_PAGE_SIZE = 5000
# Keys per WARM job submitted by a scan; bounds the per-job L1 write
# reservation and gives the caller per-batch progress via polling.
_SCAN_BATCH_KEYS = 2048


class PrefetchService:
    """Submit and poll warm prefetches on one node.

    Args:
        engine: The node's cache engine (resolves tokens and runs the load).
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._jobs = WarmPrefetchJobs()

    def submit(
        self,
        model_name: str,
        world_size: int,
        token_ids: list[int],
        cache_salt: str,
        source_tier: Tier,
        target_tier: Tier,
    ) -> dict[str, object]:
        """Submit a warm prefetch of a token sequence's chunks from L2 into L1.

        Returns:
            ``{"request_id", "chunks", "status": "submitted"}``, or
            ``{"chunks": 0, "status": "noop"}`` for a sub-chunk sequence.

        Raises:
            InvalidRequest: unsupported direction, token cap exceeded, or an
                invalid key field.
            Unavailable: no layout registered for the model (via the resolver).
        """
        if source_tier != _SOURCE_TIER or target_tier != _TARGET_TIER:
            raise InvalidRequest(
                f"unsupported prefetch direction {source_tier.value!r}->"
                f"{target_tier.value!r}; only {_SOURCE_TIER.value!r}->"
                f"{_TARGET_TIER.value!r}"
            )
        if len(token_ids) > MAX_TOKEN_IDS:
            raise InvalidRequest(
                f"too many token_ids in a single request "
                f"(limit={MAX_TOKEN_IDS}, got={len(token_ids)})"
            )
        obj_keys, chunks, layout_desc = resolve_l1_keys(
            self._engine, model_name, world_size, token_ids, cache_salt
        )
        if not chunks:
            return {"chunks": 0, "status": "noop"}
        request_id = self._jobs.submit(
            self._engine.storage_manager, obj_keys, layout_desc
        )
        return {"request_id": request_id, "chunks": chunks, "status": "submitted"}

    def submit_scan(
        self,
        model_name: str,
        world_size: int,
        cache_salt: str | None = None,
    ) -> dict[str, object]:
        """Warm-prefetch EVERY object resident in the primary L2 adapter into L1.

        Scans the adapter's full key inventory (``list_l2_keys`` pagination),
        keeps keys matching ``model_name`` (and ``cache_salt`` when given), and
        submits them as batched WARM jobs of ``_SCAN_BATCH_KEYS`` keys each.
        This is a bulk L2->L1 state injection: disk reads into pinned L1
        buffers, retained and unpinned, zero GPU work.

        Args:
            model_name: Model whose layout (and keys) to load.
            world_size: Tensor-parallel world size selecting the layout.
            cache_salt: If set, restrict to keys with exactly this salt.

        Returns:
            ``{"request_ids", "total_keys", "status": "submitted"}`` (poll each
            id), or ``{"request_ids": [], "total_keys": 0, "status": "noop"}``
            when the adapter holds nothing matching.

        Raises:
            Unavailable: no layout registered for the model, no L2 adapters
                configured, or the adapter does not support listing.
        """
        layout_desc = self._engine.context.layout_desc_registry.find(
            model_name, world_size
        )
        if layout_desc is None:
            raise Unavailable(
                f"no layout registered for model_name={model_name!r} "
                f"world_size={world_size}; the model has not allocated "
                f"KV cache on this node yet"
            )
        adapters = self._engine.storage_manager.l2_adapters()
        if not adapters:
            raise Unavailable("no L2 adapters configured")
        desc, adapter = adapters[0]

        keys: list[ObjectKey] = []
        cursor: str | None = None
        while True:
            try:
                page = adapter.list_l2_keys(
                    model_name=model_name,
                    page_size=_SCAN_PAGE_SIZE,
                    cursor=cursor,
                )
            except NotImplementedError as exc:
                raise Unavailable(
                    f"L2 adapter {desc.type_name!r} does not support "
                    f"listing: {exc}"
                ) from None
            for entry in page.entries:
                key = entry.key.to_object_key()
                if cache_salt is not None and key.cache_salt != cache_salt:
                    continue
                keys.append(key)
            if page.next_page_token is None:
                break
            cursor = page.next_page_token

        if not keys:
            return {"request_ids": [], "total_keys": 0, "status": "noop"}
        request_ids = [
            self._jobs.submit(
                self._engine.storage_manager,
                keys[i : i + _SCAN_BATCH_KEYS],
                layout_desc,
            )
            for i in range(0, len(keys), _SCAN_BATCH_KEYS)
        ]
        return {
            "request_ids": request_ids,
            "total_keys": len(keys),
            "status": "submitted",
        }

    def status(self, request_id: str) -> dict[str, object]:
        """Report a job's status, finalizing it on the first completed poll.

        Returns:
            ``{"request_id", "status": "pending"}`` or ``{"request_id",
            "status": "completed", "found_keys", "total_keys"}``.

        Raises:
            NotFound: unknown id (already completed-and-consumed, or never
                submitted).
        """
        status = self._jobs.poll(self._engine.storage_manager, request_id)
        if status.state == UNKNOWN:
            raise NotFound(
                f"unknown prefetch request_id={request_id!r} "
                f"(already completed or never submitted)"
            )
        if status.state == COMPLETED:
            return {
                "request_id": request_id,
                "status": COMPLETED,
                "found_keys": status.found_keys,
                "total_keys": status.total_keys,
            }
        return {"request_id": request_id, "status": status.state}
