# SPDX-License-Identifier: Apache-2.0
# autoresearch patch: pluggable KV-store scheduling policy (the D4 decision surface).
#
# If LMCACHE_STORE_POLICY_REF="pkg.module:attr" is set, that object is loaded once per
# process (PYTHONPATH must make it importable) and its on_store(chunks, ctx) decides,
# per chunk, which backends receive the KV ("cpu"/"disk"/"remote"; empty = skip) and a
# priority. Unset env -> hook inactive -> stock write-through fan-out.
"""Loader for the external KV-store policy (see the testbed/store_policy.py spec)."""

# Standard
import importlib
import os
import threading

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

ENV = "LMCACHE_STORE_POLICY_REF"

# In-process (v1 cache_engine) path: backend registry name -> short target name.
# Dormant in MP mode; kept for the non-MP connector.
BACKEND_TARGET = {
    "LocalCPUBackend": "cpu",
    "LocalDiskBackend": "disk",
    "RemoteBackend": "remote",
}
IN_PROCESS_TARGETS = ("cpu", "disk", "remote")

# MP (multiprocess server) path: the policy decides per chunk among these tiers.
MP_TARGETS = ("l1", "l2")

_policy = None
_loaded = False


def apply_store_policy(policy, chunk_list, req_id, total_tokens):
    """chunk_list = [(start, end, CacheEngineKey), ...] in request order.
    Returns per-chunk target sets (subset of {"cpu","disk","remote"}; empty = skip)."""
    # Imported here: resolvable via the same PYTHONPATH that made the policy loadable.
    from testbed.store_policy import ChunkInfo, StoreContext, validate_decisions

    chunks = [ChunkInfo(key=key.chunk_hash, start=start, end=end, index=i)
              for i, (start, end, key) in enumerate(chunk_list)]
    ctx = StoreContext(req_id=str(req_id), total_tokens=total_tokens)
    decisions = validate_decisions(policy.on_store(chunks, ctx), len(chunks),
                                   known=IN_PROCESS_TARGETS)
    return [targets for targets, _prio in decisions]


# ---------------------------------------------------------------------------
# MP (multiprocess server) decision surface.
#
# Flow: modules/lmcache_driven_transfer.store() consults the policy per chunk
# BEFORE reserve_write (empty targets = SKIP: no L1 reservation, no D2H copy)
# and records the surviving keys' targets here; the "external" StorePolicy
# (storage_controllers/store_policy.py) consumes the records to decide the L2
# fan-out and whether to drop the key from L1 after the L2 store ({"l2"}-only
# chunks use L1 purely as a copy buffer).
#
# Records are popped on consumption; the caps below only bound pathological
# leftovers (e.g. stores that failed between record and controller pickup).
# ---------------------------------------------------------------------------

_mp_lock = threading.Lock()
_mp_pending: dict = {}   # ObjectKey -> frozenset(targets), awaiting StoreController
_mp_l1_drop: dict = {}   # ObjectKey -> True, drop from L1 after successful L2 store
_mp_priority: dict = {}  # ObjectKey -> float, lives as long as the key does (read by
                         # the priority eviction policy; forgotten on key removal)
_mp_chunk_size = 256     # cached from the store path (prefetch has no config handle)
_MP_CAP = 262144


def apply_mp_store_policy(policy, obj_keys, request_id, instance_id, chunk_size,
                          extras=None):
    """obj_keys: group-0 ObjectKeys in request chunk order. Returns per-chunk
    decisions [(targets, priority), ...] (targets ⊆ {"l1","l2"}; empty = SKIP)."""
    from testbed.store_policy import ChunkInfo, StoreContext, validate_decisions

    global _mp_chunk_size
    _mp_chunk_size = chunk_size
    chunks = [ChunkInfo(key=k.chunk_hash.hex(), start=i * chunk_size,
                        end=(i + 1) * chunk_size, index=i)
              for i, k in enumerate(obj_keys)]
    ctx = StoreContext(req_id=str(request_id),
                       total_tokens=len(obj_keys) * chunk_size,
                       instance_id=int(instance_id),
                       extras=dict(extras or {}))
    return validate_decisions(policy.on_store(chunks, ctx), len(chunks),
                              known=MP_TARGETS)


def apply_mp_hit_policy(policy, obj_keys):
    """L2-hit path (prefetch L2->L1): per-chunk (retain_in_l1, priority).
    Called with the keys about to be loaded into L1; no request identity is
    available at this layer (ctx.req_id empty, instance_id -1)."""
    from testbed.store_policy import ChunkInfo, StoreContext, validate_hit_decisions

    cs = _mp_chunk_size
    chunks = [ChunkInfo(key=k.chunk_hash.hex(), start=i * cs, end=(i + 1) * cs,
                        index=i)
              for i, k in enumerate(obj_keys)]
    ctx = StoreContext(req_id="", total_tokens=len(obj_keys) * cs)
    return validate_hit_decisions(policy.on_hit(chunks, ctx), len(chunks))


def mp_record_hit_retentions(keys, decisions) -> None:
    """Record retained prefetch loads: they re-enter L1 as {"l1"} (the store
    controller must NOT re-store them to L2 — that's where they came from)
    with the priority on_hit assigned."""
    with _mp_lock:
        for k, (retain, prio) in zip(keys, decisions):
            if retain:
                _mp_pending[k] = frozenset(("l1",))
                _mp_priority[k] = prio
        _evict_overflow(_mp_pending)
        _evict_overflow(_mp_priority)


def _evict_overflow(d: dict) -> None:
    while len(d) > _MP_CAP:
        d.pop(next(iter(d)))


def mp_record_decisions(keys, decisions) -> None:
    """Record (targets, priority) for keys that WILL be stored (nonempty targets),
    all groups. Targets are popped by the store controller; priorities persist for
    the key's L1 lifetime (consumed by the priority eviction policy)."""
    with _mp_lock:
        for k, (targets, prio) in zip(keys, decisions):
            _mp_pending[k] = frozenset(targets)
            _mp_priority[k] = prio
        _evict_overflow(_mp_pending)
        _evict_overflow(_mp_priority)


def mp_take_store_decision(key):
    """Pop and return the recorded targets for `key` (None if unrecorded). Keys
    without "l1" are queued for L1 deletion after their L2 store completes."""
    with _mp_lock:
        targets = _mp_pending.pop(key, None)
        if targets is not None and "l1" not in targets:
            _mp_l1_drop[key] = True
            _evict_overflow(_mp_l1_drop)
    return targets


def mp_take_l1_drop(key) -> bool:
    """Pop the drop-from-L1 mark for `key` (set by an {"l2"}-only decision)."""
    with _mp_lock:
        return _mp_l1_drop.pop(key, False)


def mp_priority_of(key) -> float:
    """The priority the external policy attached to `key` at store time
    (0.0 for keys it never saw: hook inactive, prefetch loads, ...)."""
    with _mp_lock:
        return _mp_priority.get(key, 0.0)


def mp_forget_priorities(keys) -> None:
    """Drop priority records for keys leaving L1 (eviction / deletion)."""
    with _mp_lock:
        for k in keys:
            _mp_priority.pop(k, None)


def get_store_policy():
    """Return the policy object (with .on_store) or None. Never raises at call site
    beyond the first load: a broken ref should fail LOUDLY at startup, not silently
    fall back to stock behavior mid-experiment."""
    global _policy, _loaded
    if not _loaded:
        _loaded = True
        ref = os.environ.get(ENV)
        if ref:
            mod_name, _, attr = ref.partition(":")
            mod = importlib.import_module(mod_name)
            _policy = getattr(mod, attr) if attr else getattr(mod, "MyPolicy")
            if isinstance(_policy, type):
                _policy = _policy()            # a class was referenced: instantiate it
            if not hasattr(_policy, "on_store"):
                raise TypeError(f"{ref} has no on_store()")
            logger.info("Store policy loaded from %s: %s", ref, _policy)
    return _policy
