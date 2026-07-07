# SPDX-License-Identifier: Apache-2.0
# rc-testbed patch: pluggable KV-store scheduling policy (the D4 decision surface).
#
# If LMCACHE_STORE_POLICY_REF="pkg.module:attr" is set, that object is loaded once per
# process (PYTHONPATH must make it importable) and its on_store(chunks, ctx) decides,
# per chunk, which backends receive the KV ("cpu"/"disk"/"remote"; empty = skip) and a
# priority. Unset env -> hook inactive -> stock write-through fan-out.
"""Loader for the external KV-store policy (see rc-autoresearch real/store_policy.py)."""

# Standard
import importlib
import os

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

ENV = "LMCACHE_STORE_POLICY_REF"

# backend registry name -> policy-facing short target name
BACKEND_TARGET = {
    "LocalCPUBackend": "cpu",
    "LocalDiskBackend": "disk",
    "RemoteBackend": "remote",
}

_policy = None
_loaded = False


def apply_store_policy(policy, chunk_list, req_id, total_tokens):
    """chunk_list = [(start, end, CacheEngineKey), ...] in request order.
    Returns per-chunk target sets (subset of {"cpu","disk","remote"}; empty = skip)."""
    # Imported here: resolvable via the same PYTHONPATH that made the policy loadable.
    from real.store_policy import ChunkInfo, StoreContext, validate_decisions

    chunks = [ChunkInfo(key=key.chunk_hash, start=start, end=end, index=i)
              for i, (start, end, key) in enumerate(chunk_list)]
    ctx = StoreContext(req_id=str(req_id), total_tokens=total_tokens)
    decisions = validate_decisions(policy.on_store(chunks, ctx), len(chunks))
    return [targets for targets, _prio in decisions]


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
            if not hasattr(_policy, "on_store"):
                raise TypeError(f"{ref} has no on_store()")
            logger.info("Store policy loaded from %s: %s", ref, _policy)
    return _policy
