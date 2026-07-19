# SPDX-License-Identifier: Apache-2.0
"""autoresearch patch: opt-in per-request KV event ledger for the MP cache server.

When the environment variable ``LMCACHE_REQUEST_LOG_PATH`` is set, the server
appends one JSON line per COMPLETED store / retrieve operation, keyed by the
vLLM request id — the per-request ledger behind the harness's aggregate
store/hit/retrieve counters. When unset (the default), both record functions
return after a single cached None-check and the stock path is untouched.

Line schema (``ev`` discriminates; ``t`` is unix seconds at completion):

    {"ev": "store", "t": ..., "req": <request_id>, "inst": <instance_id>,
     "offered_tokens": N,     # the miss segment the replica offered (chunk-aligned)
     "stored_tokens": N,      # tokens actually committed (0 = all skipped/failed)
     "sec": <op seconds>,
     "l1_chunks": N, "l2_chunks": N, "skip_chunks": N}
                              # the D4 policy's DECLARED targets per chunk —
                              # present only when the policy hook is active;
                              # decision vs effect can differ (e.g. L1 full)

    {"ev": "retrieve", "t": ..., "req": ..., "inst": ...,
     "retrieved_tokens": N, "sec": ...}

Failed / fail-closed operations (block-id underflow, transfer exception) return
early and are NOT recorded — the server log carries those.

Overhead when enabled: one json.dumps + one lock-guarded line-buffered write
per operation (microseconds), against operations that move megabytes over
PCIe (milliseconds). store is off the TTFT critical path entirely; retrieve
already emits an INFO log line per call on the same path.
"""

from __future__ import annotations

# Standard
import json
import os
import threading
import time
from typing import List, Optional, Set, TextIO, Tuple

_lock = threading.Lock()
_handle: Optional[TextIO] = None
_checked = False


def _get_handle() -> Optional[TextIO]:
    """The ledger file handle, opened lazily on first use.

    Returns:
        A line-buffered append handle when LMCACHE_REQUEST_LOG_PATH is set,
        None otherwise (recording disabled).
    """
    global _handle, _checked
    if not _checked:
        with _lock:
            if not _checked:
                path = os.environ.get("LMCACHE_REQUEST_LOG_PATH", "")
                if path:
                    _handle = open(path, "a", buffering=1)
                _checked = True
    return _handle


def _emit(record: dict) -> None:
    handle = _get_handle()
    if handle is None:
        return
    line = json.dumps(record, separators=(",", ":")) + "\n"
    with _lock:
        handle.write(line)


def record_store(
    request_id: str,
    instance_id: int,
    offered_tokens: int,
    stored_tokens: int,
    seconds: float,
    decisions: Optional[List[Tuple[Set[str], float]]],
) -> None:
    """Record one completed store operation.

    Args:
        request_id: The vLLM request id the KV belongs to.
        instance_id: The replica (MP client) that produced the KV.
        offered_tokens: Chunk-aligned tokens of the offered miss segment.
        stored_tokens: Tokens actually committed (0 when every chunk was
            skipped by the policy or dropped by a full tier).
        seconds: Wall time of the store operation.
        decisions: The D4 policy's per-chunk (targets, priority) list, or
            None when the policy hook is inactive.
    """
    if _get_handle() is None:
        return
    record = {
        "ev": "store",
        "t": time.time(),
        "req": request_id,
        "inst": instance_id,
        "offered_tokens": offered_tokens,
        "stored_tokens": stored_tokens,
        "sec": round(seconds, 6),
    }
    if decisions is not None:
        l1_chunks = l2_chunks = skip_chunks = 0
        for targets, _priority in decisions:
            if not targets:
                skip_chunks += 1
                continue
            if "l1" in targets:
                l1_chunks += 1
            if "l2" in targets:
                l2_chunks += 1
        record["l1_chunks"] = l1_chunks
        record["l2_chunks"] = l2_chunks
        record["skip_chunks"] = skip_chunks
    _emit(record)


def record_retrieve(
    request_id: str,
    instance_id: int,
    retrieved_tokens: int,
    seconds: float,
) -> None:
    """Record one completed retrieve operation.

    Args:
        request_id: The vLLM request id being served.
        instance_id: The replica (MP client) loading the KV.
        retrieved_tokens: Chunk-aligned tokens loaded from the shared pool.
        seconds: Wall time of the retrieve operation.
    """
    if _get_handle() is None:
        return
    _emit(
        {
            "ev": "retrieve",
            "t": time.time(),
            "req": request_id,
            "inst": instance_id,
            "retrieved_tokens": retrieved_tokens,
            "sec": round(seconds, 6),
        }
    )
