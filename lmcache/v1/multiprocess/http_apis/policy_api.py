# SPDX-License-Identifier: Apache-2.0
"""Policy-management HTTP endpoint on the MP server (node-local).

``POST /policy/reload`` re-imports the external KV-store policy module
(``LMCACHE_STORE_POLICY_REF``) and swaps the live object in place. This is the
hot-swap half of a persistent evaluation environment: the policy file is edited
between experiment turns while the server (and its pinned L1 pool) stays up.
"""

# Standard
from http import HTTPStatus
import asyncio

# Third Party
from fastapi import APIRouter, HTTPException, Request

# First Party
from lmcache.logging import init_logger
from lmcache.v1.store_policy_hook import reload_store_policy

logger = init_logger(__name__)

router = APIRouter()


@router.post("/policy/reload", response_model=None)
async def policy_reload(request: Request) -> dict[str, object]:
    """Re-import and swap the external store policy.

    Responses:
        200: ``{"status": "ok", "policy": "<repr of the new object>"}``.
        409: no policy ref configured, or the reloaded object is invalid
            (the previous policy stays active).
    """
    try:
        new_repr = await asyncio.to_thread(reload_store_policy)
    except (RuntimeError, TypeError, ImportError, AttributeError) as exc:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT,
            detail=f"policy reload failed (previous policy kept): {exc}",
        ) from None
    return {"status": "ok", "policy": new_repr}
