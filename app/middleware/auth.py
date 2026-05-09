"""
app/middleware/auth.py

Unified authentication for POST /query and the /api-keys endpoints.

Two auth types, one AuthContext output:

    Supabase JWT  (dashboard owner)
        Header: Authorization: Bearer <supabase_jwt>
        Resolves: profiles.id → workspace_id via workspaces table
        Used by: dashboard chat UI

    API key  (external developer / chatbot widget)
        Header: Authorization: Bearer cm_live_<key>
        OR      X-API-Key: cm_live_<key>
        Resolves: sha256(key) → api_keys row → workspace_id
        Used by: external website chatbots, integrations

Both paths return the same AuthContext dataclass.
Route handlers only ever see AuthContext — never the raw token or auth type
(unless they need to restrict to JWT-only, which api_keys.py does).

Token type detection:
    Tokens starting with "cm_" → API key path
    Everything else             → Supabase JWT path

Redis caching:
    Valid API keys:        key apikey:{sha256}  TTL 5 min
    Valid JWT workspace:   key jwt_ws:{user_id} TTL 5 min
    Auth check costs ~0ms on cache hit.
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from app.clients.redis import get_redis
from app.clients.supabase import get_supabase
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# ── Security scheme objects ────────────────────────────────────────────────────
# These are imported by main.py to register in the OpenAPI schema.
# They are also used as FastAPI dependencies to extract tokens from requests.

# Bearer token — accepts both Supabase JWT and cm_live_ API keys
bearer_scheme = HTTPBearer(
    auto_error=False,
    description="Paste your **Supabase JWT** (dashboard) or **API key** (`cm_live_xxx`) here.",
)

# X-API-Key header — alternative entry point for API keys only
api_key_header_scheme = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="Alternative to Bearer for external developers. Paste `cm_live_xxx` here.",
)

_API_KEY_PREFIX = "cm_"
_AUTH_CACHE_TTL = 300  # 5 minutes


@dataclass
class AuthContext:
    """
    Resolved identity after successful authentication.
    Injected into route handlers via Depends(get_auth_context).

    workspace_id:  the workspace this request operates on behalf of
    auth_type:     "jwt" (dashboard user) or "api_key" (external developer)
    identity_id:   profiles.id for jwt, api_keys.id for api_key
    """
    workspace_id: str
    auth_type: str       # "jwt" | "api_key"
    identity_id: str     # profiles.id or api_keys.id


async def get_auth_context(
    request: Request,
    bearer: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    x_api_key: str | None = Depends(api_key_header_scheme),
) -> AuthContext:
    """
    FastAPI dependency — resolves auth from either JWT or API key.

    Checks in order:
        1. Authorization: Bearer <token>
        2. X-API-Key: <token>

    Token routing:
        Starts with "cm_"  → API key verification
        Anything else      → Supabase JWT verification

    Usage:
        @router.post("/query")
        async def my_route(auth: AuthContext = Depends(get_auth_context)):
            workspace_id = auth.workspace_id

    Raises:
        401 — no credentials provided
        403 — invalid, expired, or revoked credentials
    """
    # Extract raw token from either header
    raw_token: str | None = None
    if bearer:
        raw_token = bearer.credentials
    if not raw_token and x_api_key:
        raw_token = x_api_key.strip()

    if not raw_token:
        raise HTTPException(
            status_code=401,
            detail=(
                "Authentication required. "
                "Provide 'Authorization: Bearer <token>' or 'X-API-Key: <key>'."
            ),
        )

    if raw_token.startswith(_API_KEY_PREFIX):
        logger.info("[AUTH] path=%s method=%s type=api_key", request.url.path, request.method)
        # Get Origin header for domain validation
        origin = request.headers.get("origin") or request.headers.get("Origin") or request.headers.get("Referer")
        return await _verify_api_key(raw_token, origin)
    else:
        logger.info("[AUTH] path=%s method=%s type=jwt", request.url.path, request.method)
        return await _verify_jwt(raw_token)


# ── API key verification ───────────────────────────────────────────────────────

async def _verify_api_key(raw_key: str, origin: str | None = None) -> AuthContext:
    """
    Verify a CallMind API key (cm_live_xxxx format).

    Flow: sha256(key) → Redis cache → api_keys table → is_active check → domain validation
    The raw key is NEVER stored — only the sha256 hash lives in the DB.
    """
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    cache_key = f"apikey:{key_hash}"
    redis = get_redis()

    try:
        cached = await redis.get(cache_key)
        if cached:
            data = json.loads(cached)
            logger.debug("API key cache HIT: prefix=%s", raw_key[:12])
            # Check domain restriction even for cached keys
            if data.get("allowed_domains"):
                if not _is_domain_allowed(origin, data["allowed_domains"]):
                    logger.warning("API key used from unauthorized domain: %s", origin)
                    raise HTTPException(status_code=403, detail="Domain not authorized for this API key")
            return AuthContext(
                workspace_id=data["workspace_id"],
                auth_type="api_key",
                identity_id=data["id"],
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Redis cache read failed for API key: %s", exc)

    try:
        supabase = get_supabase()
        result = (
            await supabase.table("api_keys")
            .select("id, workspace_id, is_active, name, allowed_domains")
            .eq("key_hash", key_hash)
            .single()
            .execute()
        )
    except Exception as exc:
        logger.error("API key DB lookup failed: %s", exc)
        raise HTTPException(status_code=403, detail="Invalid API key")

    if not result.data:
        raise HTTPException(status_code=403, detail="Invalid API key")

    row = result.data
    if not row.get("is_active", False):
        raise HTTPException(status_code=403, detail="API key has been revoked")

    # Check domain restriction
    allowed_domains = row.get("allowed_domains")
    if allowed_domains:
        if not _is_domain_allowed(origin, allowed_domains):
            logger.warning(
                "API key used from unauthorized domain: key=%s domain=%s allowed=%s",
                row["id"], origin, allowed_domains
            )
            raise HTTPException(status_code=403, detail="Domain not authorized for this API key")

    asyncio.create_task(_update_key_last_used(row["id"]))

    try:
        cache_data = {"id": row["id"], "workspace_id": row["workspace_id"]}
        if allowed_domains:
            cache_data["allowed_domains"] = allowed_domains
        await redis.setex(
            cache_key,
            _AUTH_CACHE_TTL,
            json.dumps(cache_data),
        )
    except Exception:
        pass

    logger.debug("API key verified: id=%s workspace=%s", row["id"], row["workspace_id"])
    return AuthContext(
        workspace_id=row["workspace_id"],
        auth_type="api_key",
        identity_id=row["id"],
    )


def _is_domain_allowed(origin: str | None, allowed_domains: list[str]) -> bool:
    """
    Check if the request origin is in the list of allowed domains.
    
    Supports:
    - Exact match: example.com matches example.com
    - Wildcard: *.example.com matches sub.example.com, app.example.com
    - localhost: Always allowed for development
    """
    if not origin:
        # No origin header - allow for now (some clients don't send it)
        return True
    
    # Parse origin to get just the hostname
    try:
        if origin.startswith("http://") or origin.startswith("https://"):
            hostname = origin.split("://")[1].split(":")[0].split("/")[0]
        else:
            hostname = origin
    except Exception:
        return False
    hostname = _normalize_domain(hostname)
    
    # Always allow localhost for development
    if hostname in ("localhost", "127.0.0.1", "::1"):
        return True
    
    for allowed in allowed_domains:
        if not allowed:
            continue
        allowed = _normalize_domain(allowed)
            
        # Exact match
        if hostname == allowed:
            return True
        
        # Wildcard match: *.example.com
        if allowed.startswith("*."):
            suffix = allowed[2:]  # Remove "*."
            if hostname.endswith(f".{suffix}"):
                return True
        
        # Handle www subdomain automatically
        if allowed.startswith("www."):
            without_www = allowed[4:]
            if hostname == without_www or hostname == allowed:
                return True
    
    return False


def _normalize_domain(value: str) -> str:
    """Normalize domain text for comparisons (lowercase, trim, drop trailing dot)."""
    return value.lower().strip().rstrip(".")


async def _update_key_last_used(api_key_id: str) -> None:
    """Fire-and-forget: update last_used_at on the api_keys row."""
    try:
        supabase = get_supabase()
        await (
            supabase.table("api_keys")
            .update({"last_used_at": "now()"})
            .eq("id", api_key_id)
            .execute()
        )
    except Exception as exc:
        logger.warning("Failed to update api_key last_used_at: %s", exc)


# ── Supabase JWT verification ──────────────────────────────────────────────────

async def _verify_jwt(token: str) -> AuthContext:
    """
    Verify a Supabase JWT from a logged-in dashboard user.

    Flow: Decode JWT to extract user_id → Redis cache
          → workspaces WHERE owner_id = user_id

    Note: We extract user_id from JWT claims rather than calling supabase.auth.get_user()
    because the JWT is self-validating (signed by Supabase) and contains user_id in the sub claim.
    """
    try:
        # Decode JWT by splitting and base64-decoding the payload
        # JWT format: header.payload.signature
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid JWT format")
        
        # Add padding if needed
        payload_part = parts[1]
        padding = 4 - len(payload_part) % 4
        if padding != 4:
            payload_part += "=" * padding
        
        import base64
        payload_json = base64.urlsafe_b64decode(payload_part)
        payload = json.loads(payload_json)
        
        user_id = payload.get("sub")  # Supabase JWT has user_id in 'sub' claim
        if not user_id:
            raise ValueError("No 'sub' claim in JWT")
    except Exception as exc:
        logger.warning("JWT verification failed: %s", exc)
        raise HTTPException(status_code=403, detail="Invalid or expired session token")

    if not user_id:
        raise HTTPException(status_code=403, detail="Invalid session token")
    cache_key = f"jwt_ws:{user_id}"
    redis = get_redis()

    try:
        cached = await redis.get(cache_key)
        if cached:
            data = json.loads(cached)
            logger.debug("JWT workspace cache HIT: user=%s", user_id)
            return AuthContext(
                workspace_id=data["workspace_id"],
                auth_type="jwt",
                identity_id=user_id,
            )
    except Exception as exc:
        logger.warning("Redis cache read failed for JWT: %s", exc)

    try:
        supabase = get_supabase()
        ws_result = (
            await supabase.table("workspaces")
            .select("id")
            .eq("owner_id", user_id)
            .single()
            .execute()
        )
    except Exception as exc:
        logger.error("Workspace lookup failed for user %s: %s", user_id, exc)
        raise HTTPException(status_code=403, detail="No workspace found for this account")

    if not ws_result.data:
        raise HTTPException(status_code=403, detail="No workspace found for this account")

    workspace_id = ws_result.data["id"]

    try:
        await redis.setex(
            cache_key,
            _AUTH_CACHE_TTL,
            json.dumps({"workspace_id": workspace_id}),
        )
    except Exception:
        pass

    logger.debug("JWT verified: user=%s workspace=%s", user_id, workspace_id)
    return AuthContext(
        workspace_id=workspace_id,
        auth_type="jwt",
        identity_id=user_id,
    )
