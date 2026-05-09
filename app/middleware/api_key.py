"""
app/routers/api_keys.py — /api-keys

Create, list, and revoke API keys for external integrations.
JWT auth only — only workspace owners can manage keys.

Key format: cm_live_{48 hex chars}
Security: only SHA-256 hash stored — raw key shown once and never again.

Supabase table required (handle migration manually):
    CREATE TABLE api_keys (
        id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        workspace_id uuid NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
        name         text NOT NULL,
        key_hash     text NOT NULL UNIQUE,
        key_prefix   text NOT NULL,
        is_active    bool NOT NULL DEFAULT true,
        created_at   timestamptz DEFAULT now(),
        last_used_at timestamptz,
        created_by   uuid REFERENCES profiles(id)
    );
"""

import hashlib
import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.clients.supabase import get_supabase
from app.middleware.auth import AuthContext, get_auth_context

router = APIRouter(prefix="/api-keys", tags=["api-keys"])
logger = logging.getLogger(__name__)

_KEY_PREFIX = "cm_live_"


# ── Models ─────────────────────────────────────────────────────────────────────

class CreateKeyRequest(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Human-readable label for this key",
        examples=["Website chatbot", "iOS app", "Partner integration"],
    )
    allowed_domains: list[str] | None = Field(
        default=None,
        description="Optional list of allowed domains (e.g., ['example.com', '*.example.com']). Null = no restrictions.",
        examples=[["example.com", "www.example.com"], ["*.example.com"]],
    )


class CreateKeyResponse(BaseModel):
    id: str = Field(description="API key UUID", examples=["3f6e8a21-..."])
    name: str = Field(description="Name you gave this key", examples=["Website chatbot"])
    key: str = Field(
        description="**The full API key. Copy it now — it will NEVER be shown again.**",
        examples=["cm_live_a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6"],
    )
    key_prefix: str = Field(
        description="First 16 chars — shown in the dashboard key list",
        examples=["cm_live_a1b2c3d4"],
    )
    created_at: str = Field(examples=["2026-03-18T10:00:00+00:00"])


class ApiKeyListItem(BaseModel):
    id: str = Field(examples=["3f6e8a21-..."])
    name: str = Field(examples=["Website chatbot"])
    key_prefix: str = Field(examples=["cm_live_a1b2c3d4"])
    is_active: bool = Field(examples=[True])
    created_at: str = Field(examples=["2026-03-18T10:00:00+00:00"])
    last_used_at: str | None = Field(default=None, examples=["2026-03-18T12:34:56+00:00"])
    allowed_domains: list[str] | None = Field(
        default=None,
        description="List of allowed domains. Null means no restrictions.",
        examples=[["example.com"], ["*.example.com"]],
    )


class RevokeKeyResponse(BaseModel):
    id: str = Field(examples=["3f6e8a21-..."])
    revoked: bool = Field(examples=[True])


class ErrorResponse(BaseModel):
    detail: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=CreateKeyResponse,
    status_code=201,
    summary="Create an API key",
    response_description="The new key — raw value shown once only",
    responses={
        201: {
            "description": "Key created. **Copy the `key` field immediately — it cannot be recovered.**",
            "content": {
                "application/json": {
                    "example": {
                        "id": "3f6e8a21-b2c3-4d5e-f6a7-890123456789",
                        "name": "Website chatbot",
                        "key": "cm_live_a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6",
                        "key_prefix": "cm_live_a1b2c3d4",
                        "created_at": "2026-03-18T10:00:00+00:00",
                    }
                }
            },
        },
        403: {
            "description": "Not a JWT-authenticated user (API key callers cannot create keys)",
            "content": {"application/json": {"example": {"detail": "Only dashboard users (JWT auth) can create API keys"}}},
        },
    },
)
async def create_api_key(
    body: CreateKeyRequest,
    auth: AuthContext = Depends(get_auth_context),
) -> CreateKeyResponse:
    """
    Create a new API key for your workspace.

    **JWT auth only** — you must be logged in via the dashboard. API key callers cannot create keys.

    The raw key is returned **exactly once** in the `key` field.
    It is not stored on our side — only a SHA-256 hash is kept.
    If you lose it, revoke the key and create a new one.

    **Dashboard usage:** show the key in a modal with a "Copy" button.
    Warn the user clearly that it cannot be retrieved again.
    """
    if auth.auth_type != "jwt":
        raise HTTPException(
            status_code=403,
            detail="Only dashboard users (JWT auth) can create API keys",
        )

    raw_key = _KEY_PREFIX + os.urandom(24).hex()
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    key_prefix = raw_key[:16]

    try:
        supabase = get_supabase()
        insert_data = {
            "workspace_id": auth.workspace_id,
            "name": body.name.strip(),
            "key_hash": key_hash,
            "key_prefix": key_prefix,
            "is_active": True,
            "created_by": auth.identity_id,
        }
        
        # Add allowed_domains if provided
        if body.allowed_domains is not None:
            insert_data["allowed_domains"] = body.allowed_domains
        
        result = (
            await supabase.table("api_keys")
            .insert(insert_data)
            .execute()
        )
    except Exception as exc:
        logger.error("Failed to create API key: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create API key")

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create API key")

    row = result.data[0]
    logger.info("API key created: id=%s workspace=%s name=%s", row["id"], auth.workspace_id, body.name)

    return CreateKeyResponse(
        id=row["id"],
        name=row["name"],
        key=raw_key,
        key_prefix=key_prefix,
        created_at=str(row["created_at"]),
    )


@router.get(
    "",
    response_model=list[ApiKeyListItem],
    summary="List API keys",
    response_description="All keys for your workspace (metadata only — no raw keys)",
    responses={
        200: {
            "description": "List of keys. Raw key and hash are never returned.",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": "3f6e8a21-b2c3-4d5e-f6a7-890123456789",
                            "name": "Website chatbot",
                            "key_prefix": "cm_live_a1b2c3d4",
                            "is_active": True,
                            "created_at": "2026-03-18T10:00:00+00:00",
                            "last_used_at": "2026-03-18T12:34:56+00:00",
                        }
                    ]
                }
            },
        },
        401: {"description": "No credentials", "content": {"application/json": {"example": {"detail": "Authentication required."}}}},
    },
)
async def list_api_keys(
    auth: AuthContext = Depends(get_auth_context),
) -> list[ApiKeyListItem]:
    """
    List all API keys for your workspace.

    Returns metadata only — the raw key and hash are **never** returned after creation.
    Use `key_prefix` to identify which key is which in the dashboard.

    Both JWT and API key callers can list keys (workspace-scoped — you only see your own).
    """
    try:
        supabase = get_supabase()
        result = (
            await supabase.table("api_keys")
            .select("id, name, key_prefix, is_active, created_at, last_used_at, allowed_domains")
            .eq("workspace_id", auth.workspace_id)
            .order("created_at", desc=True)
            .execute()
        )
    except Exception as exc:
        logger.error("Failed to list API keys: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to list API keys")

    return [
        ApiKeyListItem(
            id=row["id"],
            name=row["name"],
            key_prefix=row["key_prefix"],
            is_active=row["is_active"],
            created_at=str(row["created_at"]),
            last_used_at=str(row["last_used_at"]) if row.get("last_used_at") else None,
            allowed_domains=row.get("allowed_domains"),
        )
        for row in (result.data or [])
    ]


@router.delete(
    "/{key_id}",
    response_model=RevokeKeyResponse,
    summary="Revoke an API key",
    response_description="Confirmation that the key has been revoked",
    responses={
        200: {
            "description": "Key revoked. Takes effect within 5 minutes (Redis TTL).",
            "content": {
                "application/json": {
                    "example": {"id": "3f6e8a21-b2c3-4d5e-f6a7-890123456789", "revoked": True}
                }
            },
        },
        403: {
            "description": "Not a JWT user, or key belongs to a different workspace",
            "content": {"application/json": {"example": {"detail": "Not authorised to revoke this key"}}},
        },
        404: {
            "description": "Key not found",
            "content": {"application/json": {"example": {"detail": "API key not found"}}},
        },
    },
)
async def revoke_api_key(
    key_id: str,
    auth: AuthContext = Depends(get_auth_context),
) -> RevokeKeyResponse:
    """
    Revoke an API key by setting `is_active = false`.

    **JWT auth only.** You can only revoke keys belonging to your workspace.

    The key stops working within **5 minutes** (Redis cache TTL).
    The operation is idempotent — revoking an already-revoked key returns `revoked: true`.
    """
    if auth.auth_type != "jwt":
        raise HTTPException(
            status_code=403,
            detail="Only dashboard users (JWT auth) can revoke API keys",
        )

    try:
        supabase = get_supabase()
        check = (
            await supabase.table("api_keys")
            .select("id, workspace_id, is_active")
            .eq("id", key_id)
            .single()
            .execute()
        )
    except Exception:
        raise HTTPException(status_code=404, detail="API key not found")

    if not check.data:
        raise HTTPException(status_code=404, detail="API key not found")

    if check.data["workspace_id"] != auth.workspace_id:
        raise HTTPException(status_code=403, detail="Not authorised to revoke this key")

    if not check.data["is_active"]:
        return RevokeKeyResponse(id=key_id, revoked=True)

    try:
        supabase = get_supabase()
        await (
            supabase.table("api_keys")
            .update({"is_active": False})
            .eq("id", key_id)
            .execute()
        )
    except Exception as exc:
        logger.error("Failed to revoke API key %s: %s", key_id, exc)
        raise HTTPException(status_code=500, detail="Failed to revoke API key")

    logger.info("API key revoked: id=%s workspace=%s", key_id, auth.workspace_id)
    return RevokeKeyResponse(id=key_id, revoked=True)


class UpdateKeyRequest(BaseModel):
    allowed_domains: list[str] | None = Field(
        default=None,
        description="List of allowed domains. Empty list or null means no restrictions.",
        examples=[["example.com", "*.example.com"]],
    )


class UpdateKeyResponse(BaseModel):
    id: str
    name: str
    key_prefix: str
    is_active: bool
    allowed_domains: list[str] | None
    created_at: str
    last_used_at: str | None


@router.patch(
    "/{key_id}",
    response_model=UpdateKeyResponse,
    summary="Update an API key",
    response_description="Updated key details",
    responses={
        200: {
            "description": "Key updated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "3f6e8a21-b2c3-4d5e-f6a7-890123456789",
                        "name": "Website chatbot",
                        "key_prefix": "cm_live_a1b2c3d4",
                        "is_active": True,
                        "allowed_domains": ["example.com", "*.example.com"],
                        "created_at": "2026-03-18T10:00:00+00:00",
                        "last_used_at": None,
                    }
                }
            },
        },
        403: {
            "description": "Not a JWT user, or key belongs to a different workspace",
            "content": {"application/json": {"example": {"detail": "Not authorised to update this key"}}},
        },
        404: {
            "description": "Key not found",
            "content": {"application/json": {"example": {"detail": "API key not found"}}},
        },
    },
)
async def update_api_key(
    key_id: str,
    body: UpdateKeyRequest,
    auth: AuthContext = Depends(get_auth_context),
) -> UpdateKeyResponse:
    """
    Update an API key's configuration (e.g., allowed domains).

    **JWT auth only.** You can only update keys belonging to your workspace.
    
    Use this to restrict where the API key can be used (e.g., only allow on your domain).
    """
    if auth.auth_type != "jwt":
        raise HTTPException(
            status_code=403,
            detail="Only dashboard users (JWT auth) can update API keys",
        )

    try:
        supabase = get_supabase()
        check = (
            await supabase.table("api_keys")
            .select("id, workspace_id, name, key_prefix, is_active, created_at, last_used_at, allowed_domains")
            .eq("id", key_id)
            .single()
            .execute()
        )
    except Exception:
        raise HTTPException(status_code=404, detail="API key not found")

    if not check.data:
        raise HTTPException(status_code=404, detail="API key not found")

    if check.data["workspace_id"] != auth.workspace_id:
        raise HTTPException(status_code=403, detail="Not authorised to update this key")

    # Build update data
    update_data: dict = {}
    if body.allowed_domains is not None:
        update_data["allowed_domains"] = body.allowed_domains if body.allowed_domains else None

    if not update_data:
        # Nothing to update, return current state
        return UpdateKeyResponse(
            id=check.data["id"],
            name=check.data["name"],
            key_prefix=check.data["key_prefix"],
            is_active=check.data["is_active"],
            allowed_domains=check.data.get("allowed_domains"),
            created_at=str(check.data["created_at"]),
            last_used_at=str(check.data["last_used_at"]) if check.data.get("last_used_at") else None,
        )

    try:
        supabase = get_supabase()
        result = (
            await supabase.table("api_keys")
            .update(update_data)
            .eq("id", key_id)
            .execute()
        )
    except Exception as exc:
        logger.error("Failed to update API key %s: %s", key_id, exc)
        raise HTTPException(status_code=500, detail="Failed to update API key")

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to update API key")

    row = result.data[0]
    
    logger.info("API key updated: id=%s workspace=%s domains=%s", key_id, auth.workspace_id, body.allowed_domains)
    
    return UpdateKeyResponse(
        id=row["id"],
        name=row["name"],
        key_prefix=row["key_prefix"],
        is_active=row["is_active"],
        allowed_domains=row.get("allowed_domains"),
        created_at=str(row["created_at"]),
        last_used_at=str(row["last_used_at"]) if row.get("last_used_at") else None,
    )