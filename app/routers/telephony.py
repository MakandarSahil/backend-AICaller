"""
app/routers/telephony.py — /telephony/providers

Manage Bring-Your-Own (BYO) telephony provider connections.
Users can connect their own Twilio account and use their own phone numbers.

Security:
- Credentials stored in Supabase Vault (encrypted at rest)
- Only vault_secret_id stored in regular tables
- Service role required to access Vault
- Credentials NEVER returned to frontend
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.clients.supabase import get_supabase
from app.middleware.auth import AuthContext, get_auth_context

router = APIRouter(prefix="/telephony", tags=["telephony"])
logger = logging.getLogger(__name__)


# ── Models ────────────────────────────────────────────────────────────────────


class ConnectProviderRequest(BaseModel):
    """Request to connect a telephony provider (Twilio)."""
    
    provider: str = Field(
        default="twilio",
        description="Provider type. Currently only 'twilio' supported.",
        examples=["twilio"],
    )
    display_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Human-readable name for this connection",
        examples=["My Twilio Account", "Production Twilio"],
    )
    account_sid: str = Field(
        ...,
        min_length=34,
        description="Twilio Account SID (starts with AC...)",
        examples=["ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"],
    )
    auth_token: str = Field(
        ...,
        min_length=32,
        description="Twilio Auth Token (kept secret, stored in Vault)",
        examples=["xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"],
    )


class ProviderResponse(BaseModel):
    """Provider info returned to clients (NO credentials)."""
    
    id: str
    provider: str
    display_name: str
    is_active: bool
    is_verified: bool
    created_at: str
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "provider": "twilio",
                "display_name": "My Twilio Account",
                "is_active": True,
                "is_verified": True,
                "created_at": "2026-05-09T10:00:00+00:00",
            }
        }
    }


class VerifyProviderResponse(BaseModel):
    """Verification test result."""
    
    success: bool
    message: str
    account_friendly_name: Optional[str] = None
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "message": "Credentials verified successfully",
                "account_friendly_name": "My Project",
            }
        }
    }


class WebhookInstructionsResponse(BaseModel):
    """Webhook configuration instructions for user."""
    
    provider: str
    webhook_url_template: str
    instructions: list[str]
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "provider": "twilio",
                "webhook_url_template": "https://api.callmind.com/voice?agent_id={agent_id}",
                "instructions": [
                    "1. Log in to your Twilio Console",
                    "2. Go to Phone Numbers → Manage → Active Numbers",
                    "3. Click on your phone number",
                    "4. Under 'Voice & Fax', set:",
                    "   - Accept incoming: Webhooks, URLs, etc.",
                    "   - A call comes in: Webhook",
                    "   - URL: https://api.callmind.com/voice?agent_id=YOUR_AGENT_ID",
                    "   - Method: HTTP POST",
                    "5. Click Save",
                ],
            }
        }
    }


# ── Routes ─────────────────────────────────────────────────────────────────────


@router.get(
    "/providers",
    response_model=list[ProviderResponse],
    summary="List connected telephony providers",
    response_description="All BYO telephony providers for this workspace (no credentials)",
)
async def list_providers(
    auth: AuthContext = Depends(get_auth_context),
) -> list[ProviderResponse]:
    """
    List all connected telephony providers for the current workspace.
    
    Returns metadata only - credentials are NEVER included.
    """
    try:
        supabase = get_supabase()
        result = (
            await supabase.table("workspace_telephony_providers")
            .select("id, provider, display_name, is_active, is_verified, created_at")
            .eq("workspace_id", auth.workspace_id)
            .order("created_at", desc=True)
            .execute()
        )
    except Exception as exc:
        logger.error("Failed to list telephony providers: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to list providers")
    
    return [
        ProviderResponse(
            id=row["id"],
            provider=row["provider"],
            display_name=row["display_name"],
            is_active=row["is_active"],
            is_verified=row["is_verified"],
            created_at=str(row["created_at"]),
        )
        for row in (result.data or [])
    ]


@router.post(
    "/providers",
    response_model=ProviderResponse,
    status_code=201,
    summary="Connect a telephony provider",
    response_description="Connected provider info (no credentials)",
)
async def connect_provider(
    body: ConnectProviderRequest,
    auth: AuthContext = Depends(get_auth_context),
) -> ProviderResponse:
    """
    Connect a new telephony provider (BYO Twilio).
    
    **Security:** Credentials are stored encrypted in Supabase Vault.
    Only a vault reference ID is stored in the regular database.
    Credentials are NEVER returned to the frontend.
    
    **Verification:** Credentials are tested immediately. If invalid,
    the connection is rejected.
    """
    if auth.auth_type != "jwt":
        raise HTTPException(
            status_code=403,
            detail="Only dashboard users can connect telephony providers",
        )
    
    # Validate Twilio credentials first
    is_valid, account_info = await _verify_twilio_credentials(
        body.account_sid, body.auth_token
    )
    
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail="Invalid Twilio credentials. Please check your Account SID and Auth Token.",
        )
    
    # Store credentials in Vault
    try:
        vault_secret_id = await _store_credentials_in_vault(
            workspace_id=auth.workspace_id,
            provider=body.provider,
            account_sid=body.account_sid,
            auth_token=body.auth_token,
        )
    except Exception as exc:
        logger.error("Failed to store credentials in Vault: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to securely store credentials. Please try again.",
        )
    
    # Create provider record
    try:
        supabase = get_supabase()
        result = (
            await supabase.table("workspace_telephony_providers")
            .insert({
                "workspace_id": auth.workspace_id,
                "provider": body.provider,
                "display_name": body.display_name.strip(),
                "vault_secret_id": vault_secret_id,
                "is_active": True,
                "is_verified": True,
                "provider_type": "own",
            })
            .execute()
        )
    except Exception as exc:
        # Try to clean up vault secret
        await _delete_vault_secret(vault_secret_id)
        logger.error("Failed to create provider record: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create provider")
    
    if not result.data:
        await _delete_vault_secret(vault_secret_id)
        raise HTTPException(status_code=500, detail="Failed to create provider")
    
    row = result.data[0]
    logger.info(
        "Telephony provider connected: id=%s workspace=%s provider=%s",
        row["id"],
        auth.workspace_id,
        body.provider,
    )
    
    return ProviderResponse(
        id=row["id"],
        provider=row["provider"],
        display_name=row["display_name"],
        is_active=row["is_active"],
        is_verified=row["is_verified"],
        created_at=str(row["created_at"]),
    )


@router.post(
    "/providers/{provider_id}/verify",
    response_model=VerifyProviderResponse,
    summary="Verify provider credentials",
    response_description="Verification test result",
)
async def verify_provider(
    provider_id: str,
    auth: AuthContext = Depends(get_auth_context),
) -> VerifyProviderResponse:
    """
    Test that stored provider credentials are still valid.
    
    Retrieves credentials from Vault and makes a test API call.
    """
    # Get credentials from Vault
    creds = await _get_credentials_from_vault(provider_id, auth.workspace_id)
    if not creds:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    is_valid, account_info = await _verify_twilio_credentials(
        creds["account_sid"], creds["auth_token"]
    )
    
    if is_valid:
        return VerifyProviderResponse(
            success=True,
            message="Credentials verified successfully",
            account_friendly_name=account_info.get("friendly_name") if account_info else None,
        )
    else:
        return VerifyProviderResponse(
            success=False,
            message="Credentials are invalid or expired. Please reconnect your provider.",
        )


@router.delete(
    "/providers/{provider_id}",
    status_code=204,
    summary="Disconnect a telephony provider",
)
async def disconnect_provider(
    provider_id: str,
    auth: AuthContext = Depends(get_auth_context),
):
    """
    Disconnect a telephony provider and delete stored credentials.
    
    **Warning:** Any phone numbers using this provider will stop working.
    """
    if auth.auth_type != "jwt":
        raise HTTPException(
            status_code=403,
            detail="Only dashboard users can disconnect providers",
        )
    
    # Get vault_secret_id first
    try:
        supabase = get_supabase()
        check = (
            await supabase.table("workspace_telephony_providers")
            .select("id, vault_secret_id, workspace_id")
            .eq("id", provider_id)
            .single()
            .execute()
        )
    except Exception:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    if not check.data or check.data["workspace_id"] != auth.workspace_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Delete from database
    try:
        await (
            supabase.table("workspace_telephony_providers")
            .delete()
            .eq("id", provider_id)
            .execute()
        )
    except Exception as exc:
        logger.error("Failed to delete provider: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to disconnect provider")
    
    # Clean up vault secret
    vault_secret_id = check.data.get("vault_secret_id")
    if vault_secret_id:
        await _delete_vault_secret(vault_secret_id)
    
    logger.info("Telephony provider disconnected: id=%s workspace=%s", provider_id, auth.workspace_id)


@router.get(
    "/webhook-instructions",
    response_model=WebhookInstructionsResponse,
    summary="Get webhook configuration instructions",
)
async def get_webhook_instructions() -> WebhookInstructionsResponse:
    """
    Get instructions for configuring webhooks on your Twilio number.
    
    This endpoint is public (no auth required) so it can be shown
    before the user connects their provider.
    """
    from app.config import get_settings
    settings = get_settings()
    
    base_url = settings.public_url or "https://api.callmind.com"
    
    return WebhookInstructionsResponse(
        provider="twilio",
        webhook_url_template=f"{base_url}/voice?agent_id={{agent_id}}",
        instructions=[
            "1. Log in to your Twilio Console (console.twilio.com)",
            "2. Go to Phone Numbers → Manage → Active Numbers",
            "3. Click on the phone number you want to configure",
            "4. Scroll down to 'Voice & Fax' section",
            "5. Under 'A call comes in':",
            "   - Select: Webhook",
            f"   - URL: {base_url}/voice?agent_id=YOUR_AGENT_ID",
            "   - Method: HTTP POST",
            "6. Click 'Save Configuration'",
            "7. Repeat for each number you want to use with CallMind",
        ],
    )


# ── Encryption Helper ─────────────────────────────────────────────────────────

# Import settings here for encryption functions
from app.config import get_settings as _get_settings

# Initialize Fernet cipher with encryption key from settings
def _get_cipher():
    """Get Fernet cipher instance."""
    from cryptography.fernet import Fernet
    settings = _get_settings()
    
    if not settings.encryption_key:
        raise ValueError("ENCRYPTION_KEY not set in environment. Generate one with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'")
    
    return Fernet(settings.encryption_key.encode())


async def _encrypt_credentials(data: dict) -> str:
    """Encrypt credentials dictionary and return base64-encoded ciphertext."""
    import json
    from base64 import b64encode
    
    cipher = _get_cipher()
    json_data = json.dumps(data)
    encrypted = cipher.encrypt(json_data.encode())
    return b64encode(encrypted).decode()


async def _decrypt_credentials(ciphertext: str) -> dict:
    """Decrypt base64-encoded ciphertext and return credentials dictionary."""
    import json
    from base64 import b64decode
    
    cipher = _get_cipher()
    encrypted = b64decode(ciphertext.encode())
    decrypted = cipher.decrypt(encrypted)
    return json.loads(decrypted.decode())


# ── Helper Functions ───────────────────────────────────────────────────────────


async def _verify_twilio_credentials(account_sid: str, auth_token: str) -> tuple[bool, Optional[dict]]:
    """Verify Twilio credentials by making a test API call."""
    try:
        from twilio.rest import Client
        
        client = Client(account_sid, auth_token)
        # Fetch account info as a simple test
        account = client.api.accounts(account_sid).fetch()
        
        return True, {
            "friendly_name": account.friendly_name,
            "status": account.status,
        }
    except Exception as exc:
        logger.warning("Twilio credential verification failed: %s", exc)
        return False, None


async def _store_credentials_in_vault(
    workspace_id: str,
    provider: str,
    account_sid: str,
    auth_token: str,
) -> str:
    """
    Store credentials securely using Fernet encryption.
    
    Credentials are encrypted and stored in the database.
    Returns a secret ID (UUID format) that can be used to retrieve credentials later.
    """
    supabase = get_supabase()
    
    # Generate UUID for secret ID (must match database UUID type)
    secret_id = str(uuid.uuid4())
    
    # Encrypt credentials
    try:
        encrypted_data = await _encrypt_credentials({
            "provider": provider,
            "account_sid": account_sid,
            "auth_token": auth_token,
            "workspace_id": str(workspace_id),
        })
    except ValueError as e:
        logger.error("Encryption failed: %s", e)
        raise HTTPException(status_code=500, detail="Encryption not configured properly")
    
    # Store in database
    result = await supabase.table("provider_secrets").insert({
        "id": secret_id,
        "workspace_id": workspace_id,
        "provider": provider,
        "encrypted_data": encrypted_data,
        "created_at": "now()",
    }).execute()
    
    if not result.data:
        logger.error("Failed to store credentials in database")
        raise Exception("Failed to store credentials")
    
    logger.info("Credentials stored successfully: %s", secret_id)
    return secret_id


async def _get_credentials_from_vault(
    provider_id: str,
    workspace_id: str,
) -> Optional[dict]:
    """
    Retrieve and decrypt credentials from secure storage.
    """
    try:
        supabase = get_supabase()
        
        # Get the vault_secret_id from the provider record
        provider = (
            await supabase.table("workspace_telephony_providers")
            .select("vault_secret_id, workspace_id")
            .eq("id", provider_id)
            .single()
            .execute()
        )
        
        if not provider.data or provider.data["workspace_id"] != workspace_id:
            return None
        
        secret_id = provider.data.get("vault_secret_id")
        if not secret_id:
            return None
        
        # Retrieve encrypted data from database
        result = await supabase.table("provider_secrets").select("encrypted_data").eq("id", secret_id).single().execute()
        
        if not result.data or not result.data.get("encrypted_data"):
            logger.error("No encrypted data found for secret: %s", secret_id)
            return None
        
        # Decrypt credentials
        try:
            credentials = await _decrypt_credentials(result.data["encrypted_data"])
            return {
                "account_sid": credentials.get("account_sid"),
                "auth_token": credentials.get("auth_token"),
            }
        except Exception as decrypt_error:
            logger.error("Failed to decrypt credentials: %s", decrypt_error)
            return None
        
    except Exception as exc:
        logger.error("Error getting provider credentials: %s", exc)
        return None


async def _delete_vault_secret(secret_id: str) -> bool:
    """Delete encrypted credentials from storage."""
    try:
        supabase = get_supabase()
        
        # Delete from database
        await supabase.table("provider_secrets").delete().eq("id", secret_id).execute()
        
        logger.info("Deleted encrypted credentials: %s", secret_id)
        return True
    except Exception as exc:
        logger.error("Failed to delete credentials: %s", exc)
        return False
