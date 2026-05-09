"""
app/routers/voice.py — /voice

Twilio webhook. Called automatically when a call arrives on a configured number.
Not used by the dashboard or external developers.

BYO Twilio Support:
- Detects if the called number belongs to a connected provider
- Uses provider-specific credentials for webhook validation
- Falls back to platform credentials for platform numbers
"""

import logging
from html import escape
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

from app.clients.supabase import get_supabase
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

router = APIRouter()

_TWIML_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
        <Stream url="{ws_url}">
            <Parameter name="agent_id" value="{agent_id}" />
                        <Parameter name="From" value="{from_number}" />
                        <Parameter name="To" value="{to_number}" />
                        <Parameter name="CallSid" value="{call_sid}" />
        </Stream>
  </Connect>
</Response>"""


@router.api_route(
    "/voice",
    methods=["GET", "POST"],
    summary="Twilio TwiML webhook",
    response_description="TwiML XML that tells Twilio to open a WebSocket stream",
    tags=["telephony"],
    responses={
        200: {
            "description": "TwiML XML returned to Twilio",
            "content": {
                "text/xml": {
                    "example": (
                        '<?xml version="1.0" encoding="UTF-8"?>\n'
                        "<Response>\n"
                        "  <Connect>\n"
                        '    <Stream url="wss://api.callmind.com/call?agent_id=uuid" />\n'
                        "  </Connect>\n"
                        "</Response>"
                    )
                }
            },
        },
        403: {
            "description": "Invalid Twilio signature (production only)",
            "content": {"application/json": {"example": {"detail": "Invalid Twilio signature"}}},
        },
        404: {
            "description": "Agent not found or inactive",
            "content": {"application/json": {"example": {"detail": "Agent not found: abc-123"}}},
        },
    },
)
async def twiml_webhook(
    request: Request,
    agent_id: str = Query(
        ...,
        description="Agent UUID. Set this when configuring the Twilio number webhook URL.",
        examples=["b1c2d3e4-f5a6-7890-abcd-ef1234567890"],
    ),
) -> Response:
    """
    Called by Twilio when a phone call arrives on a configured number.

    **Not called by the dashboard or external developers.**

    Configure your Twilio number's voice webhook URL as:
    ```
    https://api.callmind.com/voice?agent_id={your_agent_uuid}
    ```

    This endpoint validates the Twilio request signature (production only),
    checks the agent is active, and returns TwiML that tells Twilio to open
    a WebSocket stream to `/call?agent_id={uuid}`.
    """
    # Get the called number from form data (available in POST requests)
    form = await request.form() if request.method == "POST" else {}
    to_number = str(form.get("To") or "")
    
    # Determine which credentials to use for validation
    auth_token = settings.twilio_auth_token
    is_byo = False
    
    if to_number:
        # Look up if this is a BYO number
        provider_creds = await _get_provider_credentials_for_number(to_number)
        if provider_creds:
            auth_token = provider_creds["auth_token"]
            is_byo = True
            logger.info("Using BYO Twilio credentials for number=%s", to_number)
    
    # Validate Twilio signature with appropriate credentials
    if settings.is_production and auth_token:
        if not await _validate_twilio_signature(request, auth_token, agent_id):
            logger.warning(
                "Twilio signature FAILED: agent_id=%s ip=%s is_byo=%s",
                agent_id,
                request.client.host if request.client else "unknown",
                is_byo,
            )
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    if not await _agent_exists(agent_id):
        logger.warning("TwiML: unknown/inactive agent_id=%s", agent_id)
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")

    # Use form data already fetched above
    from_number = str(form.get("From") or "")
    call_sid = str(form.get("CallSid") or "")
    if not from_number:
        logger.warning("TwiML webhook missing From: agent_id=%s call_sid=%s", agent_id, call_sid)

    ws_url = _resolve_ws_call_url(request)
    request_host = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    ws_host = urlparse(ws_url).netloc
    if request_host and ws_host and request_host != ws_host:
        logger.warning(
            "TwiML host mismatch: request_host=%s ws_host=%s ws_url=%s",
            request_host,
            ws_host,
            ws_url,
        )

    twiml = _TWIML_TEMPLATE.format(
        ws_url=escape(ws_url, quote=True),
        agent_id=escape(agent_id, quote=True),
        from_number=escape(from_number, quote=True),
        to_number=escape(to_number, quote=True),
        call_sid=escape(call_sid, quote=True),
    )

    logger.info(
        "TwiML served: agent_id=%s from=%s call_sid=%s ws_url=%s",
        agent_id,
        from_number,
        call_sid,
        ws_url,
    )
    return Response(content=twiml, media_type="text/xml")


async def _validate_twilio_signature(request: Request, auth_token: str | None = None, agent_id: str | None = None) -> bool:
    """Validate Twilio request signature using platform or BYO credentials."""
    try:
        from twilio.request_validator import RequestValidator
        
        token = auth_token or settings.twilio_auth_token
        if not token:
            logger.warning("No Twilio auth token available for validation")
            return False
        
        signature = request.headers.get("X-Twilio-Signature", "")
        
        # Twilio signs the public webhook URL with query params.
        # Behind Traefik, request.url can be internal unless forwarded headers are respected.
        # We MUST include the agent_id query param as Twilio signed the full URL.
        forwarded_proto = request.headers.get("x-forwarded-proto")
        forwarded_host = request.headers.get("x-forwarded-host") or request.headers.get("host")
        
        # Build query string - MUST include agent_id as Twilio signs the full URL
        query_parts = []
        if agent_id:
            query_parts.append(f"agent_id={agent_id}")
        # Also include any other query params from the original URL
        if request.url.query and "agent_id" not in str(request.url.query):
            query_parts.append(str(request.url.query))
        
        query = "?" + "&".join(query_parts) if query_parts else ""
        
        if forwarded_proto and forwarded_host:
            url = f"{forwarded_proto}://{forwarded_host}{request.url.path}{query}"
        else:
            url = f"{request.url}{query}"

        params = dict(await request.form()) if request.method == "POST" else {}
        
        logger.debug(
            "Twilio signature validation: url=%s signature=%s params=%s",
            url,
            signature[:20] + "..." if signature else "None",
            list(params.keys())
        )
        
        is_valid = RequestValidator(token).validate(url, params, signature)
        if not is_valid:
            logger.warning(
                "Twilio signature mismatch: url=%s method=%s host=%s fwd_host=%s fwd_proto=%s",
                url,
                request.method,
                request.headers.get("host"),
                request.headers.get("x-forwarded-host"),
                request.headers.get("x-forwarded-proto"),
            )
        return is_valid
    except Exception as exc:
        logger.error("Twilio signature validation error: %s", exc)
        return False


async def _get_provider_credentials_for_number(phone_number: str) -> dict | None:
    """
    Get Twilio credentials for a BYO phone number.
    
    Returns {"account_sid": str, "auth_token": str} if BYO, None if platform number.
    """
    try:
        supabase = get_supabase()
        
        # Look up the phone number
        result = (
            await supabase.table("phone_numbers")
            .select("telephony_provider_id")
            .eq("number", phone_number)
            .eq("is_active", True)
            .maybe_single()
            .execute()
        )
        
        if not result.data:
            logger.debug("Phone number not found: %s", phone_number)
            return None
        
        provider_id = result.data.get("telephony_provider_id")
        if not provider_id:
            # No provider = platform number
            logger.debug("Using platform credentials for number: %s", phone_number)
            return None
        
        # Get provider record
        provider_result = (
            await supabase.table("workspace_telephony_providers")
            .select("vault_secret_id, provider")
            .eq("id", provider_id)
            .eq("is_active", True)
            .single()
            .execute()
        )
        
        if not provider_result.data:
            logger.warning("Provider not found or inactive: %s", provider_id)
            return None
        
        vault_secret_id = provider_result.data.get("vault_secret_id")
        if not vault_secret_id:
            logger.warning("No vault secret for provider: %s", provider_id)
            return None
        
        # Retrieve encrypted credentials from provider_secrets table
        secret_result = await supabase.table("provider_secrets")\
            .select("encrypted_data")\
            .eq("id", vault_secret_id)\
            .single()\
            .execute()
        
        if not secret_result.data or not secret_result.data.get("encrypted_data"):
            logger.error("Failed to retrieve credentials: %s", vault_secret_id)
            return None
        
        # Decrypt credentials
        try:
            credentials = await _decrypt_credentials(secret_result.data["encrypted_data"])
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


async def _decrypt_credentials(ciphertext: str) -> dict:
    """Decrypt base64-encoded ciphertext and return credentials dictionary."""
    import json
    from base64 import b64decode
    from cryptography.fernet import Fernet
    
    cipher = Fernet(settings.encryption_key.encode())
    encrypted = b64decode(ciphertext.encode())
    decrypted = cipher.decrypt(encrypted)
    return json.loads(decrypted.decode())


async def _agent_exists(agent_id: str) -> bool:
    try:
        supabase = get_supabase()
        result = (
            await supabase.table("agents")
            .select("id")
            .eq("id", agent_id)
            .eq("status", "active")
            .single()
            .execute()
        )
        return bool(result.data)
    except Exception as exc:
        logger.error("Agent existence check failed: %s", exc)
        return False


def _resolve_ws_call_url(request: Request) -> str:
    """
    Resolve the Twilio <Stream> URL.

    Priority:
    1) Explicit WS_CALL_URL_OVERRIDE from env
    2) Incoming forwarded host/proto (best for Cloudflare/ngrok tunnels)
    3) Static PUBLIC_URL-derived fallback from settings
    """
    if settings.ws_call_url_override:
        return settings.ws_call_url_override.strip()

    forwarded_host = request.headers.get("x-forwarded-host")
    host = forwarded_host or request.headers.get("host")
    forwarded_proto = request.headers.get("x-forwarded-proto")
    scheme = (forwarded_proto or request.url.scheme or "https").lower()

    if host:
        ws_scheme = "wss" if scheme == "https" else "ws"
        return f"{ws_scheme}://{host}/call"

    return settings.ws_call_url