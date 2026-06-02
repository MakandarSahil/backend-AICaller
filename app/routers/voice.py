"""
app/routers/voice.py — /voice

Twilio webhook. Called automatically when a call arrives on a configured number.
Not used by the dashboard or external developers.

BYO Twilio Support:
- Detects if the called number belongs to a connected provider
- Uses provider-specific credentials for webhook validation
- Falls back to platform credentials for platform numbers
"""

import asyncio
import json
import logging
from base64 import b64decode
from html import escape
from urllib.parse import urlparse

from cryptography.fernet import Fernet
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

from app.clients.redis import get_redis
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

_BYO_CREDS_TIMEOUT = 3.0


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
    form_data = {}
    if request.method == "POST":
        try:
            form_data = dict(await request.form())
        except Exception as form_error:
            logger.warning("Could not read form data: %s", form_error)

    to_number = str(form_data.get("To") or "")

    # ── Parallelise BYO cred lookup + agent check ─────────────────────────
    creds_task = (
        asyncio.create_task(_get_provider_credentials_for_number(to_number))
        if to_number
        else None
    )
    agent_task = asyncio.create_task(_agent_exists(agent_id))

    agent_exists = await agent_task

    provider_creds = None
    if creds_task:
        try:
            provider_creds = await asyncio.wait_for(creds_task, timeout=_BYO_CREDS_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("BYO cred lookup timed out (>%ss) for %s", _BYO_CREDS_TIMEOUT, to_number)
        except Exception as exc:
            logger.error("BYO cred lookup failed for %s: %s", to_number, exc)

    auth_token = settings.twilio_auth_token
    is_byo = False
    if provider_creds:
        auth_token = provider_creds.get("auth_token", auth_token)
        is_byo = True
        logger.info("Using BYO Twilio credentials for number=%s", to_number)

    # ── Signature validation ──────────────────────────────────────────────
    if auth_token:
        valid = await _validate_twilio_signature(request, auth_token, form_data)
        if not valid:
            logger.warning(
                "Twilio signature FAILED — allowing call to proceed: agent_id=%s ip=%s is_byo=%s",
                agent_id,
                request.client.host if request.client else "unknown",
                is_byo,
            )

    if not agent_exists:
        logger.warning("TwiML: unknown/inactive agent_id=%s", agent_id)
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")

    from_number = str(form_data.get("From") or "")
    call_sid = str(form_data.get("CallSid") or "")
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
        "TwiML served: agent_id=%s from=%s call_sid=%s ws_url=%s is_byo=%s",
        agent_id,
        from_number,
        call_sid,
        ws_url,
        is_byo,
    )
    return Response(content=twiml, media_type="text/xml")


async def _validate_twilio_signature(
    request: Request,
    auth_token: str,
    form_data: dict | None = None,
) -> bool:
    try:
        from twilio.request_validator import RequestValidator

        signature = request.headers.get("X-Twilio-Signature", "")
        if not signature:
            logger.warning("No X-Twilio-Signature header — skipping validation")
            return False

        # Reconstruct the external URL Twilio signed using forwarded headers.
        # In production behind Traefik/Cloudflare, forwarded headers carry the
        # original scheme and host. Fall back to request.url if proxies are absent.
        forwarded_proto = request.headers.get("x-forwarded-proto", "")
        forwarded_host = request.headers.get("x-forwarded-host", "")
        host = request.headers.get("host", "")

        proto = forwarded_proto.split(",")[0].strip() or request.url.scheme
        netloc = forwarded_host.split(",")[0].strip() or host or request.url.netloc

        # Strip default ports — Twilio does not include them in the signed URL
        if proto == "https" and netloc.endswith(":443"):
            netloc = netloc[:-4]
        elif proto == "http" and netloc.endswith(":80"):
            netloc = netloc[:-3]

        query = f"?{request.url.query}" if request.url.query else ""
        url = f"{proto}://{netloc}{request.url.path}{query}"

        params = form_data if form_data is not None else {}

        is_valid = RequestValidator(auth_token).validate(url, params, signature)
        if not is_valid:
            logger.warning(
                "Twilio signature mismatch: url=%s proto=%s fwd_host=%s host=%s",
                url, forwarded_proto, forwarded_host, host,
            )
        return is_valid
    except Exception as exc:
        logger.error("Twilio signature validation error: %s", exc)
        return False


async def _get_provider_credentials_for_number(phone_number: str) -> dict | None:
    if not phone_number:
        return None

    cache_key = f"voice:provider_creds:{phone_number}"
    try:
        redis = get_redis()
        cached = await redis.get(cache_key)
        if cached is not None:
            return None if cached == "null" else json.loads(cached)
    except Exception as exc:
        logger.warning("Redis cache read failed for provider creds: %s", exc)

    try:
        supabase = get_supabase()

        result = (
            await supabase.table("phone_numbers")
            .select("number_type, is_active, telephony_provider_id")
            .eq("number", phone_number)
            .maybe_single()
            .execute()
        )

        if not result or not result.data:
            await _cache_null(redis, cache_key)
            return None

        if not result.data.get("is_active", False):
            await _cache_null(redis, cache_key)
            return None

        provider_id = result.data.get("telephony_provider_id")
        number_type = result.data.get("number_type")

        if not provider_id and number_type != "own":
            await _cache_null(redis, cache_key)
            return None

        # Single query for provider + secrets using the FK relationship
        if provider_id:
            provider_result = (
                await supabase.table("workspace_telephony_providers")
                .select("vault_secret_id")
                .eq("id", provider_id)
                .eq("is_active", True)
                .single()
                .execute()
            )
        else:
            return None

        if not provider_result.data or not provider_result.data.get("vault_secret_id"):
            return None

        vault_secret_id = provider_result.data["vault_secret_id"]

        secret_result = (
            await supabase.table("provider_secrets")
            .select("encrypted_data")
            .eq("id", vault_secret_id)
            .single()
            .execute()
        )

        if not secret_result.data or not secret_result.data.get("encrypted_data"):
            logger.error("Failed to retrieve credentials for secret: %s", vault_secret_id)
            return None

        encrypted = secret_result.data["encrypted_data"]

        loop = asyncio.get_running_loop()
        credentials = await loop.run_in_executor(None, _decrypt_credentials_sync, encrypted)

        final_creds = {
            "account_sid": credentials.get("account_sid"),
            "auth_token": credentials.get("auth_token"),
        }

        try:
            await redis.setex(cache_key, 3600, json.dumps(final_creds))
        except Exception:
            pass

        return final_creds

    except Exception as exc:
        logger.error("Error getting provider credentials for %s: %s", phone_number, exc)
        return None


async def _cache_null(redis, cache_key: str) -> None:
    try:
        await redis.setex(cache_key, 3600, "null")
    except Exception:
        pass


def _decrypt_credentials_sync(ciphertext: str) -> dict:
    cipher = Fernet(settings.encryption_key.encode())
    encrypted = b64decode(ciphertext.encode())
    decrypted = cipher.decrypt(encrypted)
    return json.loads(decrypted.decode())


async def _agent_exists(agent_id: str) -> bool:
    cache_key = f"voice:agent_active:{agent_id}"
    try:
        redis = get_redis()
        cached = await redis.get(cache_key)
        if cached is not None:
            return cached == "1"
    except Exception as exc:
        logger.warning("Redis cache read failed for _agent_exists: %s", exc)

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
        exists = bool(result.data)
        try:
            redis = get_redis()
            await redis.setex(cache_key, 3600, "1" if exists else "0")
        except Exception:
            pass
        return exists
    except Exception as exc:
        logger.error("Agent existence check failed: %s", exc)
        return False


def _resolve_ws_call_url(request: Request) -> str:
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
