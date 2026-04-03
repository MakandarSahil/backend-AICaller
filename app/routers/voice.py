"""
app/routers/voice.py — /voice

Twilio webhook. Called automatically when a call arrives on a configured number.
Not used by the dashboard or external developers.
"""

import logging

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
    <Stream url="{ws_url}?agent_id={agent_id}" />
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
    if settings.is_production and settings.twilio_auth_token:
        if not await _validate_twilio_signature(request):
            logger.warning(
                "Twilio signature FAILED: agent_id=%s ip=%s",
                agent_id,
                request.client.host if request.client else "unknown",
            )
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    if not await _agent_exists(agent_id):
        logger.warning("TwiML: unknown/inactive agent_id=%s", agent_id)
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")

    ws_url = settings.ws_call_url
    twiml = _TWIML_TEMPLATE.format(ws_url=ws_url, agent_id=agent_id)

    logger.info("TwiML served: agent_id=%s", agent_id)
    return Response(content=twiml, media_type="text/xml")


async def _validate_twilio_signature(request: Request) -> bool:
    try:
        from twilio.request_validator import RequestValidator
        signature = request.headers.get("X-Twilio-Signature", "")
        url = str(request.url)
        params = dict(await request.form()) if request.method == "POST" else {}
        return RequestValidator(settings.twilio_auth_token).validate(url, params, signature)
    except Exception as exc:
        logger.error("Twilio signature validation error: %s", exc)
        return False


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