"""
Audio conversion utilities.

Twilio sends audio as mu-law (mulaw) encoded at 8kHz.
Azure STT expects PCM16 at 8kHz.
Azure TTS can return mulaw directly (we configure it to do so).

RFC 3551 mu-law encode/decode.
"""

_MULAW_BIAS = 33
_MULAW_MAX = 0x7FFF


def mulaw_to_pcm16(mulaw_bytes: bytes) -> bytes:
    """
    Convert a buffer of mu-law encoded bytes to PCM16LE.
    Input: raw mulaw bytes from Twilio media event (after base64 decode)
    Output: PCM16 little-endian bytes suitable for Azure STT push stream
    """
    out = bytearray(len(mulaw_bytes) * 2)
    for i, byte in enumerate(mulaw_bytes):
        sample = _mulaw_decode(byte)
        # Write as 16-bit little-endian
        out[i * 2] = sample & 0xFF
        out[i * 2 + 1] = (sample >> 8) & 0xFF
    return bytes(out)


def _mulaw_decode(mulaw_byte: int) -> int:
    """Decode a single mu-law byte to a signed 16-bit PCM sample."""
    mulaw_byte = (~mulaw_byte) & 0xFF
    sign = -1 if (mulaw_byte & 0x80) else 1
    exponent = (mulaw_byte >> 4) & 0x07
    mantissa = mulaw_byte & 0x0F
    sample = ((mantissa << 3) + _MULAW_BIAS) << exponent
    return -sample if sign == -1 else sample


def pcm16_to_mulaw(pcm16_bytes: bytes) -> bytes:
    """
    Convert PCM16LE bytes to mu-law bytes.
    Used when Azure TTS returns PCM16 and we need to send mulaw to Twilio.
    Note: we configure Azure TTS to return mulaw directly,
    so this is a fallback / utility function.
    """
    out = bytearray(len(pcm16_bytes) // 2)
    for i in range(len(out)):
        # Read 16-bit little-endian sample
        sample = int.from_bytes(pcm16_bytes[i * 2 : i * 2 + 2], "little", signed=True)
        out[i] = _pcm16_encode(sample)
    return bytes(out)


def _pcm16_encode(sample: int) -> int:
    """Encode a signed 16-bit PCM sample to a single mu-law byte."""
    sign = 0x80 if sample < 0 else 0
    if sample < 0:
        sample = -sample
    sample = min(sample, _MULAW_MAX)
    sample += _MULAW_BIAS

    exponent = 7
    exp_mask = 0x4000
    while (sample & exp_mask) == 0 and exponent > 0:
        exponent -= 1
        exp_mask >>= 1

    mantissa = (sample >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa)) & 0xFF


def strip_riff_header(buf: bytes) -> bytes:
    """
    Strip RIFF/WAV header if present and return raw audio data.
    Azure TTS sometimes wraps audio in a RIFF container even when
    a raw format is requested — this handles both cases safely.
    """
    if len(buf) >= 12 and buf[:4] == b"RIFF":
        # Find the 'data' chunk
        data_idx = buf.find(b"data")
        if data_idx != -1:
            # Skip 'data' (4 bytes) + chunk size (4 bytes)
            return buf[data_idx + 8 :]
        # Fallback: skip standard 44-byte WAV header
        return buf[44:]
    return buf


def calculate_volume(mulaw_bytes: bytes) -> float:
    """
    Calculate the average volume of a mulaw audio chunk.
    Used for barge-in detection and silence endpointing.

    Decodes mulaw to PCM16 internally to calculate true amplitude.
    Returns a float representing average absolute PCM amplitude (0 to 32767).
    """
    if not mulaw_bytes:
        return 0.0
    
    total = 0
    for b in mulaw_bytes:
        total += abs(_mulaw_decode(b))
        
    return total / len(mulaw_bytes)