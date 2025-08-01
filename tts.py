import os, httpx, audioop, asyncio, time
from dotenv import load_dotenv

load_dotenv()
API_KEY  = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

# ElevenLabs can stream μ‑law 8 kHz directly → no mpg123 / temp‑file / decode
URL = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream?output_format=ulaw_8000"

HEADERS = {"xi-api-key": API_KEY, "Content-Type": "application/json"}
PAYLOAD_BASE = {
    "model_id": "eleven_flash_v2_5",
    "voice_settings": {"stability": 0.5, "similarity_boost": 0.7},
}

def stream(text: str, chunk_ms: int = 20):
    """
    Yield 20 ms μ-law frames (160 B) **as the TTS bytes arrive**.
    Latency ≈ first packet in ~250 ms vs. >1 s previously.
    """
    payload = PAYLOAD_BASE | {"text": text}

    with httpx.stream("POST", URL, headers=HEADERS, json=payload, timeout=None) as r:
        r.raise_for_status()
        buf = bytearray()
        for data in r.iter_bytes():
            buf.extend(data)
            # ElevenLabs already gives μ‑law 8 kHz; just slice into 160‑byte frames
            while len(buf) >= 160:
                yield bytes(buf[:160]); del buf[:160]

    # flush trailing partial (pad with silence)
    if buf:
        buf.extend(b"\x7F"*(160-len(buf)))  # μ-law silence byte
        yield bytes(buf[:160])
