import os, base64, json, asyncio, audioop, time, wave, contextlib, torch, gc, re
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.websockets import WebSocketDisconnect, WebSocketState
import webrtcvad

from stt import new_stream
from tts import stream as tts_stream
from dotenv import load_dotenv
from lang import get_llm_response, llm_stream
import numpy as np
import scipy

load_dotenv()

# ───────────────── CONFIG ───────────────── #
PORT               = int(os.getenv("PORT", 5050))
FRAME_MS           = 20                  # Twilio frame duration
SAMPLE_RATE        = 8000                 # Hz – μ-law native
SAMPLE_BYTES       = 2                  # 16-bit PCM
BYTES_PER_FRAME    = SAMPLE_RATE // (1000 // FRAME_MS) * SAMPLE_BYTES
VAD_AGGRESSIVENESS = 3                   # 0-3 (higher = more speech detected)
MIN_VOICE_FRAMES   = 8                  # must detect speech on ≥ 6 consecutive frames (≈120 ms)
ENERGY_FLOOR       = 300                # optional extra gate on RMS
GRACE_MS           = 300                
INTRO_PATH         = "intro.wav"         # mono 16-bit 8 kHz
# ────────────────────────────────────────── #

vad    = webrtcvad.Vad(VAD_AGGRESSIVENESS)
now_ms = lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
app    = FastAPI()

# ───────── LLM warm‑up ───────── #
print("[⚙️ LLM warm-up] loading …")
_ = get_llm_response("ping")
print("[⚙️ LLM warm-up] done")

# ───────── helper utils ───────── #
_is_meaningful = re.compile(r"\w").search
def meaningful(text: str) -> bool:
    return bool(_is_meaningful(text))

_ABBREVS = {
    "dr", "mr", "mrs", "ms", "jr", "sr", "st", "prof", "inc", "ltd",
    "fig", "dept", "no", "vs", "gen", "col", "lt", "etc", "al", "u.s",
    "e.g", "i.e"
}

def is_sentence_end(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    last_char = stripped[-1]
    if last_char in "!?":
        return True
    if last_char != ".":
        return False
    pre = stripped[:-1].rstrip()
    if not pre:
        return False
    last_word = re.split(r"\s+", pre)[-1].lower()
    last_word = re.sub(r"[^\w\.]", "", last_word)
    if not last_word or last_word in _ABBREVS or len(last_word) == 1 or last_word.isdigit():
        return False
    return True
# ─────────────────────────────── #

@app.get("/")
async def index():
    return {"message": "running"}

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    host = request.url.hostname
    return HTMLResponse(
        content=f"""
        <Response>
          <Connect>
            <Stream url="wss://{host}/media-stream" />
          </Connect>
        </Response>
        """,
        media_type="application/xml",
    )

# ───────────────── media stream ───────────────── #
@app.websocket("/media-stream")
async def media_stream(ws: WebSocket):
    await ws.accept()
    print("[🎧] Caller connected")

    # session‑level objects
    stt_stream     = new_stream()
    barge_event    = asyncio.Event()
    tts_task       = None
    stream_sid     = None
    call_sid       = None
    tts_started_ts = 0

    # ───────── inner helpers ───────── #
    async def send_clear():
        if ws.application_state == WebSocketState.CONNECTED:
            await ws.send_json({"event": "clear", "streamSid": stream_sid})

    async def stream_ulaw_frames(frame_iter):
        """Send μ-law frames until barge-in or WebSocket closes."""
        for ulaw in frame_iter:
            if barge_event.is_set() or ws.application_state != WebSocketState.CONNECTED:
                break
            try:
                await ws.send_json({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": base64.b64encode(ulaw).decode()},
                })
            except RuntimeError:        # ws already closed
                break
            await asyncio.sleep(FRAME_MS / 1000)

    async def stream_tts(text: str):
        text = text.strip()
        if not meaningful(text):
            return
        try:
            await stream_ulaw_frames(tts_stream(text))
        except RuntimeError:
            pass 

    def wav_ulaw_frames(path: str):
        with wave.open(path, "rb") as wav:
            assert wav.getframerate() == SAMPLE_RATE and wav.getnchannels() == 1
            sw = wav.getsampwidth()
            while True:
                pcm = wav.readframes(BYTES_PER_FRAME // sw)
                if not pcm:
                    break
                if len(pcm) < BYTES_PER_FRAME:
                    pcm += b"\x00" * (BYTES_PER_FRAME - len(pcm))
                yield audioop.lin2ulaw(pcm, sw)

    async def play_intro():
        await stream_ulaw_frames(wav_ulaw_frames(INTRO_PATH))

    # ───────── receive audio & VAD ───────── #
    async def receive_audio():
        nonlocal stream_sid, call_sid, tts_task, tts_started_ts
        buf, streak = bytearray(), 0
        try:
            async for msg in ws.iter_text():
                data = json.loads(msg)

                if data["event"] == "start":
                    stream_sid = data["start"]["streamSid"]
                    call_sid   = data["start"]["callSid"]
                    print(f"[🔄] Stream {stream_sid} / Call {call_sid}")
                    barge_event.clear()
                    tts_started_ts = time.monotonic() * 1000
                    tts_task = asyncio.create_task(play_intro())
                    continue

                if data["event"] != "media":
                    continue

                pcm = audioop.ulaw2lin(
                    base64.b64decode(data["media"]["payload"]),
                    SAMPLE_BYTES,
                )
                buf.extend(pcm)

                while len(buf) >= BYTES_PER_FRAME:
                    frame = buf[:BYTES_PER_FRAME]; del buf[:BYTES_PER_FRAME]

                    # grace window
                    if (time.monotonic()*1000 - tts_started_ts) < GRACE_MS:
                        stt_stream.feed_audio(frame); continue

                    # VAD barge‑in
                    if tts_task and not tts_task.done() and not barge_event.is_set():
                        voiced = (
                            audioop.rms(frame, SAMPLE_BYTES) >= ENERGY_FLOOR
                            and vad.is_speech(frame, SAMPLE_RATE)
                        )
                        streak = streak + 1 if voiced else 0
                        if streak >= MIN_VOICE_FRAMES:
                            barge_event.set(); streak = 0
                            await send_clear()
                            tts_task.cancel()
                            print("[🔄] User barged in, clearing TTS task")
                            # if call_sid:
                            #     remove_last_assistant_message(call_sid)

                    stt_stream.feed_audio(frame)

        except WebSocketDisconnect:
            print("[❌] WebSocket disconnected")

    # ───────── dialog loop (STT → LLM → TTS) ───────── #
    async def talk_loop():
        nonlocal tts_task, tts_started_ts

        async def speak(sentence: str):
            nonlocal tts_task, tts_started_ts
            sentence = sentence.strip()
            if not meaningful(sentence):
                return
            if tts_task and not tts_task.done():
                with contextlib.suppress(asyncio.CancelledError):
                    await tts_task
            barge_event.clear()
            tts_started_ts = time.monotonic() * 1000
            tts_task = asyncio.create_task(stream_tts(sentence))

        try:
            print("[🎤] Waiting for user input…")
            async for seg in stt_stream:
                user_text = seg.text.strip()
                if not meaningful(user_text):
                    continue
                print(f"[{now_ms()}] User: {user_text}")

                llm_start     = time.perf_counter()
                buffer_tokens = []

                async for token in llm_stream(user_text, call_sid):
                    
                    if not token:       # guard against empty chunk
                        continue
                    buffer_tokens.append(token)
                    joined = "".join(buffer_tokens)

                    if token[-1] in ".!?" and is_sentence_end(joined):
                        sentence = joined.strip()
                        buffer_tokens.clear()
                        print(f"[{now_ms()}] Bot⋯ {sentence}")
                        await speak(sentence)

                if buffer_tokens and meaningful("".join(buffer_tokens)):
                    sentence = "".join(buffer_tokens).strip()
                    print(f"[{now_ms()}] Bot : {sentence}")
                    await speak(sentence)

                print(f"[⏱️ LLM {time.perf_counter() - llm_start:.3f}s]")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if ws.application_state == WebSocketState.CONNECTED:
                    await ws.close(code=1011, reason="GPU OOM")
            else:
                raise

    # ───────── run & cleanup ───────── #
    try:
        await asyncio.gather(receive_audio(), talk_loop())
    finally:
        barge_event.set()
        if tts_task and not tts_task.done():
            tts_task.cancel()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("[🧹] Call resources cleaned")

# ─────────────────────────────────────────────── #
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)


