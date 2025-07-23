import os, base64, json, asyncio, audioop, time, wave, contextlib, torch, gc, re
import numpy as np
import scipy.signal
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.websockets import WebSocketDisconnect, WebSocketState
import webrtcvad
from stt import new_stream
from tts import stream as tts_stream
from dotenv import load_dotenv
from lang import get_llm_response, llm_stream
from pathlib import Path

load_dotenv()

# ───────────────── CONFIG ───────────────── #
PORT               = int(os.getenv("PORT", 5050))
FRAME_MS           = 20                    # Twilio frame duration

# Media‑stream side (Twilio ⇄ FastAPI) – 8 kHz μ‑law
STREAM_SR          = 8000
STREAM_SAMPLE_BYTES= 2
STREAM_BPF         = int(STREAM_SR * FRAME_MS / 1000) * STREAM_SAMPLE_BYTES  # 320 B

# ASR side (Whisper) – 16 kHz PCM
ASR_SR             = 16000
ASR_SAMPLE_BYTES   = 2
ASR_BPF            = int(ASR_SR * FRAME_MS / 1000) * ASR_SAMPLE_BYTES        # 640 B

VAD_AGGRESSIVENESS = 3                   # 0‑3 (higher = more speech detected)
MIN_VOICE_FRAMES   = 8                   # ≥ 6 consecutive voiced frames (~120 ms)
ENERGY_FLOOR       = 300
GRACE_MS           = 300
INTRO_PATH         = "intro.wav"         # mono, 16‑bit, 8 kHz
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

def ms() -> int:                   # monotonic in ms
    return int(time.perf_counter() * 1000)

def log_latency(label: str, start_ms: int):
    print(f"[⏱️ {label} {round((ms() - start_ms) / 1000, 2)} seconds")

def open_recorder(call_sid: str):
    now = time.localtime()
    fname = f"{call_sid}_{now.tm_year}-{now.tm_mon:02}-{now.tm_mday:02}_{now.tm_hour:02}-{now.tm_min:02}-{now.tm_sec:02}.wav"
    path  = Path("records") / fname
    path.parent.mkdir(parents=True, exist_ok=True)

    w = wave.open(str(path), "wb")
    w.setnchannels(1)
    w.setsampwidth(2)            # 16‑bit PCM
    w.setframerate(STREAM_SR)    # 8000 Hz (exact input rate)
    return w


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

# ───────────────── helper I/O ───────────────── #
async def stream_ulaw_frames(ws: WebSocket, frame_iter, stream_sid: str, barge_event: asyncio.Event, label: str | None = None):
    """Send μ-law frames to Twilio until barge-in or WebSocket closes."""
    first = True
    start_ms = ms()
    for ulaw in frame_iter:
        if barge_event.is_set() or ws.application_state != WebSocketState.CONNECTED:
            break

        if first and label is not None:
            first = False
            log_latency(label, start_ms)    # <— NEW

        try:
            await ws.send_json({
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": base64.b64encode(ulaw).decode()},
            })
        except RuntimeError:
            break
        await asyncio.sleep(FRAME_MS / 1000)

def wav_ulaw_frames(path: str):
    with wave.open(path, "rb") as wav:
        assert wav.getframerate() == STREAM_SR and wav.getnchannels() == 1
        sw = wav.getsampwidth()
        while True:
            pcm = wav.readframes(STREAM_BPF // sw)
            if not pcm:
                break
            if len(pcm) < STREAM_BPF:
                pcm += b"\x00" * (STREAM_BPF - len(pcm))
            yield audioop.lin2ulaw(pcm, sw)
# ────────────────────────────────────────────── #

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
    recorder = None    # ← wav writer
    last_audio_ms = 0

    # ───────── inner helpers ───────── #
    async def send_clear():
        if ws.application_state == WebSocketState.CONNECTED:
            await ws.send_json({"event": "clear", "streamSid": stream_sid})

    async def stream_tts(text: str):
        text = text.strip()
        if not meaningful(text):
            return
        try:
            await stream_ulaw_frames(ws, tts_stream(text), stream_sid, barge_event, label="TTS-1st-frame")
        except RuntimeError:
            pass

    async def play_intro():
        await stream_ulaw_frames(ws, wav_ulaw_frames(INTRO_PATH), stream_sid, barge_event)

    # ───────── receive audio & VAD ───────── #
    async def receive_audio():
        nonlocal stream_sid, call_sid, tts_task, tts_started_ts, recorder, last_audio_ms
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
                    recorder   = open_recorder(call_sid)
                    continue

                if data["event"] != "media":
                    continue

                # 8 kHz μ‑law → 16‑bit PCM @ 8 kHz
                pcm8k = audioop.ulaw2lin(
                    base64.b64decode(data["media"]["payload"]),
                    STREAM_SAMPLE_BYTES,
                )
                buf.extend(pcm8k)

                while len(buf) >= STREAM_BPF:
                    frame8k = buf[:STREAM_BPF]; del buf[:STREAM_BPF]

                    # grace window (still feed ASR, but skip VAD)
                    if (time.monotonic()*1000 - tts_started_ts) < GRACE_MS:
                        pcm16k = scipy.signal.resample_poly(
                            np.frombuffer(frame8k, np.int16), 2, 1
                        ).astype(np.int16).tobytes()
                        stt_stream.feed_audio(pcm16k)
                        continue

                    # VAD barge‑in on 8 kHz frame
                    if tts_task and not tts_task.done() and not barge_event.is_set():
                        voiced = (
                            audioop.rms(frame8k, STREAM_SAMPLE_BYTES) >= ENERGY_FLOOR
                            and vad.is_speech(frame8k, STREAM_SR)
                        )
                        streak = streak + 1 if voiced else 0
                        if streak >= MIN_VOICE_FRAMES:
                            barge_event.set(); streak = 0
                            await send_clear()
                            tts_task.cancel()
                            print("[🔄] User barged in, clearing TTS task")

                    # up‑sample 8 kHz → 16 kHz and feed ASR
                    pcm16k = scipy.signal.resample_poly(
                        np.frombuffer(frame8k, np.int16), 2, 1
                    ).astype(np.int16).tobytes()
                    stt_stream.feed_audio(pcm16k)
                    last_audio_ms = ms()

        except WebSocketDisconnect:
            print("[❌] WebSocket disconnected")

    # ───────── dialog loop (STT → LLM → TTS) ───────── #
    async def talk_loop():
        nonlocal tts_task, tts_started_ts, last_audio_ms

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
            call_lang = None        # ← new
            async for seg in stt_stream:
                user_text = seg.text.strip()
                if not meaningful(user_text):
                    continue

                # ---------- ASR latency inside talk_loop ----------
                log_latency("ASR", last_audio_ms)
                # --------------------------------------------------

                # ------------- NEW: capture Whisper's language -------------
                if call_lang is None:
                    lang_code = getattr(stt_stream, "_lang", "en")
                    call_lang = f"Language with code '{lang_code}'"
                # -----------------------------------------------------------

                print(f"[{now_ms()}] User: {user_text}")

                llm_req_ms = ms()
                first_token = True
                buffer_tokens = []

                async for token in llm_stream(
                user_text,
                call_sid,
                response_language=call_lang 
                ):
                    if not token:
                        continue

                    if first_token:
                        log_latency("LLM-1st-tok", llm_req_ms)   # <— NEW
                        first_token = False

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
        if recorder:
            recorder.close()
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
