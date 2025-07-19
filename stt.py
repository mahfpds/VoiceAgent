import asyncio, webrtcvad, numpy as np, audioop, torch, gc
from faster_whisper import WhisperModel

ASR_SR          = 16000
FRAME_MS        = 20
BYTES_PER_FRAME = int(ASR_SR * FRAME_MS / 1000) * 2   # 640 B

# outer VAD (fast gate for silence / barge‑in)
VAD             = webrtcvad.Vad(2)
END_SIL_FR      = 15                 # 0.30 s pause → flush
MIN_VOICE_FR    = 5                  # 0.10 s speech → flush
WIN_SEC, HOP_SEC = 4, 1              # longer window = better context
ENERGY_FLOOR    = 150
MODEL_NAME      = "large-v3"

_model = WhisperModel(
    MODEL_NAME,
    device="cuda",
    compute_type="float16", 
)


# high‑quality decode parameters
DECODE_OPTS = dict(
    beam_size=5,
    best_of=5,
    patience=1.0,                    # let beam search look further
    temperature=[0.0],
    vad_filter=True,                 # use Whisper’s internal VAD too
    word_timestamps=False,
    condition_on_previous_text=True, # better coherence, fewer slips
    no_speech_threshold=0.5,         # more willing to keep quiet segments
    compression_ratio_threshold=2.4,
    log_prob_threshold=-1.0,
)

class WhisperStream:
    def __init__(self):
        self._ring = bytearray()
        self._speech = bytearray()
        self._sil = self._voice = 0
        self._q = asyncio.Queue()
        self._lang = None           # becomes "en", "de", etc.

    def feed_audio(self, pcm16: bytes):
        self._ring.extend(pcm16)
        while len(self._ring) >= BYTES_PER_FRAME:
            frame = self._ring[:BYTES_PER_FRAME]
            del self._ring[:BYTES_PER_FRAME]
            voiced = (
                audioop.rms(frame, 2) >= ENERGY_FLOOR and
                VAD.is_speech(frame, ASR_SR)
            )
            if voiced:
                self._speech.extend(frame)
                self._voice += 1
                self._sil = 0
            else:
                if self._speech:
                    self._sil += 1
                    if self._sil >= END_SIL_FR:
                        if self._voice >= MIN_VOICE_FR:
                            self._q.put_nowait(bytes(self._speech))
                        self._speech.clear()
                        self._voice = self._sil = 0
                        
    # --------------- output side --------------- #
    async def __aiter__(self):
        hop, win = HOP_SEC * ASR_SR * 2, WIN_SEC * ASR_SR * 2
        while True:
            pcm = await self._q.get()

            # Overlapping windows (4 s / 1 s hop) for higher quality
            chunks = [pcm[i:i+win] for i in range(0, max(len(pcm)-win, 0)+hop, hop)] or [pcm]

            for ch in chunks:
                audio = np.frombuffer(ch, np.int16).astype(np.float32) / 32768.0

                # -------- automatic language detection (once) -------- #
                if self._lang is None:
                    lang_detect = _model.detect_language(audio)
                    # works with both return types (tuple or dict)
                    if isinstance(lang_detect, tuple):        # (lang, probs) older API
                        self._lang = lang_detect[0]
                    elif isinstance(lang_detect, dict):       # newer API
                        self._lang = max(lang_detect, key=lang_detect.get)
                    else:
                        self._lang = "en"                     # sane fallback
                    print("🔤 Detected language:", self._lang)

                # ------------------- transcription ------------------- #
                segs, _ = _model.transcribe(
                    audio,
                    **DECODE_OPTS,
                    language=self._lang,
                )
                for s in segs:
                    if s.no_speech_prob and s.no_speech_prob > 0.7:
                        continue
                    yield s

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

def new_stream():
    return WhisperStream()

