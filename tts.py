import os, audioop, asyncio, time, torch
import numpy as np
import scipy.signal
from TTS.api import TTS

# Initialize XTTS v2 once
_tts_model = None

def get_tts_model():
    global _tts_model
    if _tts_model is None:
        print("Loading XTTS model...")
        _tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        if torch.cuda.is_available():
            _tts_model = _tts_model.to("cuda")
        print("XTTS model loaded!")
    return _tts_model

def stream(text: str, chunk_ms: int = 20, **kwargs):
    """
    Yield 20 ms μ-law frames (160 B) using XTTS.
    Uses the detected language from STT via response_language parameter.
    Drop-in replacement for ElevenLabs streaming function.
    """
    try:
        # Get TTS model
        tts = get_tts_model()
        
        # Extract language from kwargs (passed from LLM chain)
        # Format: "Language with code 'de'" -> extract 'de'
        language = "en"  # default
        if "response_language" in kwargs:
            lang_str = kwargs["response_language"]
            if "code '" in lang_str:
                # Extract language code from "Language with code 'de'"
                import re
                match = re.search(r"code '([^']+)'", lang_str)
                if match:
                    language = match.group(1)
        
        print(f"🗣️ TTS: Generating speech in '{language}' for: '{text[:50]}...'")
        
        # Generate audio with XTTS using detected language
        audio = tts.tts(text=text, language=language)
        
        # Convert tensor to numpy if needed
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        # Handle stereo -> mono
        if len(audio.shape) > 1:
            audio = audio[0]
        
        # Resample from 24kHz (XTTS default) to 8kHz
        original_sr = 24000
        target_sr = 8000
        target_length = int(len(audio) * target_sr / original_sr)
        audio_8k = scipy.signal.resample(audio, target_length)
        
        # Convert float32 to int16 PCM
        if audio_8k.dtype == np.float32:
            audio_8k = np.clip(audio_8k, -1.0, 1.0)
            audio_8k = (audio_8k * 32767).astype(np.int16)
        
        # Convert PCM to μ-law
        pcm_bytes = audio_8k.tobytes()
        ulaw_bytes = audioop.lin2ulaw(pcm_bytes, 2)
        
        # Yield in 160-byte chunks (same as ElevenLabs)
        for i in range(0, len(ulaw_bytes), 160):
            chunk = ulaw_bytes[i:i+160]
            if len(chunk) < 160:
                # Pad with μ-law silence
                chunk += b"\x7F" * (160 - len(chunk))
            yield bytes(chunk)
            
    except Exception as e:
        print(f"XTTS Error: {e}")
        # Return silence on error (same behavior as original)
        for _ in range(50):  # ~1 second of silence
            yield b"\x7F" * 160
