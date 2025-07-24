import os, audioop, asyncio, time, torch
import numpy as np
import scipy.signal
from TTS.api import TTS
import threading

# Initialize XTTS v2 once at startup (not during first call)
_tts_model = None
_model_loaded = False
_loading_lock = threading.Lock()

def ensure_tts_model():
    """Load TTS model in a thread-safe way without blocking"""
    global _tts_model, _model_loaded
    
    if _model_loaded:
        return _tts_model
        
    with _loading_lock:
        if not _model_loaded:
            print("Loading XTTS model...")
            try:
                _tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
                if torch.cuda.is_available():
                    _tts_model = _tts_model.to("cuda")
                _model_loaded = True
                print("XTTS model loaded!")
            except Exception as e:
                print(f"Failed to load XTTS: {e}")
                _tts_model = None
                
    return _tts_model

# Pre-load the model at import time (non-blocking)
def _preload_model():
    """Load model in background thread"""
    ensure_tts_model()

# Start loading immediately when module is imported
_preload_thread = threading.Thread(target=_preload_model, daemon=True)
_preload_thread.start()

def stream(text: str, chunk_ms: int = 20, **kwargs):
    """
    Yield 20 ms μ-law frames (160 B) using XTTS.
    Uses the detected language from STT via response_language parameter.
    Drop-in replacement for ElevenLabs streaming function.
    """
    try:
        # Get TTS model (should already be loaded)
        tts = ensure_tts_model()
        if tts is None:
            raise Exception("TTS model not loaded")
        
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
