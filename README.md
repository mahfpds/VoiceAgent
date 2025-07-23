# Voice Agent App: Installation & Usage Guide

## 1. Prerequisites

- **Operating System:** Linux VM
- **Python:** Version 3.10
- **Conda:** Recommended for environment and package management
- **SIP Trunk Provider:** Credentials and connection details

## 2. Full Environment Setup (with GPU Support)

Follow these steps to set up your environment and install all dependencies, including GPU support for PyTorch:

```bash
conda create -n voiceagent python=3.10
conda activate voiceagent
conda install -c conda-forge fastapi uvicorn webrtcvad python-dotenv numpy scipy httpx requests libstdcxx-ng
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pip
pip install faster-whisper langchain-core langchain-ollama
conda update --all
python app.py
```

### Notes on CUDA Version
- The command `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia` installs PyTorch and related libraries with GPU (CUDA 12.x) support.
- **You do not need to match the CUDA version exactly to your driver version.** For example, if your system CUDA driver is 12.9, using `pytorch-cuda=12.1` is correct and fully compatible. CUDA is backward compatible with minor versions.
- Make sure your system has an NVIDIA GPU and the appropriate CUDA drivers installed.
- After installation, you can verify GPU support in Python with:

```python
import torch
print(torch.cuda.is_available())
```
If it prints `True`, PyTorch can use your GPU. 

## 3. Ollama Setup

Ollama is required for running the language model backend. Install and start Ollama on your server:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
```

- Visit [https://ollama.com/download](https://ollama.com/download) for more details or alternative installation methods.
- Make sure Ollama is running before starting the app.

## 4. ElevenLabs API Keys (.env file)

To use ElevenLabs TTS, you need to set up a `.env` file in your project directory with your API credentials:

```
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
ELEVENLABS_VOICE_ID=your_elevenlabs_voice_id_here
```

- Replace `your_elevenlabs_api_key_here` and `your_elevenlabs_voice_id_here` with your actual ElevenLabs credentials.
- You can obtain these from your ElevenLabs account dashboard.

