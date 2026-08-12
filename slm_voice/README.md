# SLM Voice Agent

A local, low-latency CPU-optimized Voice Agent pipeline. It allows conversational speech commands to be processed locally on CPU, with fallback audio transcription synthesis checks.

---

## Features

- **Local Command Processing**: Receives spoken transcripts or audio files and executes local tools or chat responses.
- **Multilingual Support**: Supports transcribing and responding in English, Hindi, Tamil, Telugu, Spanish, French, and German. Input queries are translated to English to execute tools, and the final responses are translated back to the user's language.
- **Mac-Native & Fallback TTS**: Directly uses the built-in macOS `say` utility for high-quality audio playback and saving to file on macOS, with automatic fallback to `pyttsx3` on other platforms.
- **100% Torch-Free & ONNX-Compliant**: Integrates with local ONNX models using `onnxruntime-genai` for local inference without the PyTorch overhead.
- **Offline Operation**: Runs completely on CPU without internet requests.

---

## Installation

```bash
pip install -e ./slm_voice
```

---

## API Reference

### `SLMVoiceAgent`

```python
from slm_voice import SLMVoiceAgent

agent = SLMVoiceAgent(
    model_path=None,             # Path to local Qwen ONNX model weights
    tools=None,                  # Dictionary of pre-registered tool function mappings
    system_prompt=None,          # Custom system instructions
    n_threads=4,                 # Number of CPU threads for inference
    temperature=0.7,             # Sampling temperature
    top_p=0.9,                   # Nucleus sampling cutoff threshold
    max_tokens=256               # Maximum output token generation limit per response
)
```

#### `process_speech_text(speech_transcript=None, audio_file=None, language="english", output_audio_path=None, ...)`
Processes voice transcript strings or audio files and returns structured synthesizable response blocks.
- **Arguments**:
  - `speech_transcript` (str): Text transcription of spoken voice commands.
  - `audio_file` (str): Path to audio file to transcribe.
  - `language` (str): Language of input/output (e.g. "Hindi", "Tamil", "English").
  - `output_audio_path` (str): Optional path to save the synthesized TTS audio.
- **Returns**:
  - `dict`:
    ```python
    {
        "transcript": str,           # Echo of the user's spoken command
        "response": str,             # Natural language voice agent reply (translated)
        "audio_synthesized": bool    # Status of TTS speech generation check
    }
    ```

---

## Usage Example

```python
from slm_voice import SLMVoiceAgent

agent = SLMVoiceAgent()
agent.register_tool("RAG", lambda q: f"RAG response for {q}")

# Query using Hindi speech input
result = agent.process_speech_text(
    speech_transcript="आरएजी प्रश्न",
    language="Hindi"
)

print(f"User Said: {result['transcript']}")
print(f"Assistant Replied: {result['response']}")
```

### Input & Output Example

#### Input (Voice Transcript):
```text
आरएजी प्रश्न
```

#### Output:
```json
{
  "transcript": "आरएजी प्रश्न",
  "response": "[HI Translation of 'RAG response: RAG response for RAG query']",
  "audio_synthesized": true
}
```
