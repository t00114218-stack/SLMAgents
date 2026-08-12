# SLM Voice Agent

A local, low-latency CPU-optimized Voice Agent pipeline. It allows conversational speech commands to be processed locally on CPU, with fallback audio transcription synthesis checks.

---

## Features

- **Local Command Processing**: Receives spoken transcripts and returns structured dialogue confirmations.
- **Audio Synthesizer Hook**: Ready to hook into local TTS (Text-to-Speech) engines such as `pyttsx3`.
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

agent = SLMVoiceAgent()
```

#### `process_speech_text(transcript: str) -> dict`
Processes voice transcript strings and returns structured synthesizable response blocks.
- **Arguments**:
  - `transcript` (str): Text transcription of spoken voice commands.
- **Returns**:
  - `dict`:
    ```python
    {
        "transcript": str,           # Echo of the user's spoken command
        "response": str,             # Natural language voice agent reply
        "audio_synthesized": bool    # Status of TTS speech generation check
    }
    ```

---

## Usage Example

```python
from slm_voice import SLMVoiceAgent

agent = SLMVoiceAgent()
spoken_command = "Hello local CPU assistant"

result = agent.process_speech_text(spoken_command)

print(f"User Said: {result['transcript']}")
print(f"Assistant Replied: {result['response']}")
```

### Input & Output Example

#### Input (Voice Transcript):
```text
Hello local CPU assistant
```

#### Output:
```json
{
  "transcript": "Hello local CPU assistant",
  "response": "I heard you ask: 'Hello local CPU assistant'. Processing your query locally on CPU.",
  "audio_synthesized": false
}
```
