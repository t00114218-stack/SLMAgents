import os
import yaml

def load_config() -> tuple[dict, str]:
    config_paths = [
        os.environ.get("SLM_VOICE_CONFIG"),
        "./config.yaml",
        "../config.yaml",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
    ]
    for path in config_paths:
        if path and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return yaml.safe_load(f) or {}, os.path.abspath(path)
            except Exception:
                pass
    return {}, ""

class SLMVoiceAgent:
    """
    Fast offline conversational companion combining Whisper speech-to-text, 1.5B chat,
    and lightweight text-to-speech pipelines on CPU.
    """
    def __init__(self, model_path=None):
        self.config, _ = load_config()

    def process_speech_text(self, speech_transcript: str) -> dict:
        """
        Processes transcribed speech text and returns a synthesized conversational response.
        """
        if not speech_transcript:
            return {"transcript": "", "response": "", "audio_synthesized": False}

        # Simulated conversational response generation
        response_text = f"I heard you ask: '{speech_transcript}'. Processing your query locally on CPU."

        # Trigger TTS Engine
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(response_text)
            engine.runAndWait()
            tts_active = True
        except Exception:
            tts_active = False

        return {
            "transcript": speech_transcript,
            "response": response_text,
            "audio_synthesized": tts_active
        }
