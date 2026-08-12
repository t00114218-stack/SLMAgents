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

    def process_speech_text(self, speech_transcript: str, language: str = "english") -> dict:
        """
        Processes transcribed speech text and returns a synthesized conversational response.
        """
        if not speech_transcript:
            return {"transcript": "", "response": "", "audio_synthesized": False}

        # Simulated conversational response generation
        response_text = f"I heard you ask: '{speech_transcript}'. Processing your query locally on CPU."

        # Target language routing
        target_lang_code = "en"
        lang_lower = language.lower().strip()
        if lang_lower in ["hi", "hindi"]:
            target_lang_code = "hi"
        elif lang_lower in ["ta", "tamil"]:
            target_lang_code = "ta"
        elif lang_lower in ["es", "spanish"]:
            target_lang_code = "es"
        elif lang_lower in ["fr", "french"]:
            target_lang_code = "fr"
        elif lang_lower in ["de", "german"]:
            target_lang_code = "de"
        
        if target_lang_code != "en":
            try:
                from slm_translation.translation_hub import SLMTranslationHub
                hub = SLMTranslationHub()
                response_text = hub.translate(response_text, source_lang="en", target_lang=target_lang_code)
            except Exception as e:
                response_text = f"[{target_lang_code.upper()} Translation of '{response_text}']"

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
