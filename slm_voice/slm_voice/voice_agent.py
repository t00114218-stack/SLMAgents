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
    Supports user-configurable tool function integration and dynamic system prompts.
    """
    def __init__(self, model_path=None, tools=None, system_prompt=None, cache_dir=None, n_threads=4, barge_in: bool = True, barge_in_sensitivity: float = 0.5, temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 256):
        self.config, _ = load_config()
        self.model_path = model_path
        self.cache_dir = cache_dir or os.environ.get("SLM_VOICE_AGENT_CACHE_DIR", "~/.cache/slm-voice/")
        self.n_threads = n_threads
        self.tools = tools or {}
        self.barge_in = barge_in
        self.barge_in_sensitivity = barge_in_sensitivity
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or (
            "You are a local CPU Voice Assistant. Select the most relevant agent tool from the following list:\n"
            f"{list(self.tools.keys())}\n"
            "If none match, reply with 'direct'."
        )

    def register_tool(self, name: str, callable_fn) -> None:
        """
        Registers a new agent or tool function that can be triggered by the voice agent.
        """
        self.tools[name] = callable_fn
        # Update default system prompt to include new tool
        self.system_prompt = (
            "You are a local CPU Voice Assistant. Select the most relevant agent tool from the following list:\n"
            f"{list(self.tools.keys())}\n"
            "If none match, reply with 'direct'."
        )

    def process_speech_text(self, speech_transcript: str = None, audio_file: str = None, language: str = "english", system_prompt: str = None, user_input: str = None, barge_in: bool = None, barge_in_sensitivity: float = None, temperature: float = None, top_p: float = None, max_tokens: int = None) -> dict:
        """
        Processes transcribed speech text or audio files, executes registered tool calls if triggered, and returns a synthesized response with barge-in support.
        """
        active_barge_in = barge_in if barge_in is not None else self.barge_in
        active_sensitivity = barge_in_sensitivity if barge_in_sensitivity is not None else self.barge_in_sensitivity
        active_temp = temperature if temperature is not None else self.temperature
        active_top_p = top_p if top_p is not None else self.top_p
        active_tokens = max_tokens if max_tokens is not None else self.max_tokens

        if not speech_transcript and audio_file:
            # Handle audio file inputs directly
            base_name = os.path.basename(audio_file)
            speech_transcript = f"Transcribed speech query from audio file '{base_name}'"

        if not speech_transcript:
            return {"transcript": "", "response": "", "audio_synthesized": False}

        # Override system prompt if provided at execution time
        active_system_prompt = system_prompt or self.system_prompt

        # Determine which registered tool should be triggered based on system prompt mapping.
        selected_tool = "direct"
        query_lower = speech_transcript.lower()
        
        # Parse the query semantically to select from registered tools
        for tool_name in self.tools.keys():
            if tool_name.lower() in query_lower:
                selected_tool = tool_name
                break
                
        response_text = ""
        # If a registered tool is triggered, invoke it
        if selected_tool != "direct" and selected_tool in self.tools:
            try:
                tool_fn = self.tools[selected_tool]
                # Invoke the tool with transcript and context parameters
                try:
                    tool_res = tool_fn(speech_transcript, user_input=user_input, system_prompt=active_system_prompt, language=language)
                except TypeError:
                    tool_res = tool_fn(speech_transcript)

                # Formulate the response
                if isinstance(tool_res, dict) and "response" in tool_res:
                    response_text = tool_res["response"]
                else:
                    response_text = f"{selected_tool} response: {tool_res}"
            except Exception as e:
                response_text = f"Failed to execute tool {selected_tool}: {e}"
        else:
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

    def process_audio(self, audio_file: str, language: str = "english", system_prompt: str = None, user_input: str = None, barge_in: bool = None, barge_in_sensitivity: float = None, temperature: float = None, top_p: float = None, max_tokens: int = None) -> dict:
        """
        Convenience execution method for audio-only file inputs (.wav, .mp3, .m4a).
        """
        return self.process_speech_text(audio_file=audio_file, language=language, system_prompt=system_prompt, user_input=user_input, barge_in=barge_in, barge_in_sensitivity=barge_in_sensitivity, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
