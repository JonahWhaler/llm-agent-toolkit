from .open_ai_tts import OpenAITTS

__all__ = ["OpenAITTS"]


# ElevenLabs
try:
    from .elevenlabs_tts import ElevenLabsTTS

    __all__.extend(["ElevenLabsTTS"])
except ImportError:
    pass
