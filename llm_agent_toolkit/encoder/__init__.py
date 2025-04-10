from .remote import OpenAIEncoder


__all__ = [
    "OpenAIEncoder",
]

# transformers
try:
    from .transformers_emb import TransformerEncoder

    __all__.extend(["TransformerEncoder"])
except:
    pass

# ollama
try:
    from .local import OllamaEncoder

    __all__.extend(["OllamaEncoder"])
except:
    pass

# gemini
try:
    from .gemini import GeminiEncoder

    __all__.extend(["GeminiEncoder"])
except:
    pass
