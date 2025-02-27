from . import open_ai, local, deep_seek

__all__ = ["open_ai", "deep_seek"]

# gemini
try:
    from . import gemini

    __all__.extend(["gemini"])
except:
    pass

# ollama
try:
    from . import local

    __all__.extend(["local"])
except:
    pass
