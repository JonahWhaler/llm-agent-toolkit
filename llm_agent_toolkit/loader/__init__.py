from .msword_loader import MsWordLoader

from .image_loader import ImageToTextLoader
from .pdf_loader import PDFLoader
from .text_loader import TextLoader

# from .a2t2t_loader import A2T2TLoader

__all__ = [
    # "A2T2TLoader",
    "TextLoader",
    "MsWordLoader",
    "PDFLoader",
    "ImageToTextLoader",
]
