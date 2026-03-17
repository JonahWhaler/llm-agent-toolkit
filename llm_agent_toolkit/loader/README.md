# LLM Agent Toolkit - Loaders

This package provides a comprehensive set of loaders designed to ingest various file formats and convert them into text representations suitable for Large Language Model (LLM) agents. The loaders support both synchronous (`load`) and asynchronous (`load_async`) execution.

## Available Loaders

### 1. TextLoader

A lightweight, stateless loader for reading plain text and code files.

- **Supported Extensions:** `.txt`, `.md`, `.py`, `.html`, `.css`, `.js`, `.json`, `.csv`
- **Key Features:**
  - Memory-efficient file reading.
  - Configurable encoding (default: `utf-8`).

```python
from llm_agent_toolkit.loader import TextLoader

loader = TextLoader()
content = loader.load("example.py")
```

### 2. ImageToTextLoader

Uses an `ImageInterpreter` (VLM) to generate textual descriptions for image files.

- **Supported Extensions:** `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`
- **Key Features:**
  - customizable prompt (default: "What's in the image?").

```python
from llm_agent_toolkit.loader import ImageToTextLoader

loader = ImageToTextLoader(image_interpreter=my_vlm_model)
description = loader.load("diagram.png")
```

### 3. MsWordLoader

Parses Microsoft Word documents (`.docx`), extracting text, tables, and handling rich media.

- **Supported Extensions:** `.docx`
- **Key Features:**
  - **Table extraction:** Converts Word tables into Markdown tables.
  - **Image Analysis:** If an `ImageInterpreter` is provided, it extracts embedded images and generates descriptions inline.
  - **Link Summarization:** If a `TextModel` is provided, it fetches and summarizes external hyperlinks found in the document.

```python
from llm_agent_toolkit.loader import MsWordLoader

loader = MsWordLoader(
    text_only=False,
    tmp_directory="./tmp",
    image_interpreter=my_vlm_model,
    text_model=my_llm_model
)
markdown_content = loader.load("report.docx")
```

### 4. PDFLoader

A robust PDF parser combining `PyMuPDF` and `pdfplumber` for high-fidelity extraction.

- **Supported Extensions:** `.pdf`
- **Key Features:**
  - **Layout Preservation:** Attempts to preserve the reading order of text blocks.
  - **Table Extraction:** Detects tables and converts them to Markdown.
  - **Image & Link Intelligence:** Like the Word loader, it can describe images and summarize links if AI models are provided.

```python
from llm_agent_toolkit.loader import PDFLoader

loader = PDFLoader(
    text_only=False,
    tmp_directory="./tmp",
    image_interpreter=my_vlm_model,
    web_interpreter=my_llm_model
)
markdown_content = loader.load("paper.pdf")
```
