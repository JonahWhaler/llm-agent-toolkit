from setuptools import setup, find_packages  # type: ignore

DESCRIPTION = "LLM Agent Toolkit provides minimal, modular interfaces for core components in LLM-based applications."

# python3 setup.py sdist bdist_wheel
# twine upload --skip-existing dist/* --verbose

VERSION = "0.0.10"

with open("./README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="llm_agent_toolkit",
    version=VERSION,
    packages=find_packages(),
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="jonah_whaler_2348",
    author_email="jk_saga@proton.me",
    license="GPLv3",
    install_requires=[
        "python-dotenv==0.21.0",
        "openai==1.51.2",
        "ollama==0.4.1",
        "tiktoken==0.8.0",
        "torch==2.5.1",
        "transformers==4.46.2",
        "chromadb==0.5.11",
        "faiss-cpu==1.9.0.post1",
        "aiohttp==3.10.11",
        "pydub==0.25.1",
        "pydub-stubs==0.25.1.4",
        "ffmpeg-python==0.2.0",
        "pdfplumber==0.11.4",
        "PyMuPDF==1.24.11",
        "python-docx==1.1.2",
    ],
    keywords=[
        "llm",
        "agent",
        "toolkit",
        "large language model",
        "memory management",
        "tool integration",
        "multi-modality interaction",
        "multi-step workflow",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.10",
    ],
)
