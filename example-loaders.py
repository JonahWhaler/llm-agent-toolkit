import asyncio
import logging
from dotenv import load_dotenv
from llm_agent_toolkit import ChatCompletionConfig
from llm_agent_toolkit.core.local import Image_to_Text

logging.basicConfig(
    filename="./snippet/output/example-loaders.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


CONNECTION_STRING = "http://localhost:11434"
VISION_MODEL = "llava:7b"
SYSTEM_PROMPT = "You are Whales, the faithful AI assistant."
CONFIG = ChatCompletionConfig(
    name=VISION_MODEL, return_n=1, max_iteration=1, max_tokens=4096, temperature=1.5
)
IMAGE_INTERPRETER = Image_to_Text(CONNECTION_STRING, SYSTEM_PROMPT, CONFIG)

TMP_DIRECTORY = "./dev"
DOC_PATH = "./dev/sample.docx"
PDF_PATH = "./dev/sample.pdf"
IMG_PATH = "./dev/sample.jpeg"


def exec_image_loader():
    from llm_agent_toolkit.loader import ImageToTextLoader

    ldr = ImageToTextLoader(
        IMAGE_INTERPRETER, "What's in the image? Tell me in details."
    )
    img_description = ldr.load(input_path=IMG_PATH)
    logger.info("%s -> %s", IMG_PATH, img_description)


async def aexec_image_loader():
    from llm_agent_toolkit.loader import ImageToTextLoader

    ldr = ImageToTextLoader(
        IMAGE_INTERPRETER, "What's in the image? Tell me in details."
    )
    img_description = await ldr.load_async(input_path=IMG_PATH)
    logger.info("%s -> %s", IMG_PATH, img_description)


def exec_pdf_loader():
    from llm_agent_toolkit.loader import PDFLoader

    ldr = PDFLoader(text_only=True)
    pdf_content = ldr.load(PDF_PATH)
    export_path = f"{TMP_DIRECTORY}/pdf.md"
    logger.info("%s -> %s", PDF_PATH, export_path)
    with open(export_path, "w", encoding="utf-8") as markdown:
        markdown.write(pdf_content)


async def aexec_pdf_loader():
    from llm_agent_toolkit.loader import PDFLoader

    ldr = PDFLoader(text_only=True)
    pdf_content = await ldr.load_async(PDF_PATH)
    export_path = f"{TMP_DIRECTORY}/pdf.md"
    logger.info("%s -> %s", PDF_PATH, export_path)
    with open(export_path, "w", encoding="utf-8") as markdown:
        markdown.write(pdf_content)


def exec_pdf_loader_w_ii():
    from llm_agent_toolkit.loader import PDFLoader

    ldr = PDFLoader(
        text_only=False,
        tmp_directory=TMP_DIRECTORY,
        image_interpreter=IMAGE_INTERPRETER,
    )
    pdf_content = ldr.load(PDF_PATH)
    export_path = f"{TMP_DIRECTORY}/pdf_w_image-description.md"
    logger.info("%s -> %s", PDF_PATH, export_path)
    with open(export_path, "w", encoding="utf-8") as markdown:
        markdown.write(pdf_content)


async def aexec_pdf_loader_w_ii():
    from llm_agent_toolkit.loader import PDFLoader

    ldr = PDFLoader(
        text_only=False,
        tmp_directory=TMP_DIRECTORY,
        image_interpreter=IMAGE_INTERPRETER,
    )
    pdf_content = await ldr.load_async(PDF_PATH)
    export_path = f"{TMP_DIRECTORY}/pdf_w_image-description.md"
    logger.info("%s -> %s", PDF_PATH, export_path)
    with open(export_path, "w", encoding="utf-8") as markdown:
        markdown.write(pdf_content)


def exec_msw_loader():
    from llm_agent_toolkit.loader import MsWordLoader

    ldr = MsWordLoader(text_only=True)
    doc_content = ldr.load(DOC_PATH)
    export_path = f"{TMP_DIRECTORY}/doc.md"
    logger.info("%s -> %s", DOC_PATH, export_path)
    with open(export_path, "w", encoding="utf-8") as markdown:
        markdown.write(doc_content)


async def aexec_msw_loader():
    from llm_agent_toolkit.loader import MsWordLoader

    ldr = MsWordLoader(text_only=True)
    doc_content = await ldr.load_async(DOC_PATH)
    export_path = f"{TMP_DIRECTORY}/doc.md"
    logger.info("%s -> %s", DOC_PATH, export_path)
    with open(export_path, "w", encoding="utf-8") as markdown:
        markdown.write(doc_content)


def exec_msw_loader_w_ii():
    from llm_agent_toolkit.loader import MsWordLoader

    ldr = MsWordLoader(
        text_only=False,
        tmp_directory=TMP_DIRECTORY,
        image_interpreter=IMAGE_INTERPRETER,
    )
    doc_content = ldr.load(DOC_PATH)
    export_path = f"{TMP_DIRECTORY}/doc_w_image-description.md"
    logger.info("%s -> %s", DOC_PATH, export_path)
    with open(export_path, "w", encoding="utf-8") as markdown:
        markdown.write(doc_content)


async def aexec_msw_loader_w_ii():
    from llm_agent_toolkit.loader import MsWordLoader

    ldr = MsWordLoader(
        text_only=False,
        tmp_directory=TMP_DIRECTORY,
        image_interpreter=IMAGE_INTERPRETER,
    )
    doc_content = await ldr.load_async(DOC_PATH)
    export_path = f"{TMP_DIRECTORY}/doc_w_image-description.md"
    logger.info("%s -> %s", DOC_PATH, export_path)
    with open(export_path, "w", encoding="utf-8") as markdown:
        markdown.write(doc_content)


def synchronous_tasks():
    exec_image_loader()
    exec_pdf_loader()
    exec_pdf_loader_w_ii()
    exec_msw_loader()
    exec_msw_loader_w_ii()


async def asynchronous_tasks():
    tasks = [
        aexec_image_loader(),
        aexec_pdf_loader(),
        aexec_pdf_loader_w_ii(),
        aexec_msw_loader(),
        aexec_msw_loader_w_ii(),
    ]
    await asyncio.gather(*tasks)


def try_openai_examples():
    synchronous_tasks()
    asyncio.run(asynchronous_tasks())


if __name__ == "__main__":
    load_dotenv()
    try_openai_examples()
