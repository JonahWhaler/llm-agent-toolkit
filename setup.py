from setuptools import setup, find_packages  # type: ignore

DESCRIPTION = "LLM Agent Toolkit provides minimal, modular interfaces for core components in LLM-based applications."

# python3 setup.py sdist bdist_wheel
# twine upload --skip-existing dist/* --verbose

VERSION = "0.0.6"

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
    install_requires=[],
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
