![Banner](https://raw.githubusercontent.com/JonahWhaler/llm-agent-toolkit/main/images/repo-banner.jpeg)

# LLM Agent Toolkit: Modular Components for AI Workflows
LLM Agent Toolkit provides minimal, modular interfaces for core components in LLM-based applications. Simplify workflows with stateless interaction, embedding encoders, memory management, tool integration, and data loaders, designed for compatibility and scalability. It prioritizes simplicity and modularity by proposing minimal wrappers designed to work across common tools, discouraging direct access to underlying technologies. Specific implementations and examples will be documented separately in a Cookbook (planned).

PyPI: ![PyPI Downloads](https://static.pepy.tech/badge/llm-agent-toolkit)

## Attention!!!
Using this toolkit simplifies integration by providing unified and modular interfaces across platforms. Many configurations are intentionally kept at their default settings to prioritize ease of use. However, most of these components are extensible through abstract classes, allowing developers to define their own desired configurations for greater flexibility. While this approach enhances consistency and reduces complexity, advanced customization may require extending the provided abstractions. 

For developers requiring full-range customization or access to the latest features, it is recommended to consider using native libraries like `ollama` and `openai` directly.

# Table of Contents
- [LLM Agent Toolkit: Modular Components for AI Workflows](#llm-agent-toolkit-modular-components-for-ai-workflows)
  - [Attention!!!](#attention)
- [Table of Contents](#table-of-contents)
- [Dependecies](#dependecies)
- [Installation](#installation)
- [Fundamental Components](#fundamental-components)
  - [Core:](#core)
    - [Example - Ollama](#example---ollama)
    - [Example - OpenAI](#example---openai)
    - [Example - DeepSeek](#example---deepseek)
    - [Example - Tools](#example---tools)
    - [Example - Structured Output](#example---structured-output)
  - [Encoder:](#encoder)
  - [Memory:](#memory)
  - [Tool:](#tool)
  - [Loader:](#loader)
  - [Chunkers:](#chunkers)
- [Planned Feature](#planned-feature)
- [License](#license)

# Dependecies

  * **Ollama:** v0.5.4

# Installation
  ```bash
  # Text Generation + Image Generation
  pip install llm-agent-toolkit

  # transform text to embedding through transformers's API
  pip install llm-agent-toolkit["transformer"] 
  
  # transform audio to text, only works on Ubuntu
  sudo apt install ffmpeg
  pip install llm-agent-toolkit["transcriber"]

  # entire package
  sudo apt install ffmpeg
  pip install llm-agent-toolkit["all"] # entire package
  ```

# Fundamental Components
## Core: 

A stateless chat completion interface to interact with the LLM.

**Purpose**: Serves as the central execution layer that abstracts interaction with the underlying LLM model.

**Features**:
* Supports Text-to-Text and Image-to-Text.
* Enables iterative executions for multi-step workflows.
* Facilitates tool invocation as part of the workflow.
* Support models from OpenAI and Ollama.
* Support `Structured Output`.

### Example - Ollama
```python
from typing import Any
from llm_agent_toolkit import ChatCompletionConfig
from llm_agent_toolkit.core.local import Text_to_Text

CONNECTION_STRING = "http://localhost:11434"
SYSTEM_PROMPT = "You are a faithful assistant."
PROMPT = "Why is the sky blue?"

config = ChatCompletionConfig(
  name="qwen2.5:7b", temperature=0.7
)
llm = Text_to_Text(
  connection_string=CONNECTION_STRING,
  system_prompt=SYSTEM_PROMPT,
  config=config,
  tools=None
)
responses: list[dict[str, Any]] = llm.run(query=PROMPT, context=None)
for response in responses:
    print(response["content"])

```

### Example - OpenAI
```python
from typing import Any
from llm_agent_toolkit import ChatCompletionConfig
from llm_agent_toolkit.core.open_ai import Text_to_Text

SYSTEM_PROMPT = "You are a faithful assistant."
PROMPT = "Why is the sky blue?"

config = ChatCompletionConfig(
  name="gpt-4o-mini", temperature=0.7
)
llm = Text_to_Text(
  system_prompt=SYSTEM_PROMPT,
  config=config,
  tools=None
)
responses: list[dict[str, Any]] = llm.run(query=PROMPT, context=None)
for response in responses:
    print(response["content"])
```

### Example - DeepSeek
```python
from typing import Any
from llm_agent_toolkit import ChatCompletionConfig
from llm_agent_toolkit.core.deep_seek import Text_to_Text

SYSTEM_PROMPT = "You are a faithful assistant."
PROMPT = "Why is the sky blue?"

config = ChatCompletionConfig(
  name="deepseek-chat", temperature=1.0
)
llm = Text_to_Text(
  system_prompt=SYSTEM_PROMPT,
  config=config,
  tools=None
)
responses: list[dict[str, Any]] = llm.run(query=PROMPT, context=None)
for response in responses:
    print(response["content"])
```

### Example - Tools
```python
from typing import Any
from llm_agent_toolkit import ChatCompletionConfig
from llm_agent_toolkit.core.local import Text_to_Text
# This example is also compatible with llm_agent_toolkit.core.open_ai.Text_to_Text
# This example is also compatible with llm_agent_toolkit.core.deep_seek.Text_to_Text
from llm_agent_toolkit.tool import LazyTool

def adder(a: int, b: int) -> int:
    """Add a with b.

    Args:
        a (int): The first number.
        b (int): The second number.

    Returns:
        int: Results
    """
    return a + b


async def divider(a: int, b: int) -> float:
    """Divide a by b.

    Args:
        a (int): The first number.
        b (int): The second number.

    Returns:
        float: Results

    Raises:
        ValueError: When b is 0
    """
    if b == 0:
        raise ValueError("Division by zero.")
    return a / b


CONNECTION_STRING = "http://localhost:11434"
SYSTEM_PROMPT = "You are a faithful assistant."
PROMPT = "10 + 5 / 5 = ?"

add_tool = LazyTool(adder, is_coroutine_function=False)
div_tool = LazyTool(divider, is_coroutine_function=True)

llm = Text_to_Text(
    connection_string=CONNECTION_STRING,
    system_prompt=SYSTEM_PROMPT,
    config=config,
    tools=[add_tool, div_tool],
)

responses: list[dict[str, Any]] = llm.run(query=PROMPT, context=None)
for response in responses:
    print(response["content"])
```

### Example - Structured Output
```python
from pydantic import BaseModel

from llm_agent_toolkit import ChatCompletionConfig, ResponseMode
from llm_agent_toolkit.core.local import Text_to_Text_SO
# Other Core that support Structured Output
### * llm_agent_toolkit.core.local.Image_to_Text_SO
### * llm_agent_toolkit.core.open_ai.OAI_StructuredOutput_Core
### * llm_agent_toolkit.core.deep_seek.Text_to_Text_SO # JSON only
# These `Core` does not support `Tool` and multi iteration execution.
# If desired, caller can call `llm.run` iteratively with progressively updated `context`.
# File example-chain.py shows steps to achieve chained execution.

# Define the Schema through pydantic BaseModel
class QnA(BaseModel):
    question: str
    answer: str


CONNECTION_STRING = "http://localhost:11434"
SYS_PROMPT = "You are a faithful Assistant."
PROMPT = "Write a blog post about physician-assisted suicide (euthanasia)."
CONFIG = ChatCompletionConfig(
    name="llama3.2:3b", temperature=0.3, max_tokens=2048, max_output_tokens=1024
)
# Structured Output via Pydantic BaseModel
llm = Text_to_Text_SO(
    connection_string=CONNECTION_STRING, system_prompt=SYS_PROMPT, config=CONFIG,
)
response_1 = llm.run(
    query=PROMPT, context=None, mode=ResponseMode.SO, format=QnA,
)[0]

# Structured Output via JSON Mode
### It's essential to mention the expected JSON structure in the prompt 
### and highlight it to return in JSON format.
SPROMPT = f"""
You are a helpful assistant.

Response Schema:
{
    json.dumps(QnA.model_json_schema())
}

Note:
Always response in JSON format without additional comments or explanation.
"""

llm = Text_to_Text_SO(
    connection_string=CONNECTION_STRING, system_prompt=SPROMPT, config=CONFIG,
)
response_2 = llm.run(
    query=PROMPT, context=None, mode=ResponseMode.JSON
)
```

## Encoder:
A standardized wrapper for embedding models.

**Purpose**: Provides a minimal API to transform text into embeddings, usable with any common embedding model.

**Features**:
* Abstracts away model-specific details (e.g., dimensionality, framework differences).
* Allows flexible integration with downstream components like Memory or retrieval mechanisms.
* Support OpenAI, Ollama and Transformers.
* Support asynchronous operation.

## Memory: 
Offers essential context retention capabilities.

**Purpose**: Allows efficient context management without hardcoding database or memory solutions.

**Types**:
1. *Short-term Memory*:
    * Maintains recent interactions for session-based context.
2. *Vector Memory*:
    * Combines embedding and storage for retrieval-augmented workflows.
    * Includes optional metadata management for filtering results.
    * Support Faiss and Chroma
3. *Async Vector Memory*:
    * Same as Vector Memory with async support.

## Tool:
A unified interface for augmenting the LLM's functionality.

**Purpose**: Provides a lightweight abstraction for tool integration, accommodating both simple and complex tools.

**Features**:
* *Simple Tools*: Lazy wrappers for functions or basic utilities.
* *Complex Tools*: Abstract class for external APIs or multi-step operations.

## Loader:
Responsible for converting raw data into text.

**Purpose**: Handles preprocessing and content extraction from diverse formats.

**Features**:
* Covering limited type of documents, images, and audio files.

## Chunkers:
Utility to split long text into chunks.

**Features**:
* **Basic**: 
  * *FixedCharacterChunker*: Split text into fixed-size character chunks with optional overlapping.
  * *FixedGroupChunker*: Splits text into K chunks. Supporting two levels, `word` and `character`, default is `character`.
* **Semantic**:
  * *SemanticChunker*: Split text into semantically coherent chunks.
  * *SimulatedAnnealingSemanticChunker*: Enhanced with Simulated Annealing optimization technique.

# Planned Feature
- A Cookbook with detailed implementation examples.

# License
This project is licensed under the GNU General Public License v3.0 License. See the [LICENSE](LICENSE) file for details.
