# LLM Agent Toolkit: Modular Components for AI WOrkflows
LLM Agent Toolkit provides minimal, modular interfaces for core components in LLM-based applications. Simplify workflows with stateless interaction, embedding encoders, memory management, tool integration, and data loaders, designed for compatibility and scalability. It prioritizes simplicity and modularity by proposing minimal wrappers designed to work across common tools, discouraging direct access to underlying technologies. Specific implementations and examples will be documented separately in a Cookbook (planned).

![PyPI Downloads](https://static.pepy.tech/badge/llm-agent-toolkit)

## Table of Contents
- [LLM Agent Toolkit: Modular Components for AI WOrkflows](#llm-agent-toolkit-modular-components-for-ai-workflows)
  - [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Fundamental Components](#fundamental-components)
  - [Core:](#core)
  - [Encoder:](#encoder)
  - [Memory:](#memory)
  - [Tool:](#tool)
  - [Loader:](#loader)
- [Planned Feature](#planned-feature)
- [License](#license)

# Installation
  `
  pip install llm-agent-toolkit
  `
  
# Fundamental Components
## Core: 

A stateless interface to interact with the LLM.

**Purpose**: Serves as the central execution layer that abstracts interaction with the underlying LLM model.

**Features**:
* Supports multiple input-output modalities (e.g., Text-to-Text, Text-to-Image).
* Enables iterative executions for multi-step workflows.
* Facilitates tool invocation as part of the workflow.

## Encoder:
A standardized wrapper for embedding models.

**Purpose**: Provides a minimal API to transform text into embeddings, usable with any common embedding model.

**Features**:
* Abstracts away model-specific details (e.g., dimensionality, framework differences).
* Allows flexible integration with downstream components like Memory or retrieval mechanisms.

## Memory: 
Offers essential context retention capabilities.

**Purpose**: Allows efficient context management without hardcoding database or memory solutions.

**Types**:
1. *Short-term Memory*:
    * Maintains recent interactions for session-based context.
2. *Vector Memory*:
    * Combines embedding and storage for retrieval-augmented workflows.
    * Includes optional metadata management for filtering results.

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

# Planned Feature
- A Cookbook with detailed implementation examples.
- Release to PyPI.

# License
This project is licensed under the GNU General Public License v3.0 License. See the [LICENSE](LICENSE) file for details.