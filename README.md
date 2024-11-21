# llm-agent-toolkit
This project defines essential interfaces for fundamental components needed in LLM-based applications. It prioritizes simplicity and modularity by proposing minimal wrappers designed to work across common tools, discouraging direct access to underlying technologies. Specific implementations and examples will be documented separately in a Cookbook (planned).

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
