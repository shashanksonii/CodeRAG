# Codebase RAG Assistant

Codebase RAG Assistant is a production-grade, local-first RAG (Retrieval-Augmented Generation) system that parses, indexes, and queries codebases. It provides grounded, source-attributed answers to developers' natural language questions about their repositories, showing visual citations and code previews.

## Features

- **AST-Based Semantic Chunking (Python)**: Intelligently parses Python files using Abstract Syntax Trees (AST) to split at class definitions, class headers, functions, methods, and module-level imports/constants, preserving docstrings and hierarchy.
- **Incremental & Idempotent Indexing**: Uses content hashes (MD5) to track file modifications. Re-indexing only processes new or changed files, and deletes chunks from deleted files.
- **Model-Agnostic Generation**: Switch dynamically between local inference via **Ollama (Llama 3.1)** or API-driven inference via **Anthropic Claude**.
- **Zero-Cost Embeddings**: Local text embedding generation remains 100% free and offline using Ollama's `nomic-embed-text` regardless of the LLM generation backend.
- **Citation Cross-Checking**: Automatically extracts inline LLM citations (e.g., `[file_path:start_line-end_line]`), cross-checks them against retrieved document metadata to prevent hallucinations, and shows actual syntax-highlighted code snippets.
- **Premium UI**: Crafted with dynamic animations, modern glassmorphic sidebar layout, dark mode, custom status containers, and expandable citation trees.

---

## Technical Stack

- **Orchestration**: LangChain
- **Vector Store**: ChromaDB (Persistent, Local, `./chroma_db`)
- **Local Embedding**: Ollama (`nomic-embed-text`)
- **Generation Options**: Ollama (`llama3.1`) / Anthropic Claude API (`claude-3-5-sonnet-latest`)
- **UI Engine**: Streamlit

---

## Setup & Installation

### Step 1: Install Ollama & Pull Models
1. Download and install [Ollama](https://ollama.com/) on your local machine.
2. Open a terminal and start the server:
   ```bash
   ollama serve
   ```
3. Pull the embedding and generation models:
   ```bash
   ollama pull nomic-embed-text
   # Only required if using the Ollama generation backend
   ollama pull llama3.1
   ```

### Step 2: Install Python Dependencies
1. Clone or locate the repository directory:
   ```bash
   cd /path/to/codebase-rag-assistant
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Step 3: Setup Environment Variables
1. Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and add your `ANTHROPIC_API_KEY` (if you plan to use Claude):
   ```env
   ANTHROPIC_API_KEY=your_actual_anthropic_api_key_here
   OLLAMA_HOST=http://localhost:11434
   ```

---

## Running the Application

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Open the URL shown in your terminal (usually `http://localhost:8501`).
3. Enter the absolute path to any local codebase directory in the sidebar input box (e.g. `/Users/username/my-project`).
4. Click **Index / Re-index** to scan the files and create embeddings in `./chroma_db`.
5. Enter a prompt in the chat box to ask questions like:
   - *"How is the AST chunking implemented in ast_chunker.py?"*
   - *"Explain the initialization of the Chroma DB client."*
   - *"What files are indexed by CodebaseIndexer?"*

---

## Project Structure

```
├── app.py                    # Streamlit entrypoint & UI
├── indexer/
│   ├── __init__.py
│   ├── ast_chunker.py        # AST-based Python semantic chunker
│   ├── generic_chunker.py    # Fallback Recursive character splitter
│   └── codebase_indexer.py   # Codebase walking, .gitignore matching, hash-based indexing
├── rag/
│   ├── __init__.py
│   ├── retriever.py          # Chroma search & threshold filter
│   ├── prompt_templates.py   # Grounding prompt, citation extraction
│   └── llm_backends.py       # Ollama & Claude backend integrations
├── vectorstore/
│   ├── __init__.py
│   └── chroma_client.py      # Chroma client connection and SafeOllamaEmbeddings wrapper
├── requirements.txt          # Python package requirements
├── .env.example              # Dotenv template
└── README.md                 # Project documentation
```
