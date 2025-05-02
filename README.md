# RAG Pipeline with SQLite, Qdrant, and Streamlit Q&A

This project implements a Retrieval-Augmented Generation (RAG) pipeline using SQLite for storing text chunks, Qdrant for vector search, and integrates with a local LLM (via Ollama) to answer questions through a Streamlit interface.

## Tech Stack

*   **Backend API:** [FastAPI](https://fastapi.tiangolo.com/) (Serves RAG logic via REST API, run with [Uvicorn](https://www.uvicorn.org/))
*   **Frontend:** [Streamlit](https://streamlit.io/) (Connects to the FastAPI backend)
*   **Vector Database:** [Qdrant](https://qdrant.tech/) (Run locally via Docker)
*   **SQL Database:** [SQLite](https://www.sqlite.org/) (Stores document/chunk metadata)
*   **Embeddings:** [Sentence Transformers](https://www.sbert.net/) (Configurable model via `config.yaml`, default: `all-mpnet-base-v2`)
*   **Reranker (Optional):** [sentence-transformers Cross-Encoders](https://www.sbert.net/docs/usage/cross-encoder.html) (e.g., `bge-reranker-base`, requires `--rerank` flag in CLI or toggle in UI)
*   **LLM Integration:** [Ollama](https://ollama.com/) API
    *   **Provider:** Currently supports `ollama` (set in `config.yaml` -> `llm.provider`).
    *   **Model:** Configurable via `config.yaml` -> `llm.model_name` (Default: `llama3:latest`). Ensure the specified model is pulled using `ollama pull <model_name>` and that Ollama is running.
    *   **API Endpoint:** Configurable via `config.yaml` -> `llm.api_base_url` (Default: `http://localhost:11434`).
    *   **Changing the LLM:** To use a different Ollama model (e.g., `mistral:latest`), update `llm.model_name` in `config.yaml` and run `ollama pull mistral:latest`.
*   **Prompt Templating:** Defined in `config.yaml` under the `prompts` dictionary, allowing different instruction sets (e.g., `gatsby`, `financial_analyst`).
*   **JSON Directory Ingestion:** Configurable via `config.yaml` under `json_ingestion` (`input_directory`, `file_limit`).
*   **CLI:** [Typer](https://typer.tiangolo.com/) (`src/cli.py`) for backend operations (DB init, ingestion, querying).
*   **Packaging:** [uv](https://github.com/astral-sh/uv) for dependency management.

## Workflow

### 1. Ingestion Pipeline

This process loads documents, extracts text, chunks it, generates embeddings, and stores the information in the databases.

*   **Initiation:** Triggered via CLI commands (`ingest` for single files, `ingest-json-chunks` for pre-chunked JSON).
*   **Loading:** Text is loaded from the source file (`.txt`) or pre-computed chunks are loaded from JSON (`src/ingestion/loaders.py`).
*   **Document Record (SQLite):** A record for the document (name, source path) is created in the SQLite database (`src/database/crud.py`).
*   **Chunking (if applicable):** If not using pre-computed chunks, the text is split into smaller, overlapping chunks based on `config.yaml` settings (`src/ingestion/chunking.py`).
*   **Chunk Records (SQLite):** Each chunk is stored in the SQLite database, linked to its parent document (`src/database/crud.py`). Each chunk gets a unique `chunk_id`.
*   **Embedding Generation:** Vector embeddings are generated for each text chunk using the configured Sentence Transformer model (`src/embeddings.py`).
*   **Vector Upsert (Qdrant):** Each chunk's embedding is upserted into the Qdrant collection (`src/vector_store/indexing.py`). The `point_id` in Qdrant corresponds to the SQLite `chunk_id`. The payload includes metadata like document name, source, and a text preview.

### 2. Query Pipeline

This process takes a user query, retrieves relevant context, and generates an answer using the LLM, orchestrated via the API.

*   **User Input (UI):** A question is submitted via the Streamlit UI (`src/ui/app.py`).
*   **API Request (UI -> API):** Streamlit sends the query, source/prompt selections, and options to the FastAPI backend (`/query/` endpoint).
*   **Query Handling (API):** The FastAPI backend (`src/api/main.py`) receives the request.
*   **Query Embedding (API):** The user's question is converted into a vector embedding.
*   **Vector Search (API -> Qdrant):** The query embedding searches Qdrant for similar chunks, applying filters.
*   **Reranking (API - Optional):** Retrieved chunks can be reranked for relevance.
*   **Chunk Retrieval (API -> SQLite):** Full text for top chunks is retrieved from SQLite.
*   **Prompt Formatting (API):** Retrieved context and question are formatted using the selected template.
*   **LLM Interaction (API -> Ollama):** The prompt is sent to the Ollama LLM.
*   **Answer Generation (API):** The LLM generates an answer.
*   **API Response (API -> UI):** The FastAPI backend sends the final answer and details of retrieved chunks back to Streamlit as JSON.
*   **Display (UI):** Streamlit displays the answer and chunk details received from the API.

## Setup and Usage

1.  **Prerequisites:**
    *   Python 3.10+
    *   Docker (for running local Qdrant)
    *   `uv` (Python package manager): `pip install uv`
    *   **Ollama:** Install Ollama from [https://ollama.com/](https://ollama.com/) and ensure it's running.
    *   **Ollama Model:** Pull the desired LLM. The default is `llama3:latest`. You can change this in `config.yaml`.
        ```bash
        ollama pull llama3:latest 
        # or e.g., ollama pull mistral:latest
        ```

2.  **Install Dependencies:**
    *   Install core + API + App dependencies (recommended for running both frontend and backend):
        ```bash
        uv sync --all-extras
        # Or specifically: uv sync --extra api --extra app
        ```

3.  **Configure LLM (if needed):**
    *   Edit `config.yaml`.
    *   Verify/update `llm.model_name` to match the model you pulled with Ollama.
    *   Verify/update `llm.api_base_url` if your Ollama service is running on a different address/port.

4.  **Start Qdrant:**
    *   Make sure Docker is running.
    *   Make the script executable: `chmod +x start_qdrant.sh`
    *   Run the script: 
        ```bash
        bash start_qdrant.sh 
        ```
    *   This starts the Qdrant container.

5.  **Initialize Database:**
    *   Create the SQLite database and tables:
        ```bash
        python -m src.cli init-db
        ```
    *   (Optional) To wipe and re-initialize: `python -m src.cli init-db --force`

6.  **Ingest Documents:**
    *   **Option A: Ingest Single File:**
        ```bash
        python -m src.cli ingest path/to/your/document.txt
        ```
    *   **Option B: Ingest Single Pre-Chunked JSON File:**
        ```bash
        python -m src.cli ingest-json-chunks path/to/your/chunks.json
        ```
    *   **Option C: Ingest Multiple JSON Files from Directory:**
        *   First, configure the `json_ingestion.input_directory` path in `config.yaml`.
        *   Optionally, set `json_ingestion.file_limit`.
        *   Run the command:
            ```bash
            python -m src.cli ingest-json-dir
            ```

7.  **Run the Backend API Server:**
    *   Ensure Ollama is running with the configured model.
    *   Ensure Qdrant is running.
    *   Start the FastAPI server (defaults to port 8000):
        ```bash
        uvicorn src.api.main:app --reload --port 8000
        ```
    *   You can check if it's running by visiting `http://127.0.0.1:8000/docs` in your browser.

8.  **Run the Streamlit Application:**
    *   Open a **new terminal window/tab** (keep the Uvicorn server running).
    *   Navigate to your project directory.
    *   Start the Streamlit app:
        ```bash
        streamlit run src/ui/app.py
        ```
    *   Open the Streamlit URL (usually `http://localhost:8501`) in your browser, select sources/prompts, ask questions, and view the answers.

9.  **(Optional) CLI Verification/Inspection:**
    *   The CLI commands (`init-db`, `ingest`, `ingest-json-chunks`, `inspect-db`) still work directly with the backend components (database, Qdrant) for management and testing.
    *   The `ingest-json-dir` command uses the configuration in `config.yaml`.
    *   The CLI `query` command also works directly, bypassing the FastAPI layer.
    *   The `evaluate-deere` command runs a predefined question set (from `src/evaluation/deere_eval_set.py`) against a specified source, comparing generated vs. expected answers.

10. **(Optional) Run Example Module Scripts:**
    *   Most modules (`*.py` in `src/`) have `if __name__ == "__main__":` test blocks.
        ```bash
        python -m src.embeddings
        python -m src.reranker
        # etc.
        ```

## Example: Resetting and Loading a Specific Dataset

These steps show how to clear all existing data and load only a single dataset (either the Deere JSON or The Great Gatsby).

### Option 1: Loading Deere JSON Data Only

Follow these steps to clear all existing data (including Gatsby) and load only the Deere financial data from the `DE_2022_derived.json` file.

1.  **Place Data File:** Ensure the `DE_2022_derived.json` file is in the root directory of the `rag-pipeline` project.

2.  **Reset Database and Vector Store:** Run the following command to wipe the SQLite database and recreate the Qdrant collection. **Warning:** This deletes all previously ingested data.
    ```bash
    python -m src.cli init-db --force
    ```
    *(Confirm 'y' when prompted)*

3.  **Ingest Deere JSON Chunks:** Run the command to load the pre-chunked JSON file using the `original_text` field (as currently configured in `src/ingestion/loaders.py`):
    ```bash
    # This assumes DE_2022_derived.json is the ONLY file in the configured directory, 
    # OR you have set file_limit=1 in config.yaml
    python -m src.cli ingest-json-dir 
    # Alternatively, use the single file command if directory setup is inconvenient:
    # python -m src.cli ingest-json-chunks DE_2022_derived.json
    ```

4.  **Start Backend and Frontend:**
    *   Start the FastAPI backend in one terminal:
        ```bash
        uvicorn src.api.main:app --reload --port 8000
        ```
    *   Start the Streamlit frontend in another terminal:
        ```bash
        streamlit run src/ui/app.py
        ```

5.  **Query the Data:** In the Streamlit UI:
    *   Select the `financial_analyst` prompt.
    *   The source dropdown should show `json_chunks:DE_2022_derived.json`.
    *   Ask your questions about the Deere 2022 financial report.

### Option 2: Loading The Great Gatsby Only

Follow these steps to clear all existing data (including Deere data) and load only "The Great Gatsby".

1.  **Place Data File:** Ensure the `gatsby.txt` file (downloaded from Project Gutenberg or using `curl -o gatsby.txt https://www.gutenberg.org/cache/epub/64317/pg64317.txt`) is in the root directory of the `rag-pipeline` project.

2.  **Reset Database and Vector Store:** Run the following command to wipe the SQLite database and recreate the Qdrant collection. **Warning:** This deletes all previously ingested data.
    ```bash
    python -m src.cli init-db --force
    ```
    *(Confirm 'y' when prompted)*

3.  **Ingest Gatsby Text:** Run the command to load the text file:
    ```bash
    python -m src.cli ingest gatsby.txt
    ```

4.  **Start Backend and Frontend:**
    *   Start the FastAPI backend in one terminal:
        ```bash
        uvicorn src.api.main:app --reload --port 8000
        ```
    *   Start the Streamlit frontend in another terminal:
        ```bash
        streamlit run src/ui/app.py
        ```

5.  **Query the Data:** In the Streamlit UI:
    *   Select the `gatsby` prompt.
    *   The source dropdown should show `file:gatsby.txt`.
    *   Ask your questions about "The Great Gatsby". 
# rag-pipeline
Rag Pipeline built using Qdrant &amp; llama3 by the NeuralKnights team 
