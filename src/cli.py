import typer
from typing_extensions import Annotated, Optional
from pathlib import Path
import logging
import pprint
import sys

# Configure logging for the CLI
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create the Typer app
app = typer.Typer(
    name="rag-pipeline-cli",
    help="CLI for interacting with the RAG Pipeline (SQLite + Qdrant).",
    add_completion=False
)

# --- Helper Functions (Import after app is created to avoid circular issues if needed) ---
# We defer imports into the functions where they are used to ensure modules are loaded correctly.

# --- CLI Commands ---

@app.command()
def init_db(
    force: Annotated[bool, typer.Option("--force", help="Force recreation of database tables (data will be lost!).")] = False
):
    """Initializes or re-initializes the SQLite database tables and optionally the Qdrant collection."""
    # Import necessary functions within the command
    try:
        from src.database.core import init_db as init_db_func
        from src.database.models import Base
        from src.database.core import engine
        from src.vector_store.client import get_qdrant_client
        from src.vector_store.collections import recreate_collection
        from src.embeddings import get_embedding_dimension
        from src.config import settings # Needed for collection name
    except ImportError as e:
        logger.error(f"Failed to import necessary modules: {e}. Is the src directory in your PYTHONPATH?")
        raise typer.Exit(code=1)

    if force:
        typer.confirm("Are you sure you want to drop all existing SQLite tables and recreate the Qdrant collection? All data will be lost.", abort=True)
        
        # Drop SQLite Tables
        logger.info("Dropping existing database tables...")
        try:
            Base.metadata.drop_all(bind=engine)
            logger.info("SQLite tables dropped successfully.")
        except Exception as e:
            logger.exception(f"Error dropping SQLite tables: {e}")
            # Optionally decide if you want to stop here or still try Qdrant cleanup
            raise typer.Exit(code=1) 
            
        # Recreate Qdrant Collection
        logger.info(f"Recreating Qdrant collection '{settings.qdrant.collection_name}'...")
        try:
            qdrant_client = get_qdrant_client()
            embedding_dim = get_embedding_dimension() # Get dimension for recreation
            recreate_collection(qdrant_client, embedding_dim=embedding_dim)
            logger.info(f"Qdrant collection '{settings.qdrant.collection_name}' recreated successfully.")
        except Exception as e:
             logger.exception(f"Failed to recreate Qdrant collection: {e}")
             # Decide if this should be a fatal error
             typer.echo("Warning: Failed to recreate Qdrant collection. Proceeding with SQLite initialization.", err=True)

    # Initialize SQLite Tables (always runs, unless force failed above)
    logger.info("Initializing SQLite database tables...")
    try:
        init_db_func()
        typer.echo(f"Database tables initialized successfully.")
    except Exception as e:
        logger.exception(f"Failed to initialize database: {e}")
        raise typer.Exit(code=1)

@app.command()
def ingest(
    file: Annotated[Path, typer.Argument(help="Path to the document file to ingest.", exists=True, file_okay=True, dir_okay=False, readable=True)]
):
    """Ingests a single document into the RAG pipeline."""
    try:
        from src.ingestion.pipeline import ingest_document_from_file
    except ImportError as e:
        logger.error(f"Failed to import ingestion modules: {e}. Is the src directory in your PYTHONPATH?")
        raise typer.Exit(code=1)

    typer.echo(f"Starting ingestion for: {file}")
    try:
        ingest_document_from_file(str(file))
        typer.echo(f"Ingestion completed for: {file}")
    except Exception as e:
        logger.exception(f"Ingestion failed for {file}: {e}")
        raise typer.Exit(code=1)

@app.command(name="ingest-json-chunks")
def ingest_json_chunks(
    file: Annotated[Path, typer.Argument(help="Path to the JSON file containing a list of chunk objects.", exists=True, file_okay=True, dir_okay=False, readable=True)],
    doc_name: Annotated[Optional[str], typer.Option("--doc-name", help="Explicit document name for the database.")] = None,
    source_prefix: Annotated[str, typer.Option("--source-prefix", help="Prefix for the source description (e.g., 'json_chunks').")] = "json_chunks"
):
    """Ingests pre-computed chunks from a JSON file."""
    try:
        from src.ingestion.loaders import load_chunks_from_json
        from src.ingestion.pipeline import _ingest_text_content
    except ImportError as e:
        logger.error(f"Failed to import ingestion modules: {e}. Is the src directory in your PYTHONPATH?")
        raise typer.Exit(code=1)

    # Determine document name and source description
    resolved_doc_name = doc_name if doc_name else file.stem # Use filename stem if not provided
    source_description = f"{source_prefix}:{file.name}"

    typer.echo(f"Starting ingestion for pre-chunked JSON: {file}")
    typer.echo(f"  Document Name: {resolved_doc_name}")
    typer.echo(f"  Source Description: {source_description}")

    try:
        precomputed_chunks = load_chunks_from_json(file)
        if not precomputed_chunks:
            typer.echo(f"Error: Failed to load chunks from {file}. Check format and logs.", err=True)
            raise typer.Exit(code=1)
        
        typer.echo(f"Loaded {len(precomputed_chunks)} chunks. Starting ingestion process...")
        success = _ingest_text_content(
            doc_name=resolved_doc_name,
            source_description=source_description,
            precomputed_chunks=precomputed_chunks
        )

        if success:
            typer.echo(f"Ingestion completed successfully for {resolved_doc_name}.")
        else:
            typer.echo(f"Ingestion failed for {resolved_doc_name}. Check logs.", err=True)
            raise typer.Exit(code=1)

    except Exception as e:
        logger.exception(f"Ingestion failed for {file}: {e}")
        raise typer.Exit(code=1)

@app.command(name="ingest-json-dir")
def ingest_json_directory():
    """Ingests all JSON chunk files from the directory specified in config.yaml."""
    try:
        from src.ingestion.pipeline import ingest_json_files_from_directory
        from src.config import settings # Need settings to get dir/limit
    except ImportError as e:
        logger.error(f"Failed to import ingestion modules: {e}. Is the src directory in your PYTHONPATH?")
        raise typer.Exit(code=1)

    # Check if config section exists
    if not settings.json_ingestion or not settings.json_ingestion.input_directory:
        typer.echo("Error: 'json_ingestion.input_directory' not configured in config.yaml.", err=True)
        raise typer.Exit(code=1)

    input_dir = settings.json_ingestion.input_directory
    file_limit = settings.json_ingestion.file_limit

    typer.echo(f"Starting ingestion from directory: {input_dir}")
    if file_limit:
        typer.echo(f"File limit: {file_limit}")
    else:
        typer.echo("File limit: None (process all *.json files)")

    try:
        # The function itself logs detailed progress and summary
        ingest_json_files_from_directory(input_dir=input_dir, file_limit=file_limit)
        typer.echo("Directory ingestion process finished. Check logs for details.")
    except Exception as e:
        # Catch unexpected errors during the overall process
        logger.exception(f"Directory ingestion failed: {e}")
        raise typer.Exit(code=1)

@app.command(name="evaluate-deere")
def evaluate_deere(
    source: Annotated[str, typer.Option(..., "--source", "-s", help="REQUIRED: Source description to filter results (e.g., 'json_dir:DE_2022_derived.json').")],
    prompt_name: Annotated[str, typer.Option(help="Name of the prompt template to use.")] = "financial_analyst",
    top_k: Annotated[int, typer.Option("-k", help="Number of top results to retrieve for context.")] = 5,
    rerank: Annotated[bool, typer.Option("--rerank/--no-rerank", help="Enable/disable the reranker model.")] = False
):
    """Runs the predefined Deere evaluation question set against a specific data source."""
    try:
        from src.evaluation.deere_eval_set import DEERE_EVAL_QUESTIONS
        from src.query import query_rag
        from src.llm_interface import get_llm_answer
        import time
    except ImportError as e:
        logger.error(f"Failed to import necessary modules: {e}.")
        raise typer.Exit(code=1)

    typer.echo(f"--- Running Deere Evaluation Question Set --- ")
    typer.echo(f"Target Source: {source}")
    typer.echo(f"Using Prompt: {prompt_name}")
    typer.echo(f"Top K: {top_k}, Rerank: {rerank}")
    typer.echo(f"Total Questions: {len(DEERE_EVAL_QUESTIONS)}")
    typer.echo("-----------------------------------------------")

    results = []
    start_time_all = time.time()

    for i, eval_item in enumerate(DEERE_EVAL_QUESTIONS):
        question = eval_item["question"]
        expected_answer = eval_item["expected_answer"]
        q_num = i + 1
        typer.echo(f"\n--- Question {q_num}/{len(DEERE_EVAL_QUESTIONS)} --- ")
        typer.echo(f"Q: {question}")
        typer.echo(f"EXPECTED: {expected_answer}") # Print expected answer
        
        start_time_q = time.time()
        generated_answer = "Error during processing."
        try:
            # 1. Retrieve Chunks
            retrieved_chunks_raw = query_rag(
                query_text=question,
                top_k=top_k,
                use_reranker=rerank,
                source_filter=source
            )

            if not retrieved_chunks_raw:
                logger.warning(f"[Q{q_num}] No relevant chunks found.")
                generated_answer = "No relevant context found in the specified source."
            else:
                # 2. Generate Answer using LLM
                context_for_llm = [chunk['text'] for chunk in retrieved_chunks_raw]
                generated_answer = get_llm_answer(
                    question=question,
                    context_chunks=context_for_llm,
                    prompt_name=prompt_name
                )
        except Exception as e:
            logger.exception(f"[Q{q_num}] Error processing question: {question}")
            generated_answer = f"Error during processing: {e}"
        
        end_time_q = time.time()
        typer.echo(f"GENERATED: {generated_answer}") # Print generated answer
        typer.echo(f"(Time: {end_time_q - start_time_q:.2f}s)")
        results.append({"question": question, "expected": expected_answer, "generated": generated_answer})

    end_time_all = time.time()
    typer.echo("\n--- Evaluation Complete --- ")
    typer.echo(f"Total time: {end_time_all - start_time_all:.2f} seconds")
    # Optionally: Save results
    # import json
    # with open(f"deere_eval_results_{source.replace(':', '_').replace('/', '_')}.json", "w") as f:
    #     json.dump(results, f, indent=2)
    # typer.echo(f"Results saved to deere_eval_results_{source.replace(':', '_').replace('/', '_')}.json")

@app.command()
def query(
    query_text: Annotated[str, typer.Argument(help="The query text to search for.")],
    top_k: Annotated[int, typer.Option("-k", help="Number of top results to retrieve.")] = 3,
    rerank: Annotated[bool, typer.Option("--rerank/--no-rerank", help="Enable/disable the reranker model.")] = False
):
    """Queries the RAG pipeline and prints the results."""
    try:
        from src.query import query_rag
    except ImportError as e:
        logger.error(f"Failed to import query modules: {e}. Is the src directory in your PYTHONPATH?")
        raise typer.Exit(code=1)

    typer.echo(f"Querying with text: '{query_text}' (top_k={top_k}, rerank={rerank})")
    try:
        results = query_rag(query_text, top_k=top_k, use_reranker=rerank)
        if results:
            typer.echo("Found results:")
            # Use pprint for readable dictionary output
            # Try to get terminal width, default to 100 if unavailable
            try:
                width = typer.get_terminal_size().columns - 5
            except OSError:
                width = 95 # Default width if terminal size cannot be determined
            pprint.pprint(results, indent=2, width=width)
        else:
            typer.echo("No relevant results found.")
    except Exception as e:
        logger.exception(f"Query failed: {e}")
        raise typer.Exit(code=1)

@app.command(name="inspect-db")
def inspect_database(
    show_chunks: Annotated[bool, typer.Option("--show-chunks", help="Also display the text of associated chunks.")] = False,
    limit: Annotated[int, typer.Option("-l", "--limit", help="Limit the number of documents shown.")] = -1
):
    """Inspects the contents of the SQLite database (Documents and optionally Chunks)."""
    try:
        from src.database.core import get_db
        from src.database import crud as db_crud
    except ImportError as e:
        logger.error(f"Failed to import database modules: {e}. Is the src directory in your PYTHONPATH?")
        raise typer.Exit(code=1)

    typer.echo("--- Inspecting Database Contents ---")
    doc_count = 0
    try:
        with get_db() as db:
            typer.echo("Fetching documents...")
            documents = db_crud.get_all_documents(db)

            if not documents:
                typer.echo("No documents found in the database.")
                return

            typer.echo(f"Found {len(documents)} documents.")
            if limit > 0:
                typer.echo(f"Limiting display to {limit} documents.")
                documents = documents[:limit]

            for doc in documents:
                doc_count += 1
                typer.echo(f"\n--- Document {doc_count} ---")
                typer.echo(f"  ID: {doc.doc_id}")
                typer.echo(f"  Name: {doc.doc_name}")
                typer.echo(f"  Source: {doc.source}")
                typer.echo(f"  Created At: {doc.created_at}")

                if show_chunks and doc.doc_id is not None:
                    typer.echo("  Fetching chunks...")
                    chunks = db_crud.get_chunks_by_doc_id(db, doc.doc_id)
                    if chunks:
                        typer.echo(f"  Found {len(chunks)} chunks:")
                        for i, chunk in enumerate(chunks):
                            typer.echo(f"    --- Chunk {i} (ID: {chunk.chunk_id}) ---")
                            typer.echo(f"      Number: {chunk.chunk_number}")
                            typer.echo(f"      Created At: {chunk.created_at}")
                            text_preview = chunk.chunk_text[:150].replace("\n", " ") + ("..." if len(chunk.chunk_text) > 150 else "")
                            typer.echo(f"      Text Preview: {text_preview}")
                    else:
                        typer.echo("  No chunks found for this document.")

    except Exception as e:
        logger.exception(f"Failed to inspect database: {e}")
        raise typer.Exit(code=1)

    typer.echo("\n--- Database Inspection Complete ---")

# --- Main execution --- #

if __name__ == "__main__":
    # Ensure src is in the Python path if running as a script
    # This is often needed when running `python src/cli.py` directly
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        logger.debug(f"Added {project_root} to sys.path")

    app() 