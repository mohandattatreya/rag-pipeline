import sys
import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Depends

# --- Add project root to PYTHONPATH ---
# Ensures modules in 'src' can be imported
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Added {project_root} to sys.path")
# --- End PYTHONPATH ---

# Import local modules AFTER potentially modifying sys.path
from src.config import settings
from src.query import query_rag
from src.llm_interface import get_llm_answer
from src.database.core import get_db
from src.database.crud import get_distinct_sources
from src.api.models import ( # Import API models
    QueryRequest,
    QueryResponse,
    ChunkDetail,
    SourcesResponse,
    PromptsResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app instance
app = FastAPI(
    title="RAG Pipeline API",
    description="API endpoints for the RAG pipeline backend.",
    version="0.1.0"
)

# --- API Endpoints ---

@app.get("/sources/", response_model=SourcesResponse)
async def get_sources():
    """Returns a list of distinct source descriptions ingested into the database."""
    try:
        with get_db() as db:
            sources = get_distinct_sources(db)
        return SourcesResponse(sources=sources)
    except Exception as e:
        logger.exception("Error fetching sources from database.")
        raise HTTPException(status_code=500, detail="Failed to retrieve sources from database.")

@app.get("/prompts/", response_model=PromptsResponse)
async def get_prompts():
    """Returns a list of available prompt template names from the configuration."""
    try:
        prompt_names = list(settings.prompts.keys())
        return PromptsResponse(prompt_names=prompt_names)
    except Exception as e:
        logger.exception("Error reading prompt names from configuration.")
        raise HTTPException(status_code=500, detail="Failed to retrieve prompt names from configuration.")

@app.post("/query/", response_model=QueryResponse)
async def execute_query(request: QueryRequest):
    """Executes the RAG query pipeline based on the provided request."""
    logger.info(f"Received query request: {request.query_text[:100]}... (prompt: {request.prompt_name})")

    # 1. Retrieve Chunks
    try:
        retrieved_chunks_raw = query_rag(
            query_text=request.query_text,
            top_k=request.top_k,
            use_reranker=request.use_reranker,
            source_filter=request.source_filter
        )
    except Exception as e:
        logger.exception("Error during query_rag execution in API.")
        raise HTTPException(status_code=500, detail=f"Error retrieving chunks: {e}")

    if not retrieved_chunks_raw:
        logger.warning(f"No relevant chunks found for query: {request.query_text[:100]}...")
        # Return empty answer and empty chunks list
        return QueryResponse(answer="Could not find relevant context for this query.", retrieved_chunks=[])

    # 2. Generate Answer using LLM
    context_for_llm = [chunk['text'] for chunk in retrieved_chunks_raw]
    try:
        final_answer = get_llm_answer(
            question=request.query_text,
            context_chunks=context_for_llm,
            prompt_name=request.prompt_name # Pass the prompt name
        )
    except Exception as e:
        logger.exception("Error during get_llm_answer execution in API.")
        raise HTTPException(status_code=500, detail=f"Error generating answer with LLM: {e}")

    # 3. Format Response
    response_chunks = []
    for chunk in retrieved_chunks_raw:
        score = chunk.get('rerank_score', chunk.get('initial_score'))
        # Ensure score is float for Pydantic model, handle None
        score_float = float(score) if isinstance(score, (int, float)) else 0.0 
        
        response_chunks.append(ChunkDetail(
            text=chunk.get('text', 'N/A'),
            score=score_float,
            source=chunk.get('qdrant_payload', {}).get('source'),
            sql_chunk_id=chunk.get('sql_chunk_id')
        ))

    return QueryResponse(
        answer=final_answer,
        retrieved_chunks=response_chunks
    )

# --- Main execution (for running with uvicorn directly) ---
if __name__ == "__main__":
    import uvicorn
    # Get host/port from config or use defaults
    # For simplicity, using defaults here. Could read from settings.
    host = "127.0.0.1"
    port = 8000
    print(f"Starting Uvicorn server on {host}:{port}")
    uvicorn.run("src.api.main:app", host=host, port=port, reload=True) 