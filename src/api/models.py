from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# --- Request Models ---

class QueryRequest(BaseModel):
    query_text: str
    top_k: int = Field(default=3, gt=0, le=20) # Add validation: >0, <=20
    use_reranker: bool = False
    source_filter: Optional[str] = None
    prompt_name: Optional[str] = None # If None, llm_interface will use default

# --- Response Models ---

class ChunkDetail(BaseModel):
    """Model representing details of a retrieved chunk for the API response."""
    text: str
    score: float
    source: Optional[str] = None
    sql_chunk_id: Optional[int] = None
    # qdrant_payload: Optional[Dict[str, Any]] = None # Optionally include full payload

class QueryResponse(BaseModel):
    """Model for the response to a query request."""
    answer: str
    retrieved_chunks: List[ChunkDetail]
    # Add optional fields for debugging/metadata if needed
    # query_used: Optional[str] = None
    # prompt_used: Optional[str] = None

class SourcesResponse(BaseModel):
    """Model for the list of available sources."""
    sources: List[str]

class PromptsResponse(BaseModel):
    """Model for the list of available prompt names."""
    prompt_names: List[str] 