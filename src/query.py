import logging
from typing import List, Dict, Any, Optional
import operator # For sorting

from qdrant_client import models # Import Qdrant models for filtering

from src.config import settings
from src.embeddings import generate_embedding
from src.vector_store.client import get_qdrant_client
from src.vector_store.searching import search_similar_chunks
from src.database.core import get_db
from src.database import crud as db_crud
# Import reranker functions
from src.reranker import compute_rerank_scores, load_reranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for reranking
RERANK_TOP_N = 3 # How many results to return after reranking
INITIAL_SEARCH_MULTIPLIER = 5 # Fetch this * RERANK_TOP_N initial candidates
INITIAL_SEARCH_K = RERANK_TOP_N * INITIAL_SEARCH_MULTIPLIER

def query_rag(
    query_text: str, 
    top_k: int = 3, # This now refers to the FINAL number of results after potential reranking
    use_reranker: bool = False,
    source_filter: Optional[str] = None # Re-add optional source filter
) -> List[Dict[str, Any]]:
    """Performs RAG query: embed query, search Qdrant, optionally rerank, retrieve from SQLite."""
    logger.info(f"Received query: '{query_text}' (top_k={top_k}, use_reranker={use_reranker}, source_filter='{source_filter}')")

    # 1. Generate Query Embedding
    logger.debug("Generating embedding for query...")
    try:
        query_embedding = generate_embedding(query_text).tolist()
    except Exception as e:
        logger.exception("Failed to generate query embedding.")
        return []

    # 2. Initial Search in Qdrant
    # Fetch more results if reranking is enabled
    initial_k = INITIAL_SEARCH_K if use_reranker else top_k
    logger.debug(f"Searching Qdrant for top {initial_k} initial candidates...")
    qdrant_client = get_qdrant_client()

    # Construct search filter if source_filter is provided
    search_filter = None
    if source_filter:
        logger.info(f"Applying source filter: {source_filter}")
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="source", # We need to re-add source to payload during ingestion!
                    match=models.MatchValue(value=source_filter)
                )
            ]
        )
        
    search_results = search_similar_chunks(
        client=qdrant_client,
        query_vector=query_embedding,
        top_k=initial_k,
        search_filter=search_filter # Pass the filter to the search function
    )

    if not search_results:
        logger.warning("No similar chunks found in Qdrant for the query.")
        return []
    logger.info(f"Retrieved {len(search_results)} initial candidates from Qdrant.")

    # 3. Extract Chunk IDs and Retrieve Full Text from SQLite
    qdrant_point_ids = [hit.id for hit in search_results]
    sql_chunk_ids = [pid for pid in qdrant_point_ids if isinstance(pid, int)]
    if len(sql_chunk_ids) != len(qdrant_point_ids):
        logger.warning(f"Some Qdrant point IDs were not integers: {qdrant_point_ids}")
    if not sql_chunk_ids:
        logger.warning("No valid integer chunk IDs extracted from Qdrant search results.")
        return []

    logger.debug(f"Retrieving full text from database for {len(sql_chunk_ids)} IDs...")
    initial_candidates = []
    try:
        with get_db() as db_session:
            full_chunks = db_crud.get_chunks_by_ids(db_session, sql_chunk_ids)
            retrieved_chunks_map = {chunk.chunk_id: chunk for chunk in full_chunks if chunk.chunk_id is not None}
            logger.debug(f"Retrieved {len(retrieved_chunks_map)} chunks from DB.")

            # Build initial candidate list with full data needed for reranking or final output
            for hit in search_results:
                if isinstance(hit.id, int) and hit.id in retrieved_chunks_map:
                    db_chunk = retrieved_chunks_map[hit.id]
                    candidate = {
                        "initial_score": hit.score, # Keep the original score
                        "rerank_score": None, # Placeholder for rerank score
                        "qdrant_id": hit.id,
                        "sql_chunk_id": db_chunk.chunk_id,
                        "doc_id": db_chunk.doc_id,
                        "chunk_number": db_chunk.chunk_number,
                        "text": db_chunk.chunk_text,
                        "qdrant_payload": hit.payload
                    }
                    initial_candidates.append(candidate)
                else:
                    logger.warning(f"Could not find matching chunk in DB for Qdrant hit ID: {hit.id}")

    except Exception as e:
        logger.exception("Failed to retrieve chunks from database.")
        return [] # Stop if DB retrieval fails
        
    if not initial_candidates:
         logger.warning("Could not assemble any candidates with full text from DB.")
         return []

    # 4. Apply Reranker (if enabled)
    if use_reranker:
        logger.info(f"Applying reranker to {len(initial_candidates)} candidates...")
        candidate_texts = [candidate['text'] for candidate in initial_candidates]
        try:
            # Ensure reranker model is loaded (important if it wasn't used before)
            load_reranker() 
            rerank_scores = compute_rerank_scores(query_text, candidate_texts)
            
            if len(rerank_scores) == len(initial_candidates):
                for candidate, rerank_score in zip(initial_candidates, rerank_scores):
                    candidate["rerank_score"] = rerank_score
                
                # Sort candidates by the new rerank_score (descending)
                initial_candidates.sort(key=operator.itemgetter("rerank_score"), reverse=True)
                logger.info(f"Reranking complete. Top initial score: {initial_candidates[0]['initial_score'] if initial_candidates else 'N/A'}, Top rerank score: {initial_candidates[0]['rerank_score'] if initial_candidates else 'N/A'}")
            else:
                 logger.error(f"Mismatch between number of candidates ({len(initial_candidates)}) and rerank scores ({len(rerank_scores)}). Skipping reranking.")

        except Exception as e:
            logger.exception("Failed during the reranking process. Proceeding without reranking.")
            # Optionally clear rerank scores if partially applied
            for candidate in initial_candidates:
                 candidate["rerank_score"] = None
                 
    else:
         # If not reranking, sort by initial Qdrant score (descending)
         initial_candidates.sort(key=operator.itemgetter("initial_score"), reverse=True)

    # 5. Select Top K Results and Return
    final_results = initial_candidates[:top_k]
    logger.info(f"Returning final top {len(final_results)} results.")
    return final_results

# Example Usage (remains the same structure, tests reranking implicitly if called)
if __name__ == "__main__":
    import pprint

    logger.info("--- Testing Query Pipeline with Gatsby Queries ---")

    test_queries = [
        "Find passages where Gatsby expresses his love for Daisy",
        "Discussions about social class, old money vs new money",
        # "Find passages leading up to Gatsby's death",
        # "Tom Buchanan talking about power, wealth, and control",
        # "non_existent_topic_for_gatsby_test_xyz"
    ]

    for query in test_queries:
        print(f"\n--- Querying (NO RERANKER) for: '{query}' ---")
        try:
            results_no_rerank = query_rag(query, top_k=RERANK_TOP_N, use_reranker=False)
            if results_no_rerank:
                print("Results (No Reranker):")
                pprint.pprint([ {k: v for k, v in res.items() if k != 'text'} for res in results_no_rerank], indent=2, width=120) # Print without text for brevity
            else:
                print("No results found (No Reranker).")
        except Exception as e:
            logger.exception(f"An error occurred during non-reranked query: {query}")

        print(f"\n--- Querying (WITH RERANKER) for: '{query}' ---")
        try:
            results_with_rerank = query_rag(query, top_k=RERANK_TOP_N, use_reranker=True)
            if results_with_rerank:
                print("Results (Reranked):")
                pprint.pprint([ {k: v for k, v in res.items() if k != 'text'} for res in results_with_rerank], indent=2, width=120) # Print without text for brevity
            else:
                print("No results found (Reranked).")
        except Exception as e:
            logger.exception(f"An error occurred during reranked query: {query}")

    print("\n--- Query Pipeline Reranker Test Complete ---") 