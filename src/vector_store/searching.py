from qdrant_client import QdrantClient, models
from typing import List, Tuple
import logging

from src.config import settings
from src.vector_store.client import get_qdrant_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLLECTION_NAME = settings.qdrant.collection_name

def search_similar_chunks(
    client: QdrantClient,
    query_vector: List[float],
    top_k: int = 5,
    score_threshold: float | None = None,
    search_filter: models.Filter | None = None
) -> List[models.ScoredPoint]:
    """Searches for similar chunk embeddings in Qdrant."""
    try:
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False
        )
        logger.info(f"Search returned {len(search_result)} results (top_k={top_k}, threshold={score_threshold}, filter_applied={search_filter is not None}).")
        return search_result
    except Exception as e:
        logger.error(f"Error during Qdrant search in '{COLLECTION_NAME}': {e}")
        return [] # Return empty list on error

# Example usage
if __name__ == "__main__":
    import time
    from src.embeddings import get_embedding_dimension, generate_embedding
    from src.vector_store.collections import recreate_collection
    from src.vector_store.indexing import upsert_batch_chunk_embeddings

    logger.info(f"--- Testing Vector Searching for '{COLLECTION_NAME}' ---")
    try:
        qdrant_client = get_qdrant_client()
        dim = get_embedding_dimension()

        # 1. Setup: Ensure collection exists and has data
        logger.info("Recreating collection and adding test data...")
        recreate_collection(qdrant_client, embedding_dim=dim)
        point_ids_batch = [1, 2, 3, 4, 5]
        texts_batch = [
            "The quick brown fox jumps over the lazy dog.",
            "Exploring the capabilities of vector databases.",
            "Similarity search is crucial for RAG pipelines.",
            "Qdrant provides efficient vector storage and retrieval.",
            "A lazy dog sat under the shady tree."
        ]
        vectors_batch = [generate_embedding(t).tolist() for t in texts_batch]
        payloads_batch = [
            {"doc_name": "doc1.txt", "chunk_id_sql": 101, "text_preview": texts_batch[0][:30]},
            {"doc_name": "doc2.pdf", "chunk_id_sql": 201, "text_preview": texts_batch[1][:30]},
            {"doc_name": "doc1.txt", "chunk_id_sql": 102, "text_preview": texts_batch[2][:30]},
            {"doc_name": "doc3.md", "chunk_id_sql": 301, "text_preview": texts_batch[3][:30]},
            {"doc_name": "doc1.txt", "chunk_id_sql": 103, "text_preview": texts_batch[4][:30]},
        ]
        upsert_batch_chunk_embeddings(qdrant_client, point_ids_batch, vectors_batch, payloads_batch)
        time.sleep(1) # Allow time for indexing

        # 2. Perform Search
        logger.info("\nPerforming search...")
        query_text = "information about sleepy animals"
        query_embedding = generate_embedding(query_text).tolist()

        search_results = search_similar_chunks(qdrant_client, query_embedding, top_k=3)

        logger.info(f"Search results for '{query_text}':")
        if search_results:
            for i, hit in enumerate(search_results):
                logger.info(f"  {i+1}. ID: {hit.id}, Score: {hit.score:.4f}, Payload: {hit.payload}")
        else:
            logger.warning("No search results found.")

        # 3. Test with threshold
        logger.info("\nPerforming search with score threshold...")
        threshold = 0.5 # Adjust based on expected scores for your model/data
        search_results_thresh = search_similar_chunks(
            qdrant_client,
            query_embedding,
            top_k=5,
            score_threshold=threshold
        )
        logger.info(f"Search results with threshold > {threshold}:")
        if search_results_thresh:
            for i, hit in enumerate(search_results_thresh):
                logger.info(f"  {i+1}. ID: {hit.id}, Score: {hit.score:.4f}, Payload: {hit.payload}")
        else:
            logger.warning(f"No search results found above threshold {threshold}.")

        logger.info("\n--- Vector Searching Test Complete ---")

    except Exception as e:
        logger.exception(f"An error occurred during the vector searching test: {e}")
        logger.error(f"Please ensure Qdrant is running and accessible at {settings.qdrant.url}") 