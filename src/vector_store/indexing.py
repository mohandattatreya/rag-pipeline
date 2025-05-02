from qdrant_client import QdrantClient, models
from typing import List, Dict, Any
import logging
import time # Added time import for sleep in example

from src.config import settings
from src.vector_store.client import get_qdrant_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLLECTION_NAME = settings.qdrant.collection_name

def upsert_chunk_embedding(client: QdrantClient, point_id: int, vector: List[float], payload: Dict[str, Any]):
    """Upserts a single chunk embedding and its payload into Qdrant."""
    try:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[models.PointStruct(id=point_id, vector=vector, payload=payload)],
            wait=False # Set to True if immediate consistency is critical
        )
        # logger.debug(f"Upserted point ID: {point_id} into '{COLLECTION_NAME}'")
    except Exception as e:
        logger.error(f"Failed to upsert point ID {point_id} into '{COLLECTION_NAME}': {e}")
        # Depending on the desired behavior, you might want to raise the exception
        # raise

def upsert_batch_chunk_embeddings(
    client: QdrantClient,
    point_ids: List[int],
    vectors: List[List[float]],
    payloads: List[Dict[str, Any]],
    batch_size: int = 100 # Adjust batch size based on payload size and network
):
    """Upserts chunk embeddings and payloads in batches."""
    if not (len(point_ids) == len(vectors) == len(payloads)):
        raise ValueError("Input lists (point_ids, vectors, payloads) must have the same length.")

    num_points = len(point_ids)
    logger.info(f"Starting batch upsert of {num_points} points to '{COLLECTION_NAME}'...")

    for i in range(0, num_points, batch_size):
        batch_ids = point_ids[i : i + batch_size]
        batch_vectors = vectors[i : i + batch_size]
        batch_payloads = payloads[i : i + batch_size]

        points_batch = [
            models.PointStruct(id=pid, vector=vec, payload=pld)
            for pid, vec, pld in zip(batch_ids, batch_vectors, batch_payloads)
        ]

        try:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points_batch,
                wait=False # Generally False for batch upserts for performance
            )
            logger.info(f"Upserted batch {i // batch_size + 1}/{(num_points + batch_size - 1) // batch_size} (size: {len(points_batch)})")
        except Exception as e:
            logger.error(f"Failed to upsert batch starting at index {i} into '{COLLECTION_NAME}': {e}")
            # Consider adding retry logic or raising the exception

    logger.info(f"Batch upsert completed for {num_points} points.")


# Example usage
if __name__ == "__main__":
    from src.embeddings import get_embedding_dimension, generate_embedding
    from src.vector_store.collections import recreate_collection # Ensure collection exists

    logger.info(f"--- Testing Vector Indexing for '{COLLECTION_NAME}' ---")
    try:
        qdrant_client = get_qdrant_client()
        dim = get_embedding_dimension()

        # Ensure the collection exists and is clean for the test
        logger.info("Recreating collection for test...")
        recreate_collection(qdrant_client, embedding_dim=dim)

        # 1. Test Single Upsert
        logger.info("\nTesting single upsert...")
        payload1 = {"doc_name": "test_doc.txt", "chunk_id_sql": 101}
        vector1 = generate_embedding("This is the first test chunk.").tolist()
        point_id1 = 1 # Using a simple integer ID
        upsert_chunk_embedding(qdrant_client, point_id=point_id1, vector=vector1, payload=payload1)
        logger.info(f"Single point upsert initiated for ID: {point_id1}")
        # Verification (optional, requires wait=True or a delay)
        # time.sleep(1)
        # retrieved = qdrant_client.retrieve(collection_name=COLLECTION_NAME, ids=[point_id1])
        # logger.info(f"Retrieved point 1: {retrieved}")

        # 2. Test Batch Upsert
        logger.info("\nTesting batch upsert...")
        point_ids_batch = [2, 3, 4]
        texts_batch = ["Second chunk here.", "Third piece of text.", "Final element for batch."]
        vectors_batch = [generate_embedding(t).tolist() for t in texts_batch]
        payloads_batch = [
            {"doc_name": "test_doc.txt", "chunk_id_sql": 102},
            {"doc_name": "other_doc.pdf", "chunk_id_sql": 201},
            {"doc_name": "test_doc.txt", "chunk_id_sql": 103},
        ]
        upsert_batch_chunk_embeddings(qdrant_client, point_ids_batch, vectors_batch, payloads_batch, batch_size=2)

        # Allow some time for async upserts if wait=False
        time.sleep(1)
        count = qdrant_client.count(collection_name=COLLECTION_NAME, exact=True)
        logger.info(f"Total points in collection after upserts: {count.count}")

        logger.info("\n--- Vector Indexing Test Complete ---")

    except Exception as e:
        logger.exception(f"An error occurred during the vector indexing test: {e}")
        logger.error(f"Please ensure Qdrant is running and accessible at {settings.qdrant.url}") 