from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
import time
from typing import TYPE_CHECKING

from src.config import settings
from src.vector_store.client import get_qdrant_client

# Conditional import for type hinting to avoid circular dependency at runtime
if TYPE_CHECKING:
    from src.embeddings import get_embedding_dimension

COLLECTION_NAME = settings.qdrant.collection_name

def create_collection(client: QdrantClient, embedding_dim: int):
    """Creates the Qdrant collection if it doesn't exist."""
    try:
        client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    except (UnexpectedResponse, ValueError) as e:
        # Handle cases where the collection doesn't exist
        # Check the error message specifics if needed, as behavior might vary slightly
        print(f"Collection '{COLLECTION_NAME}' not found or error checking existence: {e}. Attempting creation...")
        try:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=embedding_dim,
                    distance=models.Distance.COSINE # Or use models.Distance.DOT based on model
                )
                # You can add hnsw_config, optimizers_config etc. here if needed
                # hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
                # optimizers_config=models.OptimizersConfigDiff(indexing_threshold=20000),
            )
            print(f"Collection '{COLLECTION_NAME}' created successfully.")
        except Exception as creation_error:
            print(f"Failed to create collection '{COLLECTION_NAME}': {creation_error}")
            raise # Re-raise the creation error
    except Exception as e:
        # Catch any other unexpected errors during get_collection
        print(f"An unexpected error occurred while checking collection '{COLLECTION_NAME}': {e}")
        raise

def recreate_collection(client: QdrantClient, embedding_dim: int):
    """Deletes and recreates the Qdrant collection."""
    try:
        print(f"Attempting to delete collection '{COLLECTION_NAME}' if it exists...")
        client.delete_collection(collection_name=COLLECTION_NAME, timeout=60) # Add timeout
        # Short pause to allow potential propagation
        time.sleep(1)
        print(f"Collection '{COLLECTION_NAME}' deleted (if it existed).")
    except (UnexpectedResponse, ValueError) as e:
        print(f"Collection '{COLLECTION_NAME}' likely did not exist or error during deletion: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during deletion: {e}")
        # Decide if you want to proceed or raise depending on severity

    print(f"Recreating collection '{COLLECTION_NAME}'..." )
    try:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=embedding_dim,
                distance=models.Distance.COSINE
            ),
            timeout=60 # Add timeout
        )
        print(f"Collection '{COLLECTION_NAME}' recreated successfully.")
    except Exception as e:
        print(f"Failed to recreate collection '{COLLECTION_NAME}': {e}")
        raise

# Example usage
if __name__ == "__main__":
    # Moved import here to ensure embeddings module is loaded when needed
    from src.embeddings import get_embedding_dimension

    print(f"--- Testing Collection Management for '{COLLECTION_NAME}' ---")
    try:
        qdrant_client = get_qdrant_client()
        dim = get_embedding_dimension()
        print(f"Using embedding dimension: {dim}")

        print("\n1. Attempting to create collection (if needed)...")
        create_collection(qdrant_client, embedding_dim=dim)

        # Add a dummy point to test
        try:
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=[models.PointStruct(id=1, vector=[0.0]*dim, payload={"test": "dummy"})],
                wait=True
            )
            print(f"Dummy point added to '{COLLECTION_NAME}' for verification.")
        except Exception as upsert_err:
            print(f"Failed to add dummy point: {upsert_err}")

        print("\n2. Attempting to recreate collection...")
        recreate_collection(qdrant_client, embedding_dim=dim)

        # Verify collection exists after recreation
        collection_info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Collection info after recreate: {collection_info}")

        print("\n--- Collection Management Test Complete ---")

    except Exception as e:
        print(f"An error occurred during the collection management test: {e}")
        print(f"Please ensure Qdrant is running and accessible at {settings.qdrant.url}") 