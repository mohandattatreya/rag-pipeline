from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
import time

from src.config import settings

# --- Qdrant Client Initialization ---

_qdrant_client_instance = None

def get_qdrant_client() -> QdrantClient:
    """Initializes and returns a singleton QdrantClient instance based on config."""
    global _qdrant_client_instance
    if _qdrant_client_instance is None:
        print(f"Initializing Qdrant client for URL: {settings.qdrant.url}")
        _qdrant_client_instance = QdrantClient(
            url=str(settings.qdrant.url),
            api_key=settings.qdrant.api_key,
            prefer_grpc=settings.qdrant.prefer_grpc,
            # You can add timeouts here if needed, e.g., timeout=60
        )
        # Optional: Add a simple health check
        try:
            # A simple request to check connectivity
            _qdrant_client_instance.get_collections()
            print("Qdrant client initialized and connection verified.")
        except Exception as e:
            print(f"Warning: Could not verify connection to Qdrant at {settings.qdrant.url}. Error: {e}")
            # Depending on the use case, you might want to raise the exception here

    return _qdrant_client_instance

# --- Example Usage ---

if __name__ == "__main__":
    print("Attempting to get Qdrant client...")
    try:
        client = get_qdrant_client()
        print(f"Qdrant client type: {type(client)}")

        # Example: List collections (requires Qdrant to be running)
        try:
            collections = client.get_collections()
            print(f"Available collections: {collections}")
        except Exception as e:
            print(f"Could not list collections. Is Qdrant running at {settings.qdrant.url}? Error: {e}")

    except Exception as e:
        print(f"Failed to initialize Qdrant client: {e}") 