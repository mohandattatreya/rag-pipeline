import os
from pathlib import Path
from pydantic import BaseModel, Field, HttpUrl, FilePath, DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
from typing import Optional, Literal, Dict

# Define the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Pydantic Models for Configuration Sections ---

class DatabaseConfig(BaseModel):
    db_path: str = "rag_pipeline.db"

    @property
    def sqlalchemy_database_url(self) -> str:
        # Ensure the path is absolute for SQLAlchemy
        absolute_db_path = BASE_DIR / self.db_path
        # Create the directory if it doesn't exist
        absolute_db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{(absolute_db_path)}"

class QdrantConfig(BaseModel):
    url: HttpUrl = "http://localhost:6333"
    api_key: Optional[str] = None
    mount_dir: str = "./qdrant_storage" # Used by start_qdrant.sh, not directly by client typically
    collection_name: str = "text_chunks"
    prefer_grpc: bool = False

class EmbeddingConfig(BaseModel):
    model_name: str = "sentence-transformers/all-mpnet-base-v2"

class ChunkingConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 50

class LLMConfig(BaseModel):
    provider: Literal["ollama"] = "ollama"
    model_name: str = "llama3:latest"
    api_base_url: HttpUrl = "http://localhost:11434"
    cannot_answer_phrase: str = "Based on the provided text, I cannot answer this question."

class PromptConfig(BaseModel):
    template: str
    # Potentially add description field later
    # description: Optional[str] = None 

# New model for JSON directory ingestion settings
class JsonIngestionConfig(BaseModel):
    input_directory: DirectoryPath # Use DirectoryPath for validation
    file_limit: Optional[int] = None # Default to None (no limit)

class AppSettings(BaseSettings):
    database: DatabaseConfig
    qdrant: QdrantConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    llm: LLMConfig
    # Change prompt to a dictionary of PromptConfig objects
    prompts: Dict[str, PromptConfig] = {
        "default": PromptConfig(template="You are a helpful AI assistant. Answer based on the context.\nContext: {context}\nQuestion: {question}\nAnswer:")
    }
    # Add optional JSON ingestion config
    json_ingestion: Optional[JsonIngestionConfig] = None

    # Configure Pydantic Settings
    model_config = SettingsConfigDict(
        env_file=".env",           # Load .env file if present
        env_nested_delimiter="__", # Allow nested env vars like QDRANT__API_KEY
        extra="ignore"             # Ignore extra fields from env/files
    )

# --- Loading Function ---

def load_config(config_path: Path = BASE_DIR / "config.yaml") -> AppSettings:
    """Loads configuration from YAML file and merges with environment variables."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    # Load base settings from YAML, allowing environment variables to override
    settings = AppSettings(**config_data)

    # Specifically check for QDRANT_API_KEY environment variable
    qdrant_api_key_env = os.getenv("QDRANT_API_KEY")
    if qdrant_api_key_env:
        settings.qdrant.api_key = qdrant_api_key_env

    return settings

# --- Global Settings Instance ---
# Load settings once when the module is imported
settings: AppSettings = load_config()

# Example usage (can be removed later):
if __name__ == "__main__":
    print(f"Loaded configuration:")
    print(f"  Database URL: {settings.database.sqlalchemy_database_url}")
    print(f"  Qdrant URL: {settings.qdrant.url}")
    print(f"  Qdrant Collection: {settings.qdrant.collection_name}")
    print(f"  Embedding Model: {settings.embedding.model_name}")
    print(f"  Chunk Size: {settings.chunking.chunk_size}")
    print(f"  LLM Provider: {settings.llm.provider}")
    print(f"  LLM Model: {settings.llm.model_name}")
    print(f"  LLM API Base: {settings.llm.api_base_url}")
    # print(f"  LLM Cannot Answer: '{settings.llm.cannot_answer_phrase}'") # Less relevant now
    # Print available prompt names
    print(f"  Available Prompts: {list(settings.prompts.keys())}")
    # Print default prompt template for reference
    if "default" in settings.prompts:
        print(f"    Default Prompt Template: \n{settings.prompts['default'].template}")
    # Print JSON ingestion config if present
    if settings.json_ingestion:
        print(f"  JSON Ingestion Dir: {settings.json_ingestion.input_directory}")
        print(f"  JSON Ingestion Limit: {settings.json_ingestion.file_limit}")
    else:
        print("  JSON Ingestion config not provided.")

    # Test accessing a nested setting via env var (requires setting it first)
    # Example: export QDRANT__API_KEY="your_key_here"
    # print(f"  Qdrant API Key (from env?): {settings.qdrant.api_key}") 