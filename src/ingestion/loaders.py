from pathlib import Path
from typing import Optional, List
import logging
import json

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = [".txt"] # Initially support only text files

def load_text_from_file(file_path: Path) -> Optional[str]:
    """Loads text content from a supported file type."""
    if not file_path.is_file():
        logger.error(f"File not found: {file_path}")
        return None

    file_ext = file_path.suffix.lower()

    if file_ext == ".txt":
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            return None
    # elif file_ext == ".pdf":
    #     # Placeholder for PDF loading (requires pypdf)
    #     try:
    #         from pypdf import PdfReader
    #         reader = PdfReader(file_path)
    #         text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
    #         return text
    #     except ImportError:
    #         logger.error(f"pypdf not installed. Cannot load PDF file: {file_path}")
    #         logger.error("Install with: uv add pypdf")
    #         return None
    #     except Exception as e:
    #         logger.error(f"Error reading PDF file {file_path}: {e}")
    #         return None
    else:
        logger.warning(f"Unsupported file type: {file_ext}. Skipping {file_path}")
        return None

def load_chunks_from_json(file_path: Path) -> Optional[List[str]]:
    """Loads a list of text chunks from a JSON file.
    
    Assumes the JSON file contains a single list of objects, where 
    each object has a 'content' key containing the chunk text.
    Example: [{"filename": "...", "content": "Chunk 1 text..."}, ...]
    """
    if not file_path.exists():
        logger.error(f"JSON chunk file not found: {file_path}")
        return None
    if not file_path.is_file():
        logger.error(f"Path provided is not a file: {file_path}")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            logger.error(f"JSON file does not contain a list: {file_path}")
            return None
            
        # Extract the 'content' from each object in the list
        extracted_chunks = []
        for i, item in enumerate(data):
            if isinstance(item, dict) and "content" in item and isinstance(item["content"], str):
                extracted_chunks.append(item["content"])
            else:
                logger.warning(f"Skipping item {i} in JSON list: Expected a dictionary with a string 'content' key. Found: {type(item)}")
                # Decide if this should be an error or just a warning. Warning for now.

        if not extracted_chunks:
            logger.error(f"No valid chunks with 'content' found in {file_path}")
            return None

        logger.info(f"Successfully extracted {len(extracted_chunks)} text chunks from {file_path}")
        return extracted_chunks
        
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading JSON chunk file {file_path}: {e}")
        return None

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Create dummy files for testing
    dummy_dir = Path("./dummy_loader_test")
    dummy_dir.mkdir(exist_ok=True)
    dummy_txt = dummy_dir / "test.txt"
    dummy_unsupported = dummy_dir / "test.csv"

    dummy_txt.write_text("This is the content of the test text file.\nIt has two lines.", encoding='utf-8')
    dummy_unsupported.write_text("col1,col2\na,b", encoding='utf-8')

    print("--- Testing File Loader ---")

    # Test loading txt
    print(f"Loading: {dummy_txt}")
    content_txt = load_text_from_file(dummy_txt)
    if content_txt:
        print(f"Successfully loaded {dummy_txt}:\n---\n{content_txt}\n---")
    else:
        print(f"Failed to load {dummy_txt}")

    # Test loading unsupported
    print(f"\nLoading: {dummy_unsupported}")
    content_unsupported = load_text_from_file(dummy_unsupported)
    if content_unsupported:
        print(f"Incorrectly loaded unsupported file: {dummy_unsupported}")
    else:
        print(f"Correctly skipped unsupported file: {dummy_unsupported}")

    # Test loading non-existent
    non_existent_file = dummy_dir / "non_existent.txt"
    print(f"\nLoading: {non_existent_file}")
    content_non_existent = load_text_from_file(non_existent_file)
    if content_non_existent:
        print(f"Incorrectly loaded non-existent file: {non_existent_file}")
    else:
        print(f"Correctly handled non-existent file: {non_existent_file}")

    print("\n--- File Loader Test Complete ---")

    # Clean up dummy files
    import shutil
    shutil.rmtree(dummy_dir)
    print("Cleaned up dummy directory.") 