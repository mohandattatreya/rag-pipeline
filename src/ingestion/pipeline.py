from pathlib import Path
import logging
from typing import List, Optional
import time

from src.config import settings
from src.database.core import get_db
from src.database import crud as db_crud
from src.ingestion.loaders import load_text_from_file, load_chunks_from_json
from src.ingestion.chunking import chunk_text_by_char
from src.embeddings import generate_embeddings_batch, get_embedding_dimension
from src.vector_store.client import get_qdrant_client
from src.vector_store.collections import create_collection
from src.vector_store.indexing import upsert_batch_chunk_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _ingest_text_content(
    doc_name: str, 
    source_description: str, 
    doc_text: Optional[str] = None, 
    precomputed_chunks: Optional[List[str]] = None
):
    """Core logic to process content: store, chunk (if needed), embed, index."""
    if not doc_text and not precomputed_chunks:
        logger.error(f"Ingestion error for '{doc_name}': Must provide either doc_text or precomputed_chunks.")
        return False
    if doc_text and precomputed_chunks:
        logger.warning(f"Ingestion warning for '{doc_name}': Both doc_text and precomputed_chunks provided. Using precomputed_chunks.")
        doc_text = None # Prioritize precomputed chunks
        
    logger.info(f"Starting ingestion for document: '{doc_name}' from {source_description}")

    # Initialize clients and DB session
    qdrant_client = get_qdrant_client()
    embedding_dim = get_embedding_dimension() # Ensures model is loaded
    create_collection(qdrant_client, embedding_dim)

    processed_chunks = 0

    with get_db() as db_session:
        # Add Document Record to DB
        logger.info(f"Adding/getting document record for '{doc_name}' in database...")
        db_document = db_crud.add_document(db_session, doc_name=doc_name, source=source_description)
        if not db_document or not db_document.doc_id:
            logger.error(f"Failed to add or retrieve document record for '{doc_name}' in DB. Aborting.")
            return False # Indicate failure
        doc_id = db_document.doc_id
        logger.info(f"Using Document ID: {doc_id} for '{doc_name}'")

        # Chunk Text or use precomputed
        if precomputed_chunks:
            logger.info("Using precomputed chunks.")
            chunks = precomputed_chunks
        elif doc_text:
            logger.info("Chunking document text...")
            chunks = chunk_text_by_char(
                text=doc_text,
                chunk_size=settings.chunking.chunk_size,
                chunk_overlap=settings.chunking.chunk_overlap
            )
        else:
            # This case should be caught by the initial check, but for safety:
            logger.error(f"Internal error: No text or chunks available for '{doc_name}'. Aborting.")
            return False
            
        if not chunks:
            logger.warning(f"No chunks available (either generated or provided) for '{doc_name}'. Nothing to ingest.")
            return True # Ingestion technically succeeded (nothing to do)
        logger.info(f"Processing {len(chunks)} chunks.")

        # Prepare data for batch processing
        chunk_texts = chunks
        chunk_numbers = list(range(len(chunks)))

        # Add Chunk Records to DB
        logger.info("Adding chunk records to database...")
        db_chunks = []
        successful_chunk_indices = []
        for i, chunk_text in enumerate(chunk_texts):
            chunk_number = chunk_numbers[i]
            db_chunk = db_crud.add_chunk(db_session, doc_id=doc_id, chunk_number=chunk_number, chunk_text=chunk_text)
            if db_chunk:
                db_chunks.append(db_chunk)
                successful_chunk_indices.append(i)
            else:
                logger.warning(f"Failed to add chunk {chunk_number} for doc_id {doc_id} to database. It will be skipped.")

        if not db_chunks:
            logger.error(f"No chunks were successfully added to the database for '{doc_name}'. Aborting embedding/indexing.")
            return False # Indicate failure
        logger.info(f"Successfully added {len(db_chunks)} chunk records to the database.")

        # Filter chunk_texts to only include those successfully added to DB
        successful_chunk_texts = [chunk_texts[i] for i in successful_chunk_indices]
        successful_db_chunk_ids = [c.chunk_id for c in db_chunks if c.chunk_id is not None]

        if len(successful_chunk_texts) != len(successful_db_chunk_ids):
             logger.error(f"Mismatch between successfully processed texts ({len(successful_chunk_texts)}) and DB chunk IDs ({len(successful_db_chunk_ids)}). Aborting indexing.")
             return False # Indicate failure

        # Generate Embeddings (Batch)
        logger.info("Generating embeddings for chunks...")
        embeddings = generate_embeddings_batch(successful_chunk_texts)
        embeddings_list = [emb.tolist() for emb in embeddings]

        # Prepare Payloads and Upsert to Qdrant (Batch)
        logger.info("Preparing payloads and upserting to Qdrant...")
        point_ids = successful_db_chunk_ids
        payloads = [
            {
                "doc_name": doc_name,
                "chunk_id_sql": chunk_id,
                "text_preview": text[:100] + "...",
                "source": source_description
            }
            for chunk_id, text in zip(successful_db_chunk_ids, successful_chunk_texts)
        ]
        upsert_batch_chunk_embeddings(qdrant_client, point_ids, embeddings_list, payloads)
        processed_chunks = len(point_ids)

    logger.info(f"Ingestion process completed for: '{doc_name}'. Processed {processed_chunks} chunks.")
    return True # Indicate success

def ingest_document_from_file(file_path_str: str):
    """Processes a single document file: loads text and calls core ingestion logic."""
    file_path = Path(file_path_str)
    doc_name = file_path.name
    
    # --- Check if document already exists in DB ---
    try:
        with get_db() as db:
            existing_doc = db_crud.get_document_by_name(db, doc_name)
            if existing_doc:
                logger.warning(f"Document '{doc_name}' already exists in the database (ID: {existing_doc.doc_id}). Skipping ingestion.")
                return # Skip processing this file
    except Exception as e:
        # Log error but proceed? Or halt? Let's log and halt for safety.
        logger.exception(f"Error checking for existing document '{doc_name}' in database. Halting ingestion for this file.")
        return
    # --- End check ---

    source_description = f"file:{file_path_str}"
    logger.info(f"Attempting to load text from file: {file_path}")

    doc_text = load_text_from_file(file_path)
    if not doc_text:
        logger.error(f"Failed to load text from {file_path}. Skipping ingestion.")
        return

    # Call core logic with the loaded text
    _ingest_text_content(doc_name=doc_name, source_description=source_description, doc_text=doc_text)

def ingest_json_files_from_directory(input_dir: Path, file_limit: Optional[int] = None):
    """Finds JSON files in a directory, loads chunks, and ingests them."""
    logger.info(f"Starting ingestion from directory: {input_dir}")
    if not input_dir.is_dir():
        logger.error(f"Input directory not found or is not a directory: {input_dir.resolve()}")
        return

    json_files = sorted(list(input_dir.glob("*.json"))) # Get a sorted list for consistency
    if not json_files:
        logger.warning(f"No *.json files found in directory: {input_dir.resolve()}")
        return

    logger.info(f"Found {len(json_files)} JSON files in {input_dir}.")

    # Apply file limit if specified
    if file_limit is not None and file_limit > 0:
        logger.info(f"Applying file limit: processing first {file_limit} files.")
        files_to_process = json_files[:file_limit]
    else:
        files_to_process = json_files
        logger.info(f"Processing all {len(files_to_process)} found JSON files.")

    processed_count = 0
    failed_count = 0
    start_time_all = time.time()

    for i, file_path in enumerate(files_to_process):
        logger.info(f"--- Processing file {i+1}/{len(files_to_process)}: {file_path.name} ---")
        start_time_file = time.time()
        doc_name = file_path.stem # Use filename without extension as doc name
        
        # --- Check if document already exists in DB ---
        try:
            with get_db() as db:
                existing_doc = db_crud.get_document_by_name(db, doc_name)
                if existing_doc:
                    logger.warning(f"Document '{doc_name}' (from file {file_path.name}) already exists in the database (ID: {existing_doc.doc_id}). Skipping ingestion for this file.")
                    failed_count += 1 # Count as skipped/failed for summary
                    continue # Skip to next file
        except Exception as e:
            logger.exception(f"Error checking for existing document '{doc_name}' in database. Skipping file {file_path.name}.")
            failed_count += 1
            continue
        # --- End check ---

        source_desc = f"json_dir:{file_path.name}" # Source includes filename

        try:
            precomputed_chunks = load_chunks_from_json(file_path)
            if not precomputed_chunks:
                logger.error(f"Failed to load valid chunks from {file_path.name}. Skipping.")
                failed_count += 1
                continue # Skip to next file

            logger.info(f"Loaded {len(precomputed_chunks)} chunks. Ingesting document '{doc_name}'...")
            success = _ingest_text_content(
                doc_name=doc_name,
                source_description=source_desc,
                precomputed_chunks=precomputed_chunks
            )
            if success:
                logger.info(f"Successfully ingested {file_path.name}.")
                processed_count += 1
            else:
                logger.error(f"Ingestion failed for {file_path.name}. Check previous logs.")
                failed_count += 1
        except Exception as e:
            logger.exception(f"An unexpected error occurred processing file {file_path.name}: {e}")
            failed_count += 1
        
        end_time_file = time.time()
        logger.info(f"Finished processing {file_path.name} in {end_time_file - start_time_file:.2f} seconds.")
        logger.info("---") # Separator between files

    end_time_all = time.time()
    logger.info("--- Directory Ingestion Summary ---")
    logger.info(f"Total files attempted: {len(files_to_process)}")
    logger.info(f"Successfully processed: {processed_count}")
    logger.info(f"Failed/Skipped: {failed_count}")
    logger.info(f"Total time: {end_time_all - start_time_all:.2f} seconds")
    logger.info("---------------------------------")

# --- Example Usage / Testing Area --- #
if __name__ == "__main__":
    # Removed Gatsby example
    # Added test for ingesting pre-chunked JSON
    from src.ingestion.loaders import load_chunks_from_json
    import time

    JSON_CHUNK_FILE = Path("./DE_2022_derived.json") # Assume file is in root
    DOC_NAME = "DE_2022_derived"
    SOURCE_DESC = f"json_chunks:{JSON_CHUNK_FILE.name}"

    logger.info(f"--- Testing Ingestion Pipeline with Pre-chunked JSON: {JSON_CHUNK_FILE.name} ---")

    if not JSON_CHUNK_FILE.exists():
        logger.error(f"Test file not found: {JSON_CHUNK_FILE.resolve()}")
        logger.error("Please place the DE_2022_derived.json file in the project root directory.")
    else:
        try:
            start_time = time.time()
            logger.info("Loading chunks from JSON...")
            precomputed_chunks = load_chunks_from_json(JSON_CHUNK_FILE)

            if precomputed_chunks:
                logger.info("Starting ingestion process with precomputed chunks...")
                success = _ingest_text_content(
                    doc_name=DOC_NAME,
                    source_description=SOURCE_DESC,
                    precomputed_chunks=precomputed_chunks
                )
                if success:
                    logger.info(f"Ingestion completed successfully for {DOC_NAME}.")
                else:
                    logger.error(f"Ingestion failed for {DOC_NAME}.")
            else:
                logger.error(f"Failed to load chunks from {JSON_CHUNK_FILE.name}.")
            
            end_time = time.time()
            logger.info(f"Ingestion test took {end_time - start_time:.2f} seconds.")
            logger.info(f"Check database ('{settings.database.db_path}') and Qdrant collection ('{settings.qdrant.collection_name}') for results.")
            logger.info(f"Use 'python -m src.cli inspect-db --show-chunks' to check the database.")

        except Exception as e:
            logger.exception(f"An error occurred during the JSON chunk ingestion test: {e}")

    logger.info(f"--- Pre-chunked JSON Ingestion Test Complete ---") 