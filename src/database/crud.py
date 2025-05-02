from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List, Optional

from src.database.models import Document, Chunk

# --- Document CRUD Operations ---

def add_document(db: Session, doc_name: str, source: Optional[str] = None) -> Optional[Document]:
    """Adds a new document to the database. Returns the Document object or None if name exists."""
    existing_doc = get_document_by_name(db, doc_name)
    if existing_doc:
        print(f"Document with name '{doc_name}' already exists. Returning existing.")
        return existing_doc # Or raise an exception, depending on desired behavior

    new_doc = Document(doc_name=doc_name, source=source)
    try:
        db.add(new_doc)
        db.flush() # Use flush to get the ID before commit
        db.refresh(new_doc)
        print(f"Added Document: {new_doc}")
        return new_doc
    except IntegrityError:
        db.rollback() # Should not happen due to check above, but good practice
        print(f"Error: Integrity constraint violated trying to add document '{doc_name}'.")
        return None
    except Exception as e:
        db.rollback()
        print(f"Error adding document '{doc_name}': {e}")
        return None

def get_document_by_id(db: Session, doc_id: int) -> Optional[Document]:
    """Retrieves a document by its primary key."""
    return db.query(Document).filter(Document.doc_id == doc_id).first()

def get_document_by_name(db: Session, doc_name: str) -> Optional[Document]:
    """Retrieves a document by its unique name."""
    return db.query(Document).filter(Document.doc_name == doc_name).first()

def get_all_documents(db: Session) -> List[Document]:
    """Retrieves all documents."""
    return db.query(Document).all()

def get_distinct_sources(db: Session) -> List[str]:
    """Retrieves a list of unique source descriptions from all documents."""
    results = db.query(Document.source).distinct().all()
    # Results are tuples like ('file:/path/to/doc.txt',), extract the string
    sources = [result[0] for result in results if result[0] is not None]
    # Sort for consistent display
    return sorted(sources)

# --- Chunk CRUD Operations ---

def add_chunk(db: Session, doc_id: int, chunk_number: int, chunk_text: str) -> Optional[Chunk]:
    """Adds a new chunk linked to a document. Returns the Chunk object or None on error."""
    # Optional: Check if document exists
    # doc = get_document_by_id(db, doc_id)
    # if not doc:
    #     print(f"Error: Document with ID {doc_id} not found. Cannot add chunk.")
    #     return None

    new_chunk = Chunk(
        doc_id=doc_id,
        chunk_number=chunk_number,
        chunk_text=chunk_text
    )
    try:
        db.add(new_chunk)
        db.flush() # Get ID before commit
        db.refresh(new_chunk)
        # print(f"Added Chunk: {new_chunk}") # Can be verbose
        return new_chunk
    except IntegrityError:
        db.rollback()
        # This usually means the (doc_id, chunk_number) pair already exists
        print(f"Error: Chunk with doc_id={doc_id}, chunk_number={chunk_number} likely already exists.")
        return None
    except Exception as e:
        db.rollback()
        print(f"Error adding chunk for doc_id={doc_id}, chunk_number={chunk_number}: {e}")
        return None

def get_chunk_by_id(db: Session, chunk_id: int) -> Optional[Chunk]:
    """Retrieves a chunk by its primary key."""
    return db.query(Chunk).filter(Chunk.chunk_id == chunk_id).first()

def get_chunks_by_doc_id(db: Session, doc_id: int) -> List[Chunk]:
    """Retrieves all chunks associated with a specific document ID."""
    return db.query(Chunk).filter(Chunk.doc_id == doc_id).order_by(Chunk.chunk_number).all()

def get_chunks_by_ids(db: Session, chunk_ids: List[int]) -> List[Chunk]:
    """Retrieves multiple chunks by their primary keys."""
    if not chunk_ids:
        return []
    return db.query(Chunk).filter(Chunk.chunk_id.in_(chunk_ids)).all()


# Example usage (can be removed later):
if __name__ == "__main__":
    from src.database.core import get_db, init_db

    # Initialize DB (creates file and tables if not present)
    init_db()

    print("\n--- Testing CRUD Operations ---")
    with get_db() as db:
        print("Adding documents...")
        doc1 = add_document(db, doc_name="doc1.txt", source="/path/to/doc1.txt")
        doc2 = add_document(db, doc_name="doc2.pdf")
        doc_dup = add_document(db, doc_name="doc1.txt") # Test duplicate

        if doc1 and doc2:
            print("\nAdding chunks...")
            chunk1_1 = add_chunk(db, doc_id=doc1.doc_id, chunk_number=0, chunk_text="This is chunk 0 of doc 1.")
            chunk1_2 = add_chunk(db, doc_id=doc1.doc_id, chunk_number=1, chunk_text="This is chunk 1 of doc 1.")
            chunk2_1 = add_chunk(db, doc_id=doc2.doc_id, chunk_number=0, chunk_text="This is chunk 0 of doc 2.")
            chunk_dup = add_chunk(db, doc_id=doc1.doc_id, chunk_number=0, chunk_text="Attempt duplicate chunk.") # Test duplicate

            print("\nRetrieving data...")
            retrieved_doc = get_document_by_name(db, "doc1.txt")
            print(f"Retrieved Document by name: {retrieved_doc}")

            retrieved_chunks = get_chunks_by_doc_id(db, doc1.doc_id)
            print(f"Retrieved Chunks for doc_id={doc1.doc_id}:")
            for chunk in retrieved_chunks:
                print(f"  {chunk}")

            if chunk1_1 and chunk2_1:
                 multi_chunks = get_chunks_by_ids(db, [chunk1_1.chunk_id, chunk2_1.chunk_id])
                 print(f"Retrieved Multiple Chunks by IDs: {multi_chunks}")

        else:
            print("Failed to add initial documents.")

    print("\n--- CRUD Test Complete ---")
    # Note: Data remains in the database file (rag_pipeline.db by default) 