import datetime
from sqlalchemy import (Column, Integer, String, Text, DateTime, ForeignKey,
                        UniqueConstraint, create_engine)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"

    doc_id = Column(Integer, primary_key=True, autoincrement=True)
    doc_name = Column(String, nullable=False, unique=True)
    source = Column(String, nullable=True) # e.g., file path or URL
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship to Chunks (one-to-many)
    chunks = relationship("Chunk", back_populates="document")

    def __repr__(self):
        return f"<Document(doc_id={self.doc_id}, doc_name='{self.doc_name}', source='{self.source}')>"

class Chunk(Base):
    __tablename__ = "chunks"

    chunk_id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(Integer, ForeignKey("documents.doc_id"), nullable=False)
    chunk_number = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship to Document (many-to-one)
    document = relationship("Document", back_populates="chunks")

    # Ensure that chunk numbers are unique within a document
    __table_args__ = (UniqueConstraint("doc_id", "chunk_number", name="uq_doc_chunk"),)

    def __repr__(self):
        return f"<Chunk(chunk_id={self.chunk_id}, doc_id={self.doc_id}, chunk_number={self.chunk_number})>"

# Example of how to create the tables (usually managed by Alembic or a setup script)
if __name__ == "__main__":
    # In a real app, get the DB URL from config
    DATABASE_URL = "sqlite:///./temp_rag_pipeline.db" 
    engine = create_engine(DATABASE_URL)
    print("Creating database tables (if they don't exist)...")
    Base.metadata.create_all(bind=engine)
    print("Tables created.")
    # Clean up the temporary database
    import os
    os.remove("./temp_rag_pipeline.db")
    print("Temporary database removed.") 