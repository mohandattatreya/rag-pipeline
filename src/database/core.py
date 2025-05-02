from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator

from src.config import settings
from src.database.models import Base # Import Base from models

# Create the SQLAlchemy engine using the URL from settings
engine = create_engine(settings.database.sqlalchemy_database_url, echo=False)
# For SQLite, echo=True can be very verbose. Set to True for debugging SQL.

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initializes the database by creating tables based on the models."""
    print(f"Initializing database at: {settings.database.sqlalchemy_database_url}")
    # Create all tables defined in models that inherit from Base
    Base.metadata.create_all(bind=engine)
    print("Database initialization complete.")

# Dependency to get a DB session (useful for FastAPI or context managers)
@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Provides a transactional scope around a series of operations."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

# Example usage (can be removed later):
if __name__ == "__main__":
    # This will create the DB file and tables if they don't exist
    init_db()

    # Example of using the session context manager
    print("Testing database session...")
    try:
        with get_db() as db:
            # You could perform some test query here, e.g.:
            # result = db.execute(text("SELECT 1")).scalar()
            # print(f"Session test query result: {result}")
            print("Database session obtained successfully.")
    except Exception as e:
        print(f"Error during session test: {e}") 