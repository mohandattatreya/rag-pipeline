import logging
from typing import List, Dict, Any, Tuple

from src.query import query_rag
from src.llm_interface import get_llm_answer
from src.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# How many top chunks to pass to the LLM
CONTEXT_CHUNKS_FOR_LLM = 3

def answer_question_with_rag(question: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Orchestrates the RAG pipeline: query, rerank (optional), get context, call LLM."""
    logger.info(f"Processing question: '{question}'")

    # 1. Get relevant chunks using the existing query pipeline
    # We'll use reranking by default if available, adjust top_k for LLM context size
    try:
        # Fetch slightly more chunks than needed initially if reranking
        # The query_rag function handles the reranking logic internally now.
        # It returns results already sorted (by rerank score if enabled, else by initial score).
        retrieved_chunks = query_rag(question, top_k=CONTEXT_CHUNKS_FOR_LLM, use_reranker=True)
    except Exception as e:
        logger.exception("Error during RAG retrieval phase.")
        return "Error: Failed to retrieve relevant context.", []

    if not retrieved_chunks:
        logger.warning("No relevant chunks found for the question.")
        # Decide if we should still ask the LLM or return immediately
        # Asking LLM with no context should ideally trigger its "cannot answer" logic based on the prompt.
        llm_context = []
    else:
        # Extract the text content of the top chunks to pass to the LLM
        llm_context = [chunk['text'] for chunk in retrieved_chunks]
        logger.info(f"Passing {len(llm_context)} chunks to LLM as context.")

    # 2. Call LLM with the question and context
    try:
        llm_answer = get_llm_answer(question, llm_context)
    except Exception as e:
        logger.exception("Error during LLM call phase.")
        llm_answer = "Error: Failed to get answer from LLM."

    # 3. Return the LLM answer and the source chunks used
    # Return the original retrieved_chunks dicts which include scores etc.
    return llm_answer, retrieved_chunks

# Example Usage
if __name__ == "__main__":
    import pprint

    # Assumes Ollama is running and data for Gatsby has been ingested
    logger.info("--- Testing Full RAG App Logic ---")

    test_question = "What is the relationship between Gatsby and Daisy?"
    logger.info(f"Test Question: {test_question}")

    try:
        final_answer, source_chunks = answer_question_with_rag(test_question)

        print(f"\nLLM Final Answer:\n---\n{final_answer}\n---")

        print(f"\nSource Chunks Used ({len(source_chunks)}):")
        # Print sources without the full text for brevity
        source_chunks_preview = [
            {k: v for k, v in chunk.items() if k != 'text'}
            for chunk in source_chunks
        ]
        pprint.pprint(source_chunks_preview, indent=2, width=120)

    except Exception as e:
        logger.exception(f"An error occurred during the app logic test: {e}")

    logger.info("--- App Logic Test Complete ---") 