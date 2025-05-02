from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Optional, Tuple
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Globals --- #
_reranker_model: Optional[AutoModelForSequenceClassification] = None
_reranker_tokenizer: Optional[AutoTokenizer] = None
_reranker_model_name: str = "BAAI/bge-reranker-large" # Default reranker

# Determine device (reuse logic from embeddings if desired, or keep separate)
def _get_reranker_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    else:
        return "cpu"

RERANKER_DEVICE = _get_reranker_device()

def load_reranker() -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Loads the BGE Reranker model and tokenizer (singleton pattern)."""
    global _reranker_model, _reranker_tokenizer

    if _reranker_model is None or _reranker_tokenizer is None:
        logger.info(f"Loading reranker model: {_reranker_model_name} onto device: {RERANKER_DEVICE}")
        try:
            # Load model
            _reranker_model = AutoModelForSequenceClassification.from_pretrained(
                _reranker_model_name
                # torch_dtype=torch.float16 # Uncomment for potential speed/memory savings if using CUDA
            ).to(RERANKER_DEVICE).eval() # Set to evaluation mode

            # Load tokenizer
            _reranker_tokenizer = AutoTokenizer.from_pretrained(_reranker_model_name)

            logger.info(f"Reranker model '{_reranker_model_name}' loaded successfully.")
        except Exception as e:
            logger.exception(f"Failed to load reranker model or tokenizer '{_reranker_model_name}': {e}")
            # Clear globals on failure to allow retry on next call
            _reranker_model = None
            _reranker_tokenizer = None
            raise
    
    # Ensure model and tokenizer are not None before returning
    if _reranker_model is None or _reranker_tokenizer is None:
         raise RuntimeError("Reranker model or tokenizer failed to load previously and are still None.")

    return _reranker_model, _reranker_tokenizer

def compute_rerank_scores(query: str, documents: List[str]) -> List[float]:
    """Computes relevance scores for a query against a list of documents using the reranker."""
    if not documents:
        return []
        
    model, tokenizer = load_reranker()
    
    pairs = [[query, doc] for doc in documents]
    try:
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt').to(RERANKER_DEVICE)
            scores = model(**inputs, return_dict=True).logits.view(-1).float()
        
        # Convert scores to a list of Python floats
        # Sigmoid can optionally be applied to get scores between 0 and 1, but raw logits often work well for ranking.
        # return torch.sigmoid(scores).tolist()
        return scores.tolist()

    except Exception as e:
        logger.exception(f"Error computing rerank scores for query '{query[:50]}...': {e}")
        # Return default low scores or raise, depending on desired behavior
        return [-float('inf')] * len(documents) 

# Example Usage
if __name__ == "__main__":
    logger.info("--- Testing Reranker Loading and Scoring ---")
    
    # 1. Load Model
    try:
        model, tokenizer = load_reranker()
        logger.info(f"Reranker model device: {model.device}")
    except Exception as e:
        logger.error(f"Failed to load reranker: {e}")
        exit(1)
        
    # 2. Compute Scores
    test_query = "What is the capital of France?"
    test_docs = [
        "Paris is the capital of France.",
        "Lyon is a major city in France.",
        "The Eiffel Tower is a famous landmark in Paris."
    ]

    try:
        scores = compute_rerank_scores(test_query, test_docs)
        logger.info(f"Rerank scores for query '{test_query}':")
        for doc, score in zip(test_docs, scores):
            logger.info(f"  Score: {score:.4f} - Doc: {doc}")
    except Exception as e:
        logger.error(f"Failed to compute rerank scores: {e}")

    logger.info("--- Reranker Test Complete ---") 