import requests
import logging
from typing import List

from src.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants for Ollama interaction
OLLAMA_API_URL = f"{settings.llm.api_base_url}/api/chat"
OLLAMA_MODEL = settings.llm.model_name
# Using the prompt template dictionary and cannot_answer phrase from config
AVAILABLE_PROMPTS = settings.prompts
# Default prompt name if not specified
DEFAULT_PROMPT_NAME = "default"

def get_llm_answer(question: str, context_chunks: List[str], prompt_name: str = DEFAULT_PROMPT_NAME) -> str:
    """Gets an answer from the configured Ollama model based on the question, context, and a specified prompt name."""
    if not context_chunks:
        logger.warning("No context chunks provided to LLM. Returning default cannot answer phrase.")
        return "No context was provided to the LLM to answer the question."

    # Get the selected prompt template
    prompt_config = AVAILABLE_PROMPTS.get(prompt_name)
    if not prompt_config:
        logger.warning(f"Prompt name '{prompt_name}' not found in configuration. Falling back to default prompt '{DEFAULT_PROMPT_NAME}'.")
        prompt_config = AVAILABLE_PROMPTS.get(DEFAULT_PROMPT_NAME)
        if not prompt_config: # Should not happen if default is set, but good practice
            logger.error("Default prompt configuration is missing.")
            return "Error: Default prompt configuration is missing."
    
    prompt_template = prompt_config.template
    logger.info(f"Using prompt: '{prompt_name}'")

    # Combine context chunks into a single string
    context_str = "\n\n".join(context_chunks)

    # Format the full prompt template
    full_prompt_content = prompt_template.format(context=context_str, question=question)

    # Prepare the payload for Ollama /api/chat
    # Send the whole formatted prompt as a single user message
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            { "role": "user", "content": full_prompt_content }
        ],
        "stream": False # Request a single response object
    }

    logger.debug(f"Sending request to Ollama model {OLLAMA_MODEL} at {OLLAMA_API_URL}")
    # logger.debug(f"Payload: {payload}") # Uncomment carefully, can be very verbose

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=60 # Add a timeout (adjust as needed)
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        response_data = response.json()
        # logger.debug(f"Ollama response data: {response_data}")

        # Extract the assistant's full message content
        if "message" in response_data and "content" in response_data["message"]:
            answer = response_data["message"]["content"].strip()
            logger.info(f"LLM ({OLLAMA_MODEL}) generated answer (length: {len(answer)}).")
            return answer
        else:
            logger.error(f"Unexpected response structure from Ollama: {response_data}")
            return "Error: Could not parse LLM response."

    except requests.exceptions.Timeout:
        logger.error(f"Request to Ollama timed out: {OLLAMA_API_URL}")
        return "Error: LLM request timed out."
    except requests.exceptions.RequestException as e:
        logger.exception(f"Error communicating with Ollama API at {OLLAMA_API_URL}: {e}")
        return f"Error: Could not connect to LLM. ({e})"
    except Exception as e:
        logger.exception("An unexpected error occurred during LLM interaction.")
        return "Error: An unexpected error occurred while getting the answer."

# Example Usage
if __name__ == "__main__":
    logger.info("--- Testing LLM Interface (Ollama) ---")

    # Ensure Ollama is running with the configured model, e.g., llama3:latest
    test_question = "What kind of parties did Gatsby throw?"
    test_context = [
        "In his blue gardens men and girls came and went like moths among the whisperings and the champagne and the stars.",
        "There was music from my neighbor's house through the summer nights. In his blue gardens men and girls came and went like moths among the whisperings and the champagne and the stars.",
        "At least once a fortnight a corps of caterers came down with several hundred feet of canvas and enough colored lights to make a Christmas tree of Gatsby's enormous garden."
    ]

    logger.info(f"Test Question: {test_question}")
    logger.info(f"Test Context Chunks: {len(test_context)}")

    try:
        answer = get_llm_answer(test_question, test_context)
        print(f"\nLLM Answer:\n---\n{answer}\n---\n")
    except Exception as e:
        logger.error(f"Test failed: {e}")

    # Test with a specific prompt name (assuming 'gatsby' exists in config.yaml)
    logger.info("\nTesting with 'gatsby' prompt...")
    try:
        answer_gatsby = get_llm_answer(test_question, test_context, prompt_name="gatsby")
        print(f"\nLLM Answer (Gatsby Prompt):\n---\n{answer_gatsby}\n---\n")
    except Exception as e:
        logger.error(f"Gatsby prompt test failed: {e}")

    # Test with no context
    logger.info("\nTesting with no context...")
    try:
        answer_no_context = get_llm_answer(test_question, [])
        print(f"\nLLM Answer (No Context):\n---\n{answer_no_context}\n---\n")
        assert answer_no_context == "No context was provided to the LLM to answer the question."
        logger.info("No context test passed.")
    except Exception as e:
        logger.error(f"No context test failed: {e}")

    logger.info("--- LLM Interface Test Complete ---") 