import streamlit as st
import sys
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
import requests # Import requests library

# --- Add project root to PYTHONPATH ---
# This allows importing modules from the 'src' directory.
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Added {project_root} to sys.path")
# --- End PYTHONPATH ---

# Remove direct backend imports
# from src.config import settings # Settings still needed potentially for API URL
# from src.query import query_rag
# from src.llm_interface import get_llm_answer
# from src.database.core import get_db
# from src.database import crud as db_crud

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# TODO: Make API URL configurable (e.g., via config.yaml or env var)
API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Pipeline Query Interface", layout="wide")

st.title("Query Your Documents")
st.write("Ask questions about the documents ingested into the RAG pipeline.")

# --- Helper Functions to Fetch Data from API ---
@st.cache_data(ttl=300) # Cache sources for 5 minutes
def get_available_sources_from_api() -> List[str]:
    """Fetches distinct document sources from the backend API."""
    try:
        # Increase timeout for initial source fetching
        response = requests.get(f"{API_BASE_URL}/sources/", timeout=30)
        response.raise_for_status()
        data = response.json()
        sources = data.get('sources', [])
        logger.info(f"Fetched sources from API: {sources}")
        return sorted(sources) # Keep sorted for consistency
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch sources from API: {e}")
        st.error(f"Error fetching document sources from API: {e}")
        return [] # Return empty list on error

@st.cache_data(ttl=300) # Cache prompts for 5 minutes
def get_available_prompts_from_api() -> List[str]:
    """Fetches available prompt names from the backend API."""
    try:
        # Increase timeout for initial prompt fetching
        response = requests.get(f"{API_BASE_URL}/prompts/", timeout=30)
        response.raise_for_status()
        data = response.json()
        prompt_names = data.get('prompt_names', [])
        logger.info(f"Fetched prompts from API: {prompt_names}")
        return sorted(prompt_names)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch prompts from API: {e}")
        st.error(f"Error fetching prompts from API: {e}")
        return [] # Return empty list on error

# --- Source Selection ---
available_sources = get_available_sources_from_api()
source_options = ["All Sources"] + available_sources

# Check if sources were actually loaded before trying to display
if available_sources or not st.session_state.get('api_fetch_error', False):
    selected_source = st.selectbox(
        "Select Document Source:",
        options=source_options,
        index=0, # Default to "All Sources"
        help="Filter queries to chunks from a specific ingested source, or search all."
    )
    source_filter = selected_source if selected_source != "All Sources" else None
else:
    st.warning("Could not load document sources from the backend API. Filtering disabled.")
    source_filter = None
    st.session_state['api_fetch_error'] = True # Flag error to avoid repeated messages if needed

# --- Prompt Selection ---
available_prompts = get_available_prompts_from_api()
if available_prompts and not st.session_state.get('api_fetch_error', False):
    # Try to find 'gatsby' or 'default' as the default index
    try:
        default_prompt_index = available_prompts.index('gatsby')
    except ValueError:
        try:
            default_prompt_index = available_prompts.index('default') # Assuming 'default' exists if gatsby doesn't
        except ValueError:
            default_prompt_index = 0 # Fallback to first available prompt
        
    selected_prompt_name = st.selectbox(
        "Select Prompt Template:",
        options=available_prompts,
        index=default_prompt_index,
        help="Choose the prompt style for the LLM to use."
    )
else:
    if not st.session_state.get('api_fetch_error'): # Avoid double warning
        st.warning("Could not load prompt templates from the backend API.")
    selected_prompt_name = None # No prompt selection possible
    st.session_state['api_fetch_error'] = True

# --- Query Input ---
query_text = st.text_area("Enter your query:", height=100)

# --- Query Options ---
col1, col2 = st.columns(2)
with col1:
    top_k = st.slider("Number of Chunks (Top K):", min_value=1, max_value=10, value=3, help="How many text chunks to retrieve.")
with col2:
    use_reranker = st.toggle("Use Reranker", value=False, help="Apply a reranker model after initial retrieval (requires backend support!).")

# --- Submit Button ---
# Disable button if prompts couldn't be loaded
submit_button = st.button("Submit Query", disabled=(selected_prompt_name is None))

# --- Query Execution and Display ---
if submit_button and query_text and selected_prompt_name:
    st.divider()
    st.subheader("Results")

    # Prepare payload for API request
    payload = {
        "query_text": query_text,
        "top_k": top_k,
        "use_reranker": use_reranker,
        "source_filter": source_filter,
        "prompt_name": selected_prompt_name
    }
    
    # Call the backend API
    try:
        with st.spinner("Sending query to backend API..."):
            response = requests.post(f"{API_BASE_URL}/query/", json=payload, timeout=90) # Increase timeout for potentially long RAG+LLM calls
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
        api_response_data = response.json()
        final_answer = api_response_data.get("answer")
        retrieved_chunks = api_response_data.get("retrieved_chunks", [])
        
        if final_answer is None:
            st.error("API response missing 'answer' field.")
            st.stop()

    except requests.exceptions.Timeout:
        st.error("Request to backend API timed out. The RAG process took too long.")
        st.stop()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to backend API: {e}")
        # Try to get more details from the response if available (e.g., for 500 errors)
        try: 
            error_detail = e.response.json().get('detail', 'No detail provided')
            st.error(f"API Error Detail: {error_detail}")
        except Exception:
            pass # Ignore errors trying to parse error details
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.stop()
        
    # Display Answer
    st.markdown("#### Answer")
    st.markdown(final_answer)

    # Display Retrieved Chunks (Optional)
    with st.expander("Show Retrieved Text Chunks"):
        if retrieved_chunks:
            for i, chunk_detail in enumerate(retrieved_chunks):
                # Access data using dictionary keys based on ChunkDetail model
                score = chunk_detail.get('score')
                # Format score if it's a number, otherwise display as is (or N/A)
                if isinstance(score, (float, int)):
                    score_display = f"{score:.4f}"
                else:
                    score_display = "N/A"
                
                st.markdown(f"**Chunk {i+1} (Score: {score_display})**")
                st.markdown(f"> Source: `{chunk_detail.get('source', 'N/A')}` | SQL Chunk ID: `{chunk_detail.get('sql_chunk_id', 'N/A')}`")
                st.text(chunk_detail.get('text', 'N/A'))
                st.divider()
        else:
            st.write("No chunks were retrieved or provided by the API.")

elif submit_button and not query_text:
    st.warning("Please enter a query.")
elif submit_button and not selected_prompt_name:
    st.error("Cannot submit query: Prompt templates failed to load from API.") 