import streamlit as st
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Add Project Root to PYTHONPATH --- #
# This ensures that imports like `from src.config...` work correctly
# when running `streamlit run streamlit_app.py` from the project root.
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
logger.debug(f"Added {project_root} to sys.path")

# --- Import Core Logic --- #
try:
    from src.app_logic import answer_question_with_rag
    from src.config import settings
except ImportError as e:
    logger.exception(f"Error importing application modules: {e}")
    st.error(
        f"Failed to import application modules. Please ensure the 'src' directory is accessible "
        f"and all dependencies are installed ('uv sync --extra app'). Error: {e}"
    )
    st.stop() # Stop execution if imports fail

# --- Streamlit App UI --- #

st.set_page_config(page_title="Gatsby RAG Q&A", layout="wide")

st.title("❓ Ask Questions About The Great Gatsby")
st.caption(f"Powered by RAG pipeline using Ollama model: {settings.llm.model_name}")

# Input form
with st.form("question_form"):
    user_question = st.text_area("Enter your question:", height=100)
    submit_button = st.form_submit_button("Get Answer")

if submit_button and user_question:
    st.divider()
    st.subheader("Answer")
    with st.spinner("Thinking... (Retrieving context and asking LLM)"):
        try:
            # Call the main application logic
            llm_answer, source_chunks = answer_question_with_rag(user_question)

            # Display the answer
            if llm_answer == settings.llm.cannot_answer_phrase:
                st.warning(llm_answer)
            elif llm_answer.startswith("Error:"):
                 st.error(llm_answer)
            else:
                st.markdown(llm_answer)

            # Display the sources if any were retrieved
            if source_chunks:
                st.divider()
                st.subheader(f"Sources ({len(source_chunks)} Chunks Used)")
                for i, chunk in enumerate(source_chunks):
                    # Safely format the score; handle missing values gracefully
                    score_value = chunk.get('rerank_score', chunk.get('initial_score'))
                    if isinstance(score_value, (float, int)):
                        score_str = f"{score_value:.4f}"
                    else:
                        score_str = "N/A"

                    with st.expander(
                        f"Source {i+1}: Chunk ID {chunk.get('sql_chunk_id', 'N/A')} (Score: {score_str})"
                    ):
                        st.markdown(f"**Document:** `{chunk.get('qdrant_payload', {}).get('doc_name', 'N/A')}`")
                        st.markdown(f"**Chunk Number in Doc:** `{chunk.get('chunk_number', 'N/A')}`")
                        # Use markdown with triple backticks for code block appearance
                        st.markdown(f"```text\n{chunk.get('text', 'N/A')}\n```")
                        # st.text_area("", value=chunk.get('text', 'N/A'), height=200, disabled=True, label_visibility="collapsed")
                        # st.write(chunk) # For debugging
            else:
                 st.info("No source chunks were retrieved to generate the answer.")

        except Exception as e:
            logger.exception(f"An error occurred processing the question: {e}")
            st.error(f"An unexpected error occurred: {e}")

elif submit_button and not user_question:
    st.warning("Please enter a question.") 