# app.py

import streamlit as st
import os
from dotenv import load_dotenv
import shutil

# Import functions from your existing project files
from main import extractor, text_chunker, semantic_chunker
from embedding import embed_and_store_with_faiss, load_faiss_index, hybrid_rerank_with_cross_encoder
from rag import generate_answer_from_chunks

# --- Page Configuration ---
st.set_page_config(
    page_title="Document Q&A with RAG",
    page_icon="📚",
    layout="wide"
)

# --- Load Environment Variables ---
load_dotenv()
api_key = os.getenv("OPEN_AI_KEY")

if not api_key:
    st.error("OPEN_AI_KEY not found in environment variables. Please set it in a .env file or as a Streamlit secret.")
    st.stop()

# This import triggers the NLTK download check from main.py
import main as main_setup

# --- Constants ---
FAISS_INDEX_PATH = "faiss_index_store"
TEMP_DIR = "temp_uploaded_files"


# --- Helper Functions ---
def setup_directories():
    """Create necessary directories."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = None


# --- UI Layout ---
st.title("📄 Document Q&A with RAG")
st.markdown(
    "Upload your PDF or DOCX documents, ingest them into a vector store, and ask questions about their content.")

# Initialize
setup_directories()
initialize_session_state()

# --- Sidebar for Document Ingestion ---
with st.sidebar:
    st.header("1. Ingest Documents")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX files",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    # Chunking strategy
    chunking_strategy = st.radio(
        "Select chunking strategy:",
        ('semantic', 'text'),
        index=0,  # Default to semantic
        help="**Semantic:** Keeps related sentences together. Better for Q&A. \n\n**Text:** Simple fixed-size chunks. Faster but less context-aware."
    )

    if st.button("Process and Ingest Documents"):
        if uploaded_files:
            # Clean up the FAISS index directory for a fresh start
            if os.path.exists(FAISS_INDEX_PATH):
                shutil.rmtree(FAISS_INDEX_PATH)
            os.makedirs(FAISS_INDEX_PATH)

            with st.spinner("Processing documents... This might take a while."):
                all_pages = []
                temp_file_paths = []
                for uploaded_file in uploaded_files:
                    temp_path = os.path.join(TEMP_DIR, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    temp_file_paths.append(temp_path)

                    try:
                        pages = extractor(temp_path)
                        if pages:
                            all_pages.extend(pages)
                            st.success(f"Successfully extracted content from {uploaded_file.name}")
                        else:
                            st.warning(f"Could not extract content from {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                        continue

                if all_pages:
                    # Step 2: Chunking
                    st.write("Chunking documents...")
                    if chunking_strategy == 'semantic':
                        chunks = semantic_chunker(all_pages, 5000, 800)
                    else:  # text
                        chunks = text_chunker(all_pages, 500, 200)
                    st.write(f"Created {len(chunks)} chunks.")

                    # Step 3: Embedding and Storing
                    st.write("Embedding chunks and creating FAISS index...")
                    try:
                        faiss_index = embed_and_store_with_faiss(
                            chunks=chunks,
                            openai_api_key=api_key,
                            save_path=FAISS_INDEX_PATH
                        )
                        st.session_state.faiss_index = faiss_index
                        st.success("Documents ingested successfully! You can now ask questions.")
                    except Exception as e:
                        st.error(f"Failed to create FAISS index: {e}")

                # Clean up temporary files
                for path in temp_file_paths:
                    os.remove(path)
        else:
            st.warning("Please upload at least one document.")

# --- Main Area for Q&A ---
st.header("2. Query Your Documents")

index_exists = os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss"))

if not index_exists:
    st.info("Please ingest some documents using the sidebar first.")
else:
    if st.session_state.faiss_index is None:
        with st.spinner("Loading vector index..."):
            try:
                st.session_state.faiss_index = load_faiss_index(FAISS_INDEX_PATH, openai_api_key=api_key)
            except Exception as e:
                st.error(f"Could not load the FAISS index. Please try re-ingesting documents. Error: {e}")
                st.stop()

    query = st.text_input("Enter your question:", placeholder="e.g., What are the main findings of the report?")

    if query:
        with st.spinner("Searching for answers..."):
            try:
                faiss_index = st.session_state.faiss_index
                candidate_docs = faiss_index.similarity_search(query, k=20)

                reranked_docs = hybrid_rerank_with_cross_encoder(query, candidate_docs, top_k=5)

                answer = generate_answer_from_chunks(query, reranked_docs, api_key)

                st.markdown("### Answer")
                st.markdown(answer)

                with st.expander("Show Sources"):
                    for i, doc in enumerate(reranked_docs):
                        st.markdown(f"**Source {i + 1}** (from `{doc.metadata.get('source', 'N/A')}`)")
                        st.info(doc.page_content)

            except Exception as e:
                st.error(f"An error occurred during the query process: {e}")