from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from typing import List
from langchain_core.documents import Document
import os
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def hybrid_rerank_with_cross_encoder(query: str, docs: list, top_k: int = 5):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)

    # Attach scores and sort
    scored_docs = sorted(zip(scores, docs), reverse=True, key=lambda x: x[0])
    return [doc for score, doc in scored_docs[:top_k]]

def embed_and_store_with_faiss(chunks: List[Document], openai_api_key: str, save_path: str = "faiss_index_store") -> FAISS:
    """Embed the given chunks using OpenAI and store them in a FAISS index."""
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

    if os.path.exists(os.path.join(save_path, "index.faiss")):
        print("Loading existing FAISS index...")
        faiss_index = FAISS.load_local(save_path, embedding_model, allow_dangerous_deserialization=True)
        faiss_index.add_documents(chunks)
        faiss_index.save_local(save_path)
        print("Saved")
    else:
        print("Creating new FAISS index...")
        faiss_index = FAISS.from_documents(chunks, embedding=embedding_model)
        faiss_index.save_local(save_path)
    return faiss_index


def load_faiss_index(save_path: str, openai_api_key: str) -> FAISS:
    """Load FAISS index from disk."""
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.load_local(save_path, embedding_model, allow_dangerous_deserialization=True)


def search_faiss(faiss_index: FAISS, query: str, k: int = 3):
    """Search for relevant documents using FAISS."""
    results = faiss_index.similarity_search(query, k=k)
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content[:300])
        print("Metadata:", doc.metadata)