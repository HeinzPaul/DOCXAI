# 📄 DOCXAI – Document Chat Assistant

An intelligent **RAG-based (Retrieval-Augmented Generation) pipeline** to build a powerful document question-answering and chat assistant. Designed for structured ingestion, semantic search, and grounded response generation.

---

## 🚀 **Features**

✅ Ingests PDFs, DOCX, and text files  
✅ Semantic chunking and embedding storage  
✅ Fast retrieval with vector similarity search  
✅ LLM-based contextual response generation  
✅ Source citation for traceability  
✅ Modular pipeline for customisation

---

## 🛠️ **Architecture Overview**

1. **Ingestion Module**  
   - Parses documents into manageable chunks (e.g. 500-800 tokens).  
   - Stores chunk metadata (source, page number, chunk index).

2. **Embedding Generation**  
   - Uses `OpenAI Embeddings` for dense vector embeddings.

3. **Vector Store**  
   - Stores embeddings in `FAISS` for fast retrieval.

4. **Retriever & Reranker**  
   - Retrieves top-K relevant chunks using semantic similarity.  
   - Reranks using cross-encoders for improved relevance.

5. **Prompt Builder**  
   - Constructs final prompts combining user query with retrieved context.

6. **LLM Generator**  
   - Generates grounded, contextual responses using models like **GPT-4**.

7. **Frontend / API**  
   - Interfaces with Streamlit

---

## ⚙️ **Tech Stack**

| **Module** | **Options** |
|---|---|
| **Embeddings** | OpenAI Embeddings |
| **Vector DB** | FAISS |
| **Backend Framework** | LangChain, LlamaIndex |
| **Frontend** | Streamlit |
| **LLMs** | GPT-4 |

---


