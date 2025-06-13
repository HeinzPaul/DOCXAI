import os
from embedding import embed_and_store_with_faiss
from dotenv import load_dotenv
from embedding import load_faiss_index, hybrid_rerank_with_cross_encoder
from rag import generate_answer_from_chunks
from parsing import extractor
from chunker import text_chunker, semantic_chunker, nltk_check, session

load_dotenv()
api_key = os.getenv("OPEN_AI_KEY")

if __name__ == "__main__":
    nltk_check()
    task = input("Enter what to do - embed or query ")
    if task == "embed":
        file_path = input("Enter the name of the path")
        if session=='text':
            pages = extractor(file_path)
            print(pages)
            chunks = text_chunker(pages, 500, 200)

        elif session=='semantic':
            pages = extractor(file_path)
            output_parsed_text = "output_parsed_text"
            with open(output_parsed_text, "w", encoding="utf-8") as f:
                for i, doc_item in enumerate(pages):
                        # Correct way to access and write page content
                        f.write(f"--- Page {i+1} ---\n")
                        f.write(doc_item.page_content)
                        f.write("\n\n") # Add some separation between pages
            print(f"Number of documents (pages) extracted: {len(pages)}")
            for i, doc_item in enumerate(pages):
                print(f"Content of doc {i} (first 150 chars): {doc_item.page_content[:150]}")
            chunks = semantic_chunker(pages, 5000, 800)
            faiss_index = embed_and_store_with_faiss(
                chunks=chunks,
                openai_api_key=api_key,
                save_path="faiss_index_store"
            )
    elif task == "query":
        faiss_index = load_faiss_index('faiss_index_store', openai_api_key=api_key)
        query = input("Enter query - ")
        candidates = faiss_index.similarity_search(query, k=20)
        results = hybrid_rerank_with_cross_encoder(query, candidates, top_k=3)
        answer = generate_answer_from_chunks(query, results, api_key)
        print(answer)


