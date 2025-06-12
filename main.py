from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_core.documents import Document
from docx import Document as DocxDocument
import nltk
from nltk.tokenize import sent_tokenize

nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.download('punkt_tab', download_dir=nltk_data_path)
    zip_path = os.path.join(nltk_data_path, 'tokenizers', 'punkt' , 'punkt.zip')
    extract_to = os.path.join(nltk_data_path, 'tokenizers', 'punkt')
    if os.path.exists(zip_path):
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

if nltk_data_path not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_path)

session='semantic' #'semantic' or 'text'

def extract_text_from_docx(docx_path):
    doc = DocxDocument(docx_path)
    text = ""
    for para in doc.paragraphs:
        if para.text.strip():
            text += para.text + "\n"
    text_as_doc = Document(page_content=text, metadata={"source": os.path.basename(docx_path), "file_type": "docx"})
    return [text_as_doc]


def text_chunker(pages, chunk_size = 500 , chunk_overlap = 200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)
    print("Split into", len(chunks), "text chunks")
    with open("output_text.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            header = f"____Chunk{i + 1}____\n"
            content = f"{chunk.page_content[:500]}\n"
            print(header)
            print(content)
            f.write(header)
            f.write(content)
    f.close()
    return chunks


def semantic_chunker(pages, chunk_size=500, chunk_overlap=200):
    chunks = []
    for doc_index, doc in enumerate(pages):
        sentences = sent_tokenize(doc.page_content)
        current_sentences_for_chunk = []

        for sentence_index, sentence in enumerate(sentences):
            current_sentence_text = sentence

            potential_chunk_sentences = current_sentences_for_chunk + [current_sentence_text]
            potential_chunk_text = " ".join(potential_chunk_sentences)

            if current_sentences_for_chunk and len(potential_chunk_text) > chunk_size:
                chunk_to_emit_text = " ".join(current_sentences_for_chunk)
                chunks.append(Document(page_content=chunk_to_emit_text))

                if len(chunk_to_emit_text) > chunk_size:
                    overlap_text_for_next_chunk = chunk_to_emit_text[-chunk_overlap:]
                    current_sentences_for_chunk = [overlap_text_for_next_chunk]
                else:
                    overlap_sentences_for_next_chunk = []
                    temp_overlap_candidate_sentences = []
                    for s_overlap in reversed(current_sentences_for_chunk):
                        temp_overlap_candidate_sentences.insert(0, s_overlap)
                        current_overlap_joined_text = " ".join(temp_overlap_candidate_sentences)
                        if len(current_overlap_joined_text) >= chunk_overlap:
                            break
                    overlap_sentences_for_next_chunk = temp_overlap_candidate_sentences
                    current_sentences_for_chunk = overlap_sentences_for_next_chunk[:]

                current_sentences_for_chunk.append(current_sentence_text)
            else:
                current_sentences_for_chunk.append(current_sentence_text)

        if current_sentences_for_chunk:
            final_chunk_text = " ".join(current_sentences_for_chunk)
            chunks.append(Document(page_content=final_chunk_text))

    output_filename = "output_semantic.txt"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            for i, chunk_doc in enumerate(chunks):
                header = f"____Chunk{i + 1}____\n"
                page_content_to_write = chunk_doc.page_content
                content_display = page_content_to_write[:chunk_size]

                content_display_full = content_display
                if len(page_content_to_write) > chunk_size:
                    content_display_full += f"\n... (actual length: {len(page_content_to_write)}, chunk_size: {chunk_size})"

                f.write(header)
                f.write(content_display_full + "\n")
        print(f"Production chunking output written to {output_filename}")
    except Exception as e:
        print(f"Error writing production output file: {e}")

    return chunks


def extractor(file_path):
    if os.path.exists(file_path):
        if file_path.lower().endswith(".pdf"):

            loader = PDFPlumberLoader(file_path) # we load the file
            pages = loader.load() # we load it into documnent object

            print("Sucessfully loaded",len(pages), "from",file_path)
            return pages
        elif file_path.lower().endswith(".docx"):
            text = extract_text_from_docx(file_path)
            return text

    else:
        print("Path does not exist")


def main():
    file_path = input("Enter the name of the path")
    if session=='text':
        pages = extractor(file_path)
        print(pages)
        chunks = text_chunker(pages, 500, 200)

    elif session=='semantic':
        pages = extractor(file_path)
        print(f"Number of documents (pages) extracted: {len(pages)}")
        for i, doc_item in enumerate(pages):
            print(f"Content of doc {i} (first 150 chars): {doc_item.page_content[:150]}")
        chunks = semantic_chunker(pages, 500, 200)


main()

