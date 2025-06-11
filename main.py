from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os # to input paths
from langchain_core.documents import Document
from docx import Document as DocxDocument

def extract_text_from_docx(docx_path):
    doc = DocxDocument(docx_path)
    text = ""

    for para in doc.paragraphs:
        if para.text.strip():
            text += para.text + "\n"
    text_as_doc = Document(page_content = text)
    return [text_as_doc]


def extractor(pdf_path):
    if os.path.exists(pdf_path):
        if pdf_path.lower().endswith(".pdf"):

            loader = PDFPlumberLoader(pdf_path) # we load the file
            pages = loader.load() # we load it into documnent object

            print("Sucessfully loaded",len(pages), "from",pdf_path)
            return pages
        elif pdf_path.lower().endswith(".docx"):
            text = extract_text_from_docx(pdf_path)
            return text

    else:
        print("Path does not exist")

pdf_path = input("Enter the name of the path")
pages = extractor(pdf_path)
print(pages)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=200)
chunks = text_splitter.split_documents(pages)
print("Split into",len(chunks),"text chunks")
for i,chunk in enumerate(chunks):
    print(f"____Chunk{i+1}____")
    print(f"{chunk.page_content[:500]}")