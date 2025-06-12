from openai import OpenAI

def generate_answer_from_chunks(query: str, chunks: list, api_key: str) -> str:
    client = OpenAI(api_key=api_key)

    # Combine chunks into a single context block
    context = "\n\n".join([doc.page_content for doc in chunks])

    metadata = "\n\n".join([f"{doc.metadata}" for doc in chunks])

    prompt = f"""You are an expert assistant tasked with answering user queries using the retrieved context documents. Each document chunk has been semantically segmented to retain structural references such as article numbers, section titles, and headings. Use the content to generate an accurate, clear, and complete answer.

Your answer **must**:
1. Answer the question directly and concisely.
2. Clearly state where the information was retrieved from by citing:
   - **Article and section numbers** for legal documents,
   - **Chapter names, headings, or subheadings** for textbooks or manuals,
   - **Section titles, figure/table references, or paragraph positions** for research papers.
3. If multiple relevant chunks are retrieved, synthesize them into a unified answer and mention each source context explicitly.
4. Do not fabricate sources or content. Only cite when the relevant detail exists in the provided context.

Follow the following format:

**Answer**: (Complete Answer with references in brackets)
**Reference**: Reference, Document name

Use the metadata to obtain the document name

If no answer can be confidently given from the provided context, say so clearly and do not guess.

Context:
{context}

User Question:
{query}

Metadata:
{metadata}
"""

    # Call OpenAI Chat Completion API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()