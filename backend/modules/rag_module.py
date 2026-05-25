"""
rag_module.py — PDF / TXT ingestion into FAISS with LCEL RAG chain.
Uses LangChain Expression Language (LCEL) — compatible with LangChain >=0.2.
"""

import os
import tempfile
from fastapi import UploadFile
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


async def process_document(file: UploadFile, model: str = "gemini-2.5-flash"):
    """
    Read, chunk, embed and index an uploaded document.
    Returns an LCEL chain: chain.invoke(question_str) -> answer_str
    """
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings

    content = await file.read()
    suffix = ".pdf" if file.filename.lower().endswith(".pdf") else ".txt"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")
        documents = loader.load()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(model=model, temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. Answer the question using ONLY the "
            "document context below. If the answer is not in the context, say "
            "\"I couldn't find that in the document.\"\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ])

    # LCEL chain — invoke with a plain string question
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
