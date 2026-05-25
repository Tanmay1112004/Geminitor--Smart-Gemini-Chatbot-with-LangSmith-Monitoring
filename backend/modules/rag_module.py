"""
rag_module.py — PDF / TXT ingestion into FAISS vector store for RAG.
Accepts a FastAPI UploadFile and returns a RetrievalQA chain.
"""

import os
import tempfile
from fastapi import UploadFile
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter


async def process_document(file: UploadFile, model: str = "gemini-2.5-flash"):
    """
    Read, chunk, embed, and index an uploaded document.
    Returns a LangChain RetrievalQA chain.
    """
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.chains import RetrievalQA

    content = await file.read()
    suffix = ".pdf" if file.filename.lower().endswith(".pdf") else ".txt"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path, encoding="utf-8")
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

    llm = ChatGoogleGenerativeAI(model=model, temperature=0.2)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=False,
    )
