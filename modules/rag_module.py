"""
rag_module.py — PDF / TXT document ingestion with FAISS vector store
and HuggingFace sentence-transformer embeddings for Retrieval-Augmented Generation.
"""

import os
import tempfile

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI


def process_document(uploaded_file, model: str = "gemini-2.5-flash"):
    """
    Load, chunk, embed, and index an uploaded PDF or TXT file.
    Returns a RetrievalQA chain ready to answer questions about the document.

    Args:
        uploaded_file: Streamlit UploadedFile object.
        model: Gemini model to use for answer generation.

    Returns:
        A LangChain RetrievalQA chain.
    """
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.chains import RetrievalQA

    suffix = ".pdf" if uploaded_file.name.lower().endswith(".pdf") else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path, encoding="utf-8")

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    llm = ChatGoogleGenerativeAI(model=model, temperature=0.2)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=False,
    )

    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    return chain
