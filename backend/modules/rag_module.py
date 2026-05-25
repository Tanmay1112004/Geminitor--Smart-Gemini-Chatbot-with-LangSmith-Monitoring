"""
rag_module.py — PDF / TXT ingestion into FAISS with LCEL RAG chain.
Uses GoogleGenerativeAIEmbeddings (no local model download required).
"""

import os
import tempfile
import logging

from fastapi import UploadFile
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

log = logging.getLogger(__name__)


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


async def process_document(file: UploadFile, model: str = "gemini-2.5-flash"):
    """
    Read, chunk, embed and index an uploaded document.
    Returns an LCEL chain: chain.invoke(question_str) -> answer_str
    """
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_community.vectorstores import FAISS

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

    if not documents:
        raise ValueError("No text could be extracted from the document.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    log.info("RAG: %d chunks from %s", len(chunks), file.filename)

    # Google embeddings — fast, no download, uses existing GOOGLE_API_KEY
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(model=model, temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant answering questions about an uploaded document.\n"
            "Use the context excerpts below to answer. If the context does not contain "
            "enough information, say so and answer from your general knowledge if possible.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ])

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
