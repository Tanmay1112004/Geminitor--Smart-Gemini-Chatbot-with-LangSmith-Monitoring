"""
chat_engine.py — LangChain + Gemini chain builder.
Supports synchronous responses and async streaming with multi-turn history.
"""

from typing import AsyncGenerator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

PERSONA_PROMPTS = {
    "General AI": (
        "You are Geminitor, a helpful, smart, and friendly AI assistant. "
        "Answer clearly and concisely."
    ),
    "Code Assistant": (
        "You are Geminitor, an expert software engineer. Provide accurate, "
        "well-commented code solutions using markdown code blocks."
    ),
    "Medical Helper": (
        "You are Geminitor, a medical information assistant. Provide general "
        "health information and always recommend consulting a qualified doctor."
    ),
    "Study Buddy": (
        "You are Geminitor, an enthusiastic educational assistant. Break down "
        "complex concepts with simple language, examples, and analogies."
    ),
    "Creative Writer": (
        "You are Geminitor, a creative writing assistant. Be expressive, "
        "inventive, and help bring ideas to life with vivid language."
    ),
}


def _build_prompt(persona: str, chat_history: list) -> ChatPromptTemplate:
    """Build a ChatPromptTemplate that includes the last 10 turns of history."""
    system_msg = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["General AI"])
    messages = [("system", system_msg)]
    for msg in chat_history[-20:]:
        role = "human" if msg.get("role") == "user" else "ai"
        messages.append((role, msg.get("content", "")))
    messages.append(("human", "{question}"))
    return ChatPromptTemplate.from_messages(messages)


def get_response(
    model: str,
    temperature: float,
    max_tokens: int,
    persona: str,
    chat_history: list,
    question: str,
) -> str:
    """Return a complete response string (non-streaming)."""
    llm = ChatGoogleGenerativeAI(
        model=model, temperature=temperature, max_output_tokens=max_tokens
    )
    chain = _build_prompt(persona, chat_history) | llm | StrOutputParser()
    return chain.invoke({"question": question})


async def stream_response(
    model: str,
    temperature: float,
    max_tokens: int,
    persona: str,
    chat_history: list,
    question: str,
) -> AsyncGenerator[str, None]:
    """Async-yield string chunks for SSE streaming."""
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=max_tokens,
        streaming=True,
    )
    chain = _build_prompt(persona, chat_history) | llm | StrOutputParser()
    async for chunk in chain.astream({"question": question}):
        yield chunk


def get_followup(model: str, user_input: str, response: str) -> str:
    """Return a single concise follow-up question suggestion."""
    llm = ChatGoogleGenerativeAI(model=model, temperature=0.8, max_output_tokens=128)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Suggest exactly one short follow-up question the user could ask. Return only the question."),
        ("human", f"User: {user_input}\nAssistant: {response}"),
    ])
    return (prompt | llm | StrOutputParser()).invoke({}).strip()
