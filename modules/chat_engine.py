"""
chat_engine.py — LangChain chain setup with conversation memory.
Supports multi-turn history, persona system messages, and retry logic.
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


PERSONA_PROMPTS = {
    "General AI": (
        "You are Geminitor, a helpful, smart, and friendly AI assistant. "
        "Answer clearly and concisely, and always suggest a relevant follow-up."
    ),
    "Code Assistant": (
        "You are Geminitor, an expert software engineer. Provide accurate, well-commented "
        "code solutions with clear explanations. Use markdown code blocks."
    ),
    "Medical Helper": (
        "You are Geminitor, a medical information assistant. Provide general health "
        "information while always recommending the user consult a qualified doctor for "
        "personal medical advice."
    ),
    "Study Buddy": (
        "You are Geminitor, an enthusiastic educational assistant. Break down complex "
        "concepts into simple terms, use examples and analogies, and encourage curiosity."
    ),
    "Creative Writer": (
        "You are Geminitor, a creative writing assistant with a vivid imagination. "
        "Be expressive, inventive, and help bring ideas to life with rich language."
    ),
}


def get_llm(model: str = "gemini-2.5-flash", temperature: float = 0.7, max_tokens: int = 2048):
    """Instantiate and return the Gemini LLM."""
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )


def get_chain(
    model: str,
    temperature: float,
    max_tokens: int,
    persona: str,
    chat_history: list,
):
    """
    Build a LangChain chain that includes the last 10 turns of conversation
    history so the model has multi-turn context.

    Args:
        model: Gemini model identifier.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.
        persona: Key into PERSONA_PROMPTS.
        chat_history: List of dicts with 'role' and 'content'.

    Returns:
        A runnable chain expecting {"question": <str>}.
    """
    system_message = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["General AI"])
    llm = get_llm(model, temperature, max_tokens)

    messages = [("system", system_message)]

    recent = chat_history[-20:]
    for msg in recent:
        if msg["role"] == "user":
            messages.append(("human", msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(("ai", msg["content"]))

    messages.append(("human", "{question}"))

    prompt = ChatPromptTemplate.from_messages(messages)
    return prompt | llm | StrOutputParser()


def get_followup_suggestion(user_input: str, response: str, llm) -> str:
    """
    Ask the LLM to suggest one concise follow-up question.

    Returns an empty string on failure.
    """
    try:
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Given the conversation below, suggest exactly one short, useful "
                "follow-up question the user could ask next. Return only the question.",
            ),
            ("human", f"User: {user_input}\nAssistant: {response}"),
        ])
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({}).strip()
    except Exception:
        return ""
