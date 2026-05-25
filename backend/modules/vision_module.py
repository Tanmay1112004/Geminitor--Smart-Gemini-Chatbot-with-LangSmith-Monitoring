"""
vision_module.py — Multimodal image analysis via Gemini Vision.
Accepts a FastAPI UploadFile and a question string.
"""

import base64
from fastapi import UploadFile
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


async def analyze_image(
    file: UploadFile,
    question: str = "Describe this image in detail.",
    model: str = "gemini-2.5-flash",
) -> str:
    """Send image + question to Gemini Vision and return the text response."""
    img_bytes = await file.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    name = (file.filename or "").lower()
    if name.endswith(".png"):
        media_type = "image/png"
    elif name.endswith(".webp"):
        media_type = "image/webp"
    else:
        media_type = "image/jpeg"

    llm = ChatGoogleGenerativeAI(model=model, temperature=0.3)
    message = HumanMessage(content=[
        {"type": "text", "text": question or "Describe this image in detail."},
        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{img_b64}"}},
    ])
    response = llm.invoke([message])
    return response.content
