"""
vision_module.py — Multimodal image understanding via Gemini Vision.
Accepts an uploaded image and a text question, returns a descriptive response.
"""

import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


def analyze_image(image_file, question: str = "", model: str = "gemini-2.5-flash") -> str:
    """
    Send an image plus an optional question to Gemini Vision and return the response.

    Args:
        image_file: Streamlit UploadedFile object (jpg, jpeg, png, webp).
        question: Text question to accompany the image.
        model: Gemini model identifier supporting vision.

    Returns:
        Model response as a string.
    """
    img_bytes = image_file.read()
    image_file.seek(0)

    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    name = image_file.name.lower()
    if name.endswith(".png"):
        media_type = "image/png"
    elif name.endswith(".webp"):
        media_type = "image/webp"
    elif name.endswith(".gif"):
        media_type = "image/gif"
    else:
        media_type = "image/jpeg"

    text_prompt = question.strip() if question.strip() else "Describe this image in detail."

    llm = ChatGoogleGenerativeAI(model=model, temperature=0.3)

    message = HumanMessage(
        content=[
            {"type": "text", "text": text_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{img_b64}"},
            },
        ]
    )

    response = llm.invoke([message])
    return response.content
