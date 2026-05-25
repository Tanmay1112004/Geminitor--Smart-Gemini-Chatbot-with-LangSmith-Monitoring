"""
export_module.py — Chat export to PDF (fpdf2) and Gemini-powered conversation summary.
"""

from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def export_chat_pdf(chat_history: list) -> bytes:
    """
    Render the full chat history to a formatted PDF document.

    Args:
        chat_history: List of message dicts with 'role', 'content', 'timestamp'.

    Returns:
        PDF file as raw bytes.
    """
    from fpdf import FPDF

    class ChatPDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(30, 90, 200)
            self.cell(0, 10, "Geminitor Pro — Chat Export", ln=True, align="C")
            self.set_font("Helvetica", size=9)
            self.set_text_color(120, 120, 120)
            self.cell(
                0, 6,
                f"Exported on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ln=True, align="C",
            )
            self.ln(4)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    pdf = ChatPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    for msg in chat_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")

        label = f"You  [{timestamp}]" if role == "user" else f"Geminitor  [{timestamp}]"

        if role == "user":
            pdf.set_fill_color(235, 241, 255)
            pdf.set_text_color(20, 60, 160)
        else:
            pdf.set_fill_color(240, 248, 240)
            pdf.set_text_color(20, 120, 40)

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, label, ln=True, fill=True)

        pdf.set_font("Helvetica", size=10)
        pdf.set_text_color(30, 30, 30)

        safe_content = content.encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(0, 6, safe_content)
        pdf.ln(3)

    return bytes(pdf.output())


def get_chat_summary(chat_history: list, llm) -> str:
    """
    Use Gemini to produce a 5-bullet-point summary of the conversation.

    Args:
        chat_history: List of message dicts.
        llm: An instantiated LangChain LLM.

    Returns:
        Summary string.
    """
    conversation = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Geminitor'}: {m['content']}"
        for m in chat_history
    )
    conversation = conversation[:5000]

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a concise summarizer. Summarize the conversation below "
            "in exactly 5 bullet points. Each bullet must start with '• '.",
        ),
        ("human", conversation),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({})
