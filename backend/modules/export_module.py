"""
export_module.py — Chat export to plain text and PDF (fpdf2).
"""

from datetime import datetime


def export_to_txt(history: list) -> str:
    """Convert chat history to a formatted plain-text string."""
    lines = [
        "Geminitor Pro — Chat Export",
        "=" * 44,
        f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    for msg in history:
        role = "You" if msg.get("role") == "user" else "Geminitor"
        ts   = msg.get("timestamp", "")
        lines.append(f"[{role}]  {ts}")
        lines.append(msg.get("content", ""))
        lines.append("")
    return "\n".join(lines)


def export_to_pdf(history: list) -> bytes:
    """Render chat history to a PDF and return raw bytes."""
    from fpdf import FPDF

    class ChatPDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(16, 163, 127)
            self.cell(0, 10, "Geminitor Pro — Chat Export", ln=True, align="C")
            self.set_font("Helvetica", size=9)
            self.set_text_color(130, 130, 130)
            self.cell(0, 6, f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
            self.ln(4)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    pdf = ChatPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    for msg in history:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        ts      = msg.get("timestamp", "")
        label   = f"You  [{ts}]" if role == "user" else f"Geminitor  [{ts}]"

        if role == "user":
            pdf.set_fill_color(47, 47, 47)
            pdf.set_text_color(236, 236, 236)
        else:
            pdf.set_fill_color(25, 45, 40)
            pdf.set_text_color(16, 200, 160)

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, label, ln=True, fill=True)
        pdf.set_font("Helvetica", size=10)
        pdf.set_text_color(30, 30, 30)
        safe = content.encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(0, 6, safe)
        pdf.ln(3)

    return bytes(pdf.output())
