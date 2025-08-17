from pathlib import Path

def extract_pdf_text(pdf_path: Path) -> str:
    """
    Extract text from PDF using PyMuPDF (fast, low-mem) with fallback to pdfminer.six.
    """
    text = ""
    try:
        import fitz  # PyMuPDF
        with fitz.open(pdf_path) as doc:
            parts = []
            for page in doc:
                parts.append(page.get_text("text"))
            text = "\n".join(parts)
    except Exception:
        try:
            from pdfminer.high_level import extract_text
            text = extract_text(str(pdf_path)) or ""
        except Exception:
            text = ""
    return text

def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
