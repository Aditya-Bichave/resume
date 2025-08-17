import re
from typing import List, Tuple, Optional

def get_nlp():
    """
    Try to load spaCy 'en_core_web_sm'. Return None if unavailable.
    """
    try:
        import spacy
        try:
            return spacy.load("en_core_web_sm")
        except Exception:
            return None
    except Exception:
        return None

_WORD = re.compile(r"[A-Za-z0-9+#.-]{2,}")

def basic_tokenize(text: str, nlp=None) -> Tuple[List[str], List[str]]:
    """
    Return (tokens, sentences). If spaCy is available, use it; else regex fallback.
    """
    if nlp:
        doc = nlp(text)
        tokens = [t.text.lower() for t in doc if not t.is_space]
        sents = [s.text.strip() for s in doc.sents]
        return tokens, sents
    # fallback
    sents = [s.strip() for s in re.split(r"[.!?]\s+", text) if s.strip()]
    tokens = [m.group(0).lower() for m in _WORD.finditer(text)]
    return tokens, sents
