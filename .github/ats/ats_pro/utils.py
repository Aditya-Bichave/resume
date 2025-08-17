import json
from pathlib import Path
from typing import Any, Optional

def load_json_if_exists(path: Optional[Path]) -> Optional[Any]:
    if not path:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def as_float(v, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default
