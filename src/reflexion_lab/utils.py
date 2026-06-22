from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

from .schemas import QAExample, RunRecord


def normalize_answer(text: str) -> str:
    """HotpotQA-style normalization while preserving Unicode letters."""

    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(
        char if char.isalnum() or char.isspace() else " "
        for char in text
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_dataset(path: str | Path, limit: int | None = None) -> list[QAExample]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Dataset must contain a JSON array")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        raw = raw[:limit]
    return [QAExample.model_validate(item) for item in raw]


def save_jsonl(path: str | Path, records: Iterable[RunRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
