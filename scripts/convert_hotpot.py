from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def convert_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert one HotpotQA distractor record to the lab's QAExample schema."""

    difficulty = str(record.get("level", "hard")).lower()
    if difficulty not in VALID_DIFFICULTIES:
        difficulty = "hard"

    context = []
    for raw_chunk in record.get("context", []):
        if not isinstance(raw_chunk, list) or len(raw_chunk) != 2:
            raise ValueError(f"Invalid context chunk in {record.get('_id')}")
        title, sentences = raw_chunk
        if not isinstance(sentences, list):
            raise ValueError(f"Invalid sentence list in {record.get('_id')}")
        text = " ".join(str(sentence).strip() for sentence in sentences if str(sentence).strip())
        context.append({"title": str(title), "text": text})

    converted = {
        "qid": str(record["_id"]),
        "difficulty": difficulty,
        "question": str(record["question"]).strip(),
        "gold_answer": str(record["answer"]).strip(),
        "context": context,
    }
    validate_record(converted)
    return converted


def validate_record(record: dict[str, Any]) -> None:
    required = {"qid", "difficulty", "question", "gold_answer", "context"}
    if set(record) != required:
        raise ValueError(f"Unexpected QAExample keys: {set(record)}")
    if not all(record[key] for key in ("qid", "question", "gold_answer")):
        raise ValueError(f"Empty required value in {record.get('qid')}")
    if record["difficulty"] not in VALID_DIFFICULTIES:
        raise ValueError(f"Invalid difficulty in {record['qid']}")
    if not isinstance(record["context"], list) or not record["context"]:
        raise ValueError(f"Missing context in {record['qid']}")
    for chunk in record["context"]:
        if set(chunk) != {"title", "text"} or not chunk["title"] or not chunk["text"]:
            raise ValueError(f"Invalid context in {record['qid']}")


def convert_file(source: Path, destination: Path) -> int:
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("HotpotQA source must be a JSON array")

    converted = [convert_record(record) for record in raw]
    qids = {item["qid"] for item in converted}
    if len(qids) != len(converted):
        raise ValueError("Duplicate qid detected after conversion")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(converted, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return len(converted)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HotpotQA to QAExample JSON")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    count = convert_file(args.source, args.destination)
    print(f"Converted {count} records to {args.destination}")


if __name__ == "__main__":
    main()
