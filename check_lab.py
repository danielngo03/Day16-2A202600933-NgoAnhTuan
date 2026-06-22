from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "data/hotpot_qaexamples.json",
    "src/reflexion_lab/schemas.py",
    "src/reflexion_lab/prompts.py",
    "src/reflexion_lab/mock_runtime.py",
    "src/reflexion_lab/agents.py",
    "src/reflexion_lab/reporting.py",
    "reports/mock_100/report.json",
    "reports/mock_100/report.md",
]


def main() -> int:
    errors: list[str] = []
    print("Lab 16 submission check\n")
    for relative in REQUIRED:
        exists = (ROOT / relative).exists()
        print(f"[{'OK' if exists else 'MISSING'}] {relative}")
        if not exists:
            errors.append(f"Missing {relative}")

    source_path = ROOT / "data" / "hotpot_dev_distractor_v1.json"
    source_removed = not source_path.exists()
    print(f"[{'OK' if source_removed else 'FAIL'}] Original HotpotQA file removed")
    if not source_removed:
        errors.append("Original HotpotQA file still exists")

    markers = 0
    for path in (ROOT / "src").rglob("*"):
        if path.suffix in {".py", ".md"}:
            markers += len(
                re.findall(
                    r"\b(?:TODO|NotImplemented|Student TODO)\b",
                    path.read_text(encoding="utf-8"),
                )
            )
    print(f"[{'OK' if markers == 0 else 'FAIL'}] Scaffold markers: {markers}")
    if markers:
        errors.append("Scaffold markers remain")

    dataset_path = ROOT / "data" / "hotpot_qaexamples.json"
    if dataset_path.exists():
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        dataset_ok = (
            len(payload) == 7405
            and len({item["qid"] for item in payload}) == 7405
            and all(
                set(item) == {
                    "qid",
                    "difficulty",
                    "question",
                    "gold_answer",
                    "context",
                }
                for item in payload
            )
        )
        print(f"[{'OK' if dataset_ok else 'FAIL'}] QAExample dataset: {len(payload)} records")
        if not dataset_ok:
            errors.append("Converted dataset is invalid")

    test_result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    print(test_result.stdout.strip())
    if test_result.returncode:
        print(test_result.stderr.strip())
        errors.append("Tests failed")

    report_path = ROOT / "reports" / "mock_100" / "report.json"
    if report_path.exists():
        grade = subprocess.run(
            [
                sys.executable,
                "autograde.py",
                "--report-path",
                str(report_path.relative_to(ROOT)),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        print(grade.stdout.strip())
        if "Auto-grade total: 100/100" not in grade.stdout:
            errors.append("Autograde is below 100/100")

    print()
    if errors:
        print("Submission is not ready:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Submission is ready for grading.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
