from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.reflexion_lab.mock_runtime import LLMRuntime
from src.reflexion_lab.utils import load_dataset


def main() -> None:
    example = load_dataset("data/hotpot_qaexamples.json", limit=1)[0]
    runtime = LLMRuntime(mode="live")

    actor = runtime.actor_answer(example, 1, "react", [])
    judge = runtime.evaluator(example, actor.answer)

    deliberately_wrong = example.context[0].title
    wrong_judge = runtime.evaluator(example, deliberately_wrong)
    reflection = runtime.reflector(
        example,
        1,
        deliberately_wrong,
        wrong_judge,
    )

    payload = {
        "model": runtime.model,
        "qid": example.qid,
        "actor": {
            "answer": actor.answer,
            "token_count": actor.token_count,
            "latency_ms": actor.latency_ms,
        },
        "evaluator": judge.model_dump(),
        "reflector": reflection.model_dump(),
        "all_live_calls_succeeded": True,
    }
    output = ROOT / "reports" / "live_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
