from __future__ import annotations

import json

from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.mock_runtime import LLMRuntime
from src.reflexion_lab.reporting import build_report
from src.reflexion_lab.schemas import JudgeResult, QAExample, ReflectionEntry
from src.reflexion_lab.utils import load_dataset, normalize_answer


def sample_example(qid: str = "sample-04", difficulty: str = "hard") -> QAExample:
    return QAExample.model_validate(
        {
            "qid": qid,
            "difficulty": difficulty,
            "question": "Which river crosses the city where Ada Lovelace was born?",
            "gold_answer": "River Thames",
            "context": [
                {"title": "Ada Lovelace", "text": "Ada Lovelace was born in London."},
                {"title": "London", "text": "London is crossed by the River Thames."},
            ],
        }
    )


def test_normalize_answer() -> None:
    assert normalize_answer("The Oxford University!") == "oxford university"
    assert normalize_answer("Đà Nẵng") == "đà nẵng"


def test_dataset_is_qaexample_format() -> None:
    examples = load_dataset("data/hotpot_qaexamples.json", limit=100)
    assert len(examples) == 100
    assert len({example.qid for example in examples}) == 100
    assert all(example.context for example in examples)


def test_judge_and_reflection_schemas() -> None:
    judge = JudgeResult(
        score=0,
        reason="Stopped after first hop.",
        failure_mode="incomplete_multi_hop",
        missing_evidence=["river through London"],
        confidence=0.9,
    )
    reflection = ReflectionEntry(
        attempt_id=1,
        failure_reason=judge.reason,
        lesson="Complete all hops.",
        next_strategy="Find the birthplace, then the river.",
    )
    assert judge.score == 0
    assert reflection.next_strategy


def test_react_uses_one_attempt() -> None:
    runtime = LLMRuntime(mode="mock")
    record = ReActAgent(runtime).run(sample_example())
    assert record.attempts == 1
    assert len(record.traces) == 1
    assert record.token_estimate > 0
    assert record.latency_ms > 0


def test_reflexion_updates_memory_and_recovers() -> None:
    runtime = LLMRuntime(mode="mock")
    example = None
    # Find a deterministic mock failure without depending on a hardcoded answer.
    for index in range(100):
        candidate = sample_example(qid=f"candidate-{index}")
        if not ReActAgent(runtime).run(candidate).is_correct:
            example = candidate
            break
    assert example is not None

    record = ReflexionAgent(runtime, max_attempts=3).run(example)
    assert record.is_correct
    assert record.attempts == 2
    assert len(record.reflections) == 1
    assert record.traces[0].reflection is not None
    assert record.traces[1].score == 1


def test_adaptive_attempt_budget() -> None:
    runtime = LLMRuntime(mode="mock")
    agent = ReflexionAgent(runtime, max_attempts=4)
    assert agent._attempt_budget(sample_example(difficulty="easy")) == 2
    assert agent._attempt_budget(sample_example(difficulty="medium")) == 3
    assert agent._attempt_budget(sample_example(difficulty="hard")) == 4


def test_memory_compression_is_bounded() -> None:
    runtime = LLMRuntime(mode="mock")
    agent = ReflexionAgent(runtime, max_attempts=5, memory_limit=2)
    memory = ["lesson one", "lesson two", "lesson three", "lesson four"]
    compressed = agent._compress_memory(memory)
    assert len(compressed) == 2
    assert compressed[0].startswith("Earlier lessons:")
    assert compressed[-1] == "lesson four"


def test_report_meets_autograde_shape() -> None:
    runtime = LLMRuntime(mode="mock")
    examples = [sample_example(qid=f"report-{index}") for index in range(50)]
    records = [ReActAgent(runtime).run(item) for item in examples]
    records += [ReflexionAgent(runtime).run(item) for item in examples]
    report = build_report(records, "unit.json", mode="mock")
    payload = report.model_dump()
    assert set(payload) == {
        "meta",
        "summary",
        "failure_modes",
        "examples",
        "extensions",
        "discussion",
    }
    assert payload["meta"]["num_records"] == 100
    assert len(payload["examples"]) >= 20
    assert len(payload["failure_modes"]) >= 3
    assert len(payload["discussion"]) >= 250
    assert len(payload["extensions"]) >= 2
