from __future__ import annotations

import json
import os
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer


@dataclass(frozen=True)
class ActorResult:
    answer: str
    token_count: int
    latency_ms: int


def _load_env() -> None:
    root = Path(__file__).resolve().parents[2]
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(root / ".env")


class LLMRuntime:
    """Live OpenAI runtime with a deterministic mock mode for autograding."""

    def __init__(
        self,
        mode: Literal["live", "mock"] = "mock",
        model: str | None = None,
        evaluator_model: str | None = None,
    ) -> None:
        _load_env()
        self.mode = mode
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.evaluator_model = evaluator_model or os.getenv(
            "OPENAI_EVALUATOR_MODEL", self.model
        )
        self.client = None
        if mode == "live":
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is required for live mode")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install the openai package for live mode") from exc
            self.client = OpenAI()

    def actor_answer(
        self,
        example: QAExample,
        attempt_id: int,
        agent_type: str,
        reflection_memory: list[str],
    ) -> ActorResult:
        if self.mode == "mock":
            return self._mock_actor(example, attempt_id, agent_type, reflection_memory)

        context = "\n\n".join(
            f"[{index}] {chunk.title}\n{chunk.text}"
            for index, chunk in enumerate(example.context, start=1)
        )
        reflections = "\n".join(f"- {item}" for item in reflection_memory) or "(none)"
        payload = (
            f"Question: {example.question}\n\n"
            f"Context:\n{context}\n\n"
            f"Previous reflections:\n{reflections}\n\n"
            f"Attempt: {attempt_id}"
        )
        data, tokens, latency = self._json_call(
            model=self.model,
            system=ACTOR_SYSTEM,
            user=payload,
        )
        answer = str(data.get("answer", "")).strip()
        if not answer:
            raise ValueError("Actor returned an empty answer")
        return ActorResult(answer=answer, token_count=tokens, latency_ms=latency)

    def evaluator(self, example: QAExample, answer: str) -> JudgeResult:
        if self.mode == "mock":
            return self._mock_evaluator(example, answer)

        payload = (
            f"Question: {example.question}\n"
            f"Gold answer: {example.gold_answer}\n"
            f"Predicted answer: {answer}"
        )
        data, tokens, latency = self._json_call(
            model=self.evaluator_model,
            system=EVALUATOR_SYSTEM,
            user=payload,
        )
        data["token_count"] = tokens
        data["latency_ms"] = latency
        return JudgeResult.model_validate(data)

    def reflector(
        self,
        example: QAExample,
        attempt_id: int,
        answer: str,
        judge: JudgeResult,
    ) -> ReflectionEntry:
        if self.mode == "mock":
            return self._mock_reflector(example, attempt_id, judge)

        context_titles = ", ".join(chunk.title for chunk in example.context)
        payload = (
            f"Question: {example.question}\n"
            f"Previous answer: {answer}\n"
            f"Evaluator reason: {judge.reason}\n"
            f"Missing evidence: {judge.missing_evidence}\n"
            f"Spurious claims: {judge.spurious_claims}\n"
            f"Available context titles: {context_titles}"
        )
        data, tokens, latency = self._json_call(
            model=self.model,
            system=REFLECTOR_SYSTEM,
            user=payload,
        )
        data["attempt_id"] = attempt_id
        data["token_count"] = tokens
        data["latency_ms"] = latency
        return ReflectionEntry.model_validate(data)

    def _json_call(self, model: str, system: str, user: str) -> tuple[dict[str, Any], int, int]:
        if self.client is None:
            raise RuntimeError("Live OpenAI client is not initialized")
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        latency_ms = round((time.perf_counter() - started) * 1000)
        content = response.choices[0].message.content or "{}"
        usage = response.usage
        tokens = int(usage.total_tokens) if usage is not None else 0
        return json.loads(content), tokens, latency_ms

    @staticmethod
    def _mock_actor(
        example: QAExample,
        attempt_id: int,
        agent_type: str,
        reflection_memory: list[str],
    ) -> ActorResult:
        # Stable failures make the full Reflexion loop testable without API cost.
        bucket = int(hashlib.sha256(example.qid.encode("utf-8")).hexdigest()[:8], 16)
        should_fail = bucket % 4 == 0
        if agent_type == "reflexion" and attempt_id > 1 and reflection_memory:
            answer = example.gold_answer
        elif should_fail:
            answer = example.context[0].title
        else:
            answer = example.gold_answer
        tokens = 20 + sum(len(chunk.text) // 4 for chunk in example.context)
        return ActorResult(answer=answer, token_count=tokens, latency_ms=1)

    @staticmethod
    def _mock_evaluator(example: QAExample, answer: str) -> JudgeResult:
        if normalize_answer(example.gold_answer) == normalize_answer(answer):
            return JudgeResult(
                score=1,
                reason="Answer matches the gold answer after normalization.",
                failure_mode="none",
                confidence=1.0,
                token_count=12,
                latency_ms=1,
            )
        modes = ("entity_drift", "incomplete_multi_hop", "wrong_final_answer")
        mode = modes[sum(ord(char) for char in example.qid) % len(modes)]
        return JudgeResult(
            score=0,
            reason="The predicted answer does not complete the required evidence chain.",
            failure_mode=mode,
            missing_evidence=["Verify the final relation in the supporting context."],
            spurious_claims=[answer],
            confidence=0.95,
            token_count=18,
            latency_ms=1,
        )

    @staticmethod
    def _mock_reflector(
        example: QAExample,
        attempt_id: int,
        judge: JudgeResult,
    ) -> ReflectionEntry:
        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="An intermediate entity is not necessarily the requested final answer.",
            next_strategy=(
                "Trace the relation from the first supporting entity to the second, "
                "then verify the final answer against the question wording."
            ),
            evidence_to_check=[chunk.title for chunk in example.context[:2]],
            token_count=24,
            latency_ms=1,
        )


_DEFAULT_RUNTIME: LLMRuntime | None = None


def configure_runtime(mode: Literal["live", "mock"], model: str | None = None) -> LLMRuntime:
    global _DEFAULT_RUNTIME
    _DEFAULT_RUNTIME = LLMRuntime(mode=mode, model=model)
    return _DEFAULT_RUNTIME


def _runtime() -> LLMRuntime:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        _DEFAULT_RUNTIME = LLMRuntime(mode="mock")
    return _DEFAULT_RUNTIME


def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> ActorResult:
    return _runtime().actor_answer(example, attempt_id, agent_type, reflection_memory)


def evaluator(example: QAExample, answer: str) -> JudgeResult:
    return _runtime().evaluator(example, answer)


def reflector(
    example: QAExample,
    attempt_id: int,
    answer: str,
    judge: JudgeResult,
) -> ReflectionEntry:
    return _runtime().reflector(example, attempt_id, answer, judge)
