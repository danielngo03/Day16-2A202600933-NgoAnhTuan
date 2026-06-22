from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .mock_runtime import LLMRuntime
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord


@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    runtime: LLMRuntime
    max_attempts: int = 1
    memory_limit: int = 3
    adaptive_attempts: bool = True

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        final_failure_mode = "wrong_final_answer"

        allowed_attempts = self._attempt_budget(example)
        for attempt_id in range(1, allowed_attempts + 1):
            actor = self.runtime.actor_answer(
                example,
                attempt_id,
                self.agent_type,
                reflection_memory,
            )
            judge = self.runtime.evaluator(example, actor.answer)
            trace_tokens = actor.token_count + judge.token_count
            trace_latency = actor.latency_ms + judge.latency_ms
            reflection = None

            final_answer = actor.answer
            final_score = judge.score
            final_failure_mode = judge.failure_mode

            if (
                judge.score == 0
                and self.agent_type == "reflexion"
                and attempt_id < allowed_attempts
            ):
                reflection = self.runtime.reflector(
                    example,
                    attempt_id,
                    actor.answer,
                    judge,
                )
                reflections.append(reflection)
                reflection_memory.append(self._memory_text(reflection))
                reflection_memory = self._compress_memory(reflection_memory)
                trace_tokens += reflection.token_count
                trace_latency += reflection.latency_ms

            traces.append(
                AttemptTrace(
                    attempt_id=attempt_id,
                    answer=actor.answer,
                    score=judge.score,
                    reason=judge.reason,
                    failure_mode=judge.failure_mode,
                    reflection=reflection,
                    token_estimate=trace_tokens,
                    latency_ms=trace_latency,
                )
            )
            if judge.score == 1:
                final_failure_mode = "none"
                break

        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=sum(trace.token_estimate for trace in traces),
            latency_ms=sum(trace.latency_ms for trace in traces),
            failure_mode=final_failure_mode,
            reflections=reflections,
            traces=traces,
        )

    def _attempt_budget(self, example: QAExample) -> int:
        if self.agent_type == "react":
            return 1
        if not self.adaptive_attempts:
            return self.max_attempts
        difficulty_budget = {"easy": 2, "medium": 3, "hard": self.max_attempts}
        return min(self.max_attempts, difficulty_budget[example.difficulty])

    @staticmethod
    def _memory_text(reflection: ReflectionEntry) -> str:
        evidence = ", ".join(reflection.evidence_to_check) or "unspecified evidence"
        return (
            f"Lesson: {reflection.lesson} Strategy: {reflection.next_strategy} "
            f"Check: {evidence}"
        )

    def _compress_memory(self, memory: list[str]) -> list[str]:
        if len(memory) <= self.memory_limit:
            return memory
        older = " | ".join(memory[: -self.memory_limit + 1])
        compressed = f"Earlier lessons: {older[:800]}"
        return [compressed, *memory[-self.memory_limit + 1 :]]


class ReActAgent(BaseAgent):
    def __init__(self, runtime: LLMRuntime) -> None:
        super().__init__(agent_type="react", runtime=runtime, max_attempts=1)


class ReflexionAgent(BaseAgent):
    def __init__(
        self,
        runtime: LLMRuntime,
        max_attempts: int = 3,
        memory_limit: int = 3,
    ) -> None:
        super().__init__(
            agent_type="reflexion",
            runtime=runtime,
            max_attempts=max_attempts,
            memory_limit=memory_limit,
        )
