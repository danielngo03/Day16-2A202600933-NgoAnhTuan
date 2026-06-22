from __future__ import annotations

from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field, field_validator


FailureMode = Literal[
    "none",
    "entity_drift",
    "incomplete_multi_hop",
    "wrong_final_answer",
    "looping",
    "reflection_overfit",
]


class ContextChunk(BaseModel):
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)


class QAExample(BaseModel):
    qid: str = Field(min_length=1)
    difficulty: Literal["easy", "medium", "hard"]
    question: str = Field(min_length=1)
    gold_answer: str = Field(min_length=1)
    context: list[ContextChunk] = Field(min_length=1)


class JudgeResult(BaseModel):
    score: Literal[0, 1]
    reason: str = Field(min_length=1)
    failure_mode: FailureMode = "none"
    missing_evidence: list[str] = Field(default_factory=list)
    spurious_claims: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    token_count: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)

    @field_validator("failure_mode")
    @classmethod
    def successful_judgment_has_no_failure(cls, value: FailureMode, info):
        if info.data.get("score") == 1:
            return "none"
        return "wrong_final_answer" if value == "none" else value


class ReflectionEntry(BaseModel):
    attempt_id: int = Field(ge=1)
    failure_reason: str = Field(min_length=1)
    lesson: str = Field(min_length=1)
    next_strategy: str = Field(min_length=1)
    evidence_to_check: list[str] = Field(default_factory=list)
    token_count: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)


class AttemptTrace(BaseModel):
    attempt_id: int
    answer: str
    score: int
    reason: str
    failure_mode: FailureMode = "none"
    reflection: Optional[ReflectionEntry] = None
    token_estimate: int = 0
    latency_ms: int = 0


class RunRecord(BaseModel):
    qid: str
    question: str
    gold_answer: str
    agent_type: Literal["react", "reflexion"]
    predicted_answer: str
    is_correct: bool
    attempts: int
    token_estimate: int
    latency_ms: int
    failure_mode: FailureMode
    reflections: list[ReflectionEntry] = Field(default_factory=list)
    traces: list[AttemptTrace] = Field(default_factory=list)


class ReportPayload(BaseModel):
    meta: dict
    summary: dict
    failure_modes: dict
    examples: list[dict]
    extensions: list[str]
    discussion: str


class ReflexionState(TypedDict):
    question: str
    context: list[str]
    trajectory: list[str]
    reflection_memory: list[str]
    attempt_count: int
    success: bool
    final_answer: str
