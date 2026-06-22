# Lab 16 Benchmark Report

## Metadata

- Dataset: `hotpot_qaexamples.json`
- Mode: `mock`
- Model: `deterministic`
- Examples: 100
- Records: 200
- Token accounting: deterministic mock accounting

## Summary

| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.76 | 1.0 | 0.24 |
| Avg attempts | 1 | 1.24 | 0.24 |
| Avg tokens | 1440.43 | 1814.23 | 373.8 |
| Avg latency (ms) | 2 | 2.72 | 0.72 |

## Failure Modes

```json
{
  "entity_drift": {
    "counts": {
      "react": 10,
      "react_failed_attempts": 10,
      "reflexion_failed_attempts": 10
    },
    "diagnosis": "The reasoning chain moved to a related but unsupported entity.",
    "suggested_fix": "Track entity names per hop and verify the final entity in context."
  },
  "incomplete_multi_hop": {
    "counts": {
      "react": 7,
      "react_failed_attempts": 7,
      "reflexion_failed_attempts": 7
    },
    "diagnosis": "The answer stopped after an intermediate relation.",
    "suggested_fix": "Write the complete hop chain before selecting the final answer."
  },
  "wrong_final_answer": {
    "counts": {
      "react": 7,
      "react_failed_attempts": 7,
      "reflexion_failed_attempts": 7
    },
    "diagnosis": "The evidence chain was attempted but the final phrase was incorrect.",
    "suggested_fix": "Re-read the question target and compare it with the final evidence."
  },
  "looping": {
    "counts": {},
    "diagnosis": "Retries repeated the same unsuccessful reasoning path.",
    "suggested_fix": "Use reflection memory to force a different evidence order."
  },
  "reflection_overfit": {
    "counts": {},
    "diagnosis": "A reflection introduced assumptions not supported by context.",
    "suggested_fix": "Treat reflection as strategy only and re-ground every fact."
  }
}
```

## Extensions

- `structured_evaluator`
- `reflection_memory`
- `adaptive_max_attempts`
- `memory_compression`
- `benchmark_report_json`
- `mock_mode_for_autograding`

## Discussion

The benchmark compares ReAct and Reflexion on the same multi-hop QA examples. ReAct has the lower cost because it always stops after one Actor-Evaluator pass. Reflexion spends additional tokens and latency only after a failed judgment, then carries a structured lesson and next-step strategy into the retry. This is useful for incomplete hops and entity drift, but it cannot repair a weak evaluator or missing context. Adaptive attempt budgets limit unnecessary retries, while compressed reflection memory prevents old lessons from growing without bound. Token totals come from OpenAI response usage in live mode and latency is measured wall-clock per API call. The system also retains a deterministic mock mode so graders can verify control flow without API access. Reflection therefore improves recoverability, not correctness for free: every gain must be read together with attempts, tokens, and latency.
