from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import typer
from rich import print

from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.mock_runtime import LLMRuntime
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.schemas import QAExample, RunRecord
from src.reflexion_lab.utils import load_dataset, save_jsonl


app = typer.Typer(add_completion=False)


def _run_many(agent, examples: list[QAExample], workers: int) -> list[RunRecord]:
    if workers <= 1:
        return [agent.run(example) for example in examples]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(agent.run, examples))


@app.command()
def main(
    dataset: str = "data/hotpot_qaexamples.json",
    out_dir: str = "outputs/main_run",
    mode: Literal["live", "mock"] = "mock",
    model: str = "gpt-4o-mini",
    limit: int = 100,
    reflexion_attempts: int = 3,
    workers: int = 6,
) -> None:
    """Run ReAct and Reflexion on the same examples and save full artifacts."""

    examples = load_dataset(dataset, limit=limit)
    runtime = LLMRuntime(mode=mode, model=model)
    react = ReActAgent(runtime=runtime)
    reflexion = ReflexionAgent(
        runtime=runtime,
        max_attempts=reflexion_attempts,
        memory_limit=3,
    )

    print(
        f"Running {len(examples)} examples per agent in [bold]{mode}[/bold] mode "
        f"with {workers} worker(s)..."
    )
    react_records = _run_many(react, examples, workers)
    reflexion_records = _run_many(reflexion, examples, workers)
    all_records = react_records + reflexion_records

    output = Path(out_dir)
    save_jsonl(output / "react_runs.jsonl", react_records)
    save_jsonl(output / "reflexion_runs.jsonl", reflexion_records)
    report = build_report(
        all_records,
        dataset_name=Path(dataset).name,
        mode=mode,
        model=model if mode == "live" else "deterministic",
    )
    json_path, markdown_path = save_report(report, output)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {markdown_path}")
    print(json.dumps(report.summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
