from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.api.dependencies import make_engine  # noqa: E402
from app.schemas import BenchmarkCase, InterventionCategory, ReplayRequest  # noqa: E402
from app.services.replay import ReplayService  # noqa: E402


def percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round((len(ordered) - 1) * p))
    return float(ordered[idx])


async def evaluate_case(path: Path) -> dict:
    case = BenchmarkCase.model_validate_json(path.read_text(encoding="utf-8"))
    utterances = [u for u in case.utterances if u.id <= case.evaluation_at_utterance]
    engine = make_engine(case.delegate, meeting_id=f"benchmark-{case.case_id}")
    decisions = await ReplayService(engine).run(ReplayRequest(delegate=case.delegate, utterances=utterances))
    final = decisions[-1]
    expected_intervention = case.expected.category != InterventionCategory.KEEP_SILENCE
    actual_intervention = final.category != InterventionCategory.KEEP_SILENCE
    return {
        "case_id": case.case_id,
        "expected_category": case.expected.category,
        "actual_category": final.category,
        "category_correct": final.category == case.expected.category,
        "expected_intervention": expected_intervention,
        "actual_intervention": actual_intervention,
        "llm_latency_ms": final.llm_latency_ms or 0,
        "response": final.response,
        "manual": {"useful": None, "timing": None, "relevance": None},
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", default=str(ROOT / "benchmarks/fixtures"))
    parser.add_argument("--out", default=str(ROOT / "benchmarks/results/latest.json"))
    args = parser.parse_args()

    fixture_paths = sorted(Path(args.fixtures).glob("*.json"))
    results = [await evaluate_case(path) for path in fixture_paths]
    total = len(results) or 1
    response_count = sum(item["actual_intervention"] for item in results)
    expected_response_count = sum(item["expected_intervention"] for item in results)
    true_positive = sum(item["actual_intervention"] and item["expected_intervention"] for item in results)
    false_positive = sum(item["actual_intervention"] and not item["expected_intervention"] for item in results)
    false_negative = sum(not item["actual_intervention"] and item["expected_intervention"] for item in results)
    latencies = [item["llm_latency_ms"] for item in results if item["llm_latency_ms"]]
    metrics = {
        "case_count": len(results),
        "response_rate": response_count / total,
        "silence_rate": (total - response_count) / total,
        "category_accuracy": sum(item["category_correct"] for item in results) / total,
        "intervention_precision": true_positive / max(1, true_positive + false_positive),
        "intervention_recall": true_positive / max(1, true_positive + false_negative),
        "false_intervention_rate": false_positive / total,
        "average_llm_latency": statistics.mean(latencies) if latencies else 0,
        "p50_latency": percentile(latencies, 0.5),
        "p95_latency": percentile(latencies, 0.95),
        "expected_response_count": expected_response_count,
    }
    payload = {"metrics": metrics, "results": results}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())

