from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.asr.provider import SAP_TERMS, FasterWhisperProvider  # noqa: E402


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[\w/]+", text.lower())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, ref_word in enumerate(reference, start=1):
        current = [row]
        for col, hyp_word in enumerate(hypothesis, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current.append(min(previous[col] + 1, current[col - 1] + 1, previous[col - 1] + cost))
        previous = current
    return previous[-1]


def wer(reference: str, hypothesis: str) -> float:
    reference_words = normalize_words(reference)
    if not reference_words:
        return 0.0
    return edit_distance(reference_words, normalize_words(hypothesis)) / len(reference_words)


def sap_term_recall(reference: str, hypothesis: str) -> float | None:
    expected = [term for term in SAP_TERMS if term.lower() in reference.lower()]
    if not expected:
        return None
    matched = [term for term in expected if term.lower() in hypothesis.lower()]
    return len(matched) / len(expected)


async def evaluate_case(provider: FasterWhisperProvider, audio_path: Path, reference: str | None) -> dict:
    result = await provider.transcribe_file(audio_path)
    transcript = " ".join(segment.text for segment in result.segments)
    payload = {
        "audio": str(audio_path),
        "language": result.language,
        "audio_duration_seconds": result.audio_duration_seconds,
        "processing_time_ms": result.processing_time_ms,
        "real_time_factor": result.real_time_factor,
        "transcript": transcript,
    }
    if reference is not None:
        payload["wer"] = wer(reference, transcript)
        payload["sap_term_recall"] = sap_term_recall(reference, transcript)
    return payload


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", nargs="+", required=True)
    parser.add_argument("--reference-json", help="JSON mapping audio filename to reference transcript")
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="auto")
    parser.add_argument("--domain-prompt", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "benchmarks/results/asr_latest.json"))
    args = parser.parse_args()

    references = {}
    if args.reference_json:
        references = json.loads(Path(args.reference_json).read_text(encoding="utf-8"))

    provider = FasterWhisperProvider(
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        use_domain_prompt=args.domain_prompt,
    )
    await provider.load()
    results = []
    for audio in args.audio:
        audio_path = Path(audio)
        reference = references.get(audio_path.name) or references.get(str(audio_path))
        results.append(await evaluate_case(provider, audio_path, reference))

    latencies = [item["processing_time_ms"] for item in results]
    rtfs = [item["real_time_factor"] for item in results if item["real_time_factor"] is not None]
    metrics = {
        "model": args.model,
        "device": provider.runtime.device,
        "compute_type": provider.runtime.compute_type,
        "case_count": len(results),
        "average_processing_time_ms": statistics.mean(latencies) if latencies else 0,
        "average_real_time_factor": statistics.mean(rtfs) if rtfs else None,
        "average_wer": statistics.mean(item["wer"] for item in results if "wer" in item)
        if any("wer" in item for item in results)
        else None,
        "average_sap_term_recall": statistics.mean(
            item["sap_term_recall"] for item in results if item.get("sap_term_recall") is not None
        )
        if any(item.get("sap_term_recall") is not None for item in results)
        else None,
    }
    payload = {"metrics": metrics, "results": results}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
