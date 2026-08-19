#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
MODEL = os.getenv("MODEL", "")
TIMEOUT_SECONDS = float(os.getenv("TIMEOUT_SECONDS", "180"))


def request(method: str, path: str, payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        f"{API_URL}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8")
        raise RuntimeError(f"{method} {path} failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach backend at {API_URL}: {exc.reason}") from exc


def main() -> int:
    print(f"Checking backend: {API_URL}")
    health = request("GET", "/health")
    print(json.dumps(health, indent=2, ensure_ascii=False))
    if health.get("lm_studio") != "ok":
        sys.stdout.flush()
        print("\nLM Studio is not ready. Start the local OpenAI-compatible server and load a model.", file=sys.stderr)
        return 2

    model_query = f"?model={urllib.parse.quote(MODEL)}" if MODEL else ""
    replay_payload = {
        "delegate": {
            "name": "Bob",
            "role": "Backend Engineer",
            "meeting_intents": ["Understand the status of the voice feature"],
            "shareable_information": [
                {
                    "context": "When backend authentication is discussed",
                    "information": "Authentication integration was completed last week.",
                }
            ],
        },
        "utterances": [
            {"id": 1, "speaker": "Alice", "text": "The voice UI is ready for integration testing."},
            {"id": 2, "speaker": "Carol", "text": "Bob, can you confirm whether backend auth is ready?"},
        ],
    }

    print("\nRunning replay smoke scenario...")
    replay = request("POST", f"/replay{model_query}", replay_payload)
    final_decision = replay["decisions"][-1]
    print(json.dumps(final_decision, indent=2, ensure_ascii=False))
    if final_decision["category"] == "KEEP_SILENCE" or not final_decision.get("response"):
        print("\nExpected an explicit cue response, but the model kept silence.", file=sys.stderr)
        return 3

    meeting_id = replay["meeting_id"]
    question = {"question": "O que Bob deveria responder sobre autenticação?"}
    print("\nAsking a meeting-context question...")
    answer = request("POST", f"/meetings/{meeting_id}/questions", question)
    print(json.dumps(answer, indent=2, ensure_ascii=False))
    if not answer.get("answer"):
        print("\nExpected a non-empty answer from the model.", file=sys.stderr)
        return 4

    print("\nSmoke AI passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
