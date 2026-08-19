# Architecture

The core is platform independent.

Flow:

`audio.chunk -> ASR -> utterance.final -> MeetingEngine -> LLMProvider -> intervention.decided`

Main components:

- `MeetingEngine`: owns the progressive meeting state and calls the decider.
- `ContextManager`: builds a prompt from delegate profile, summary, recent utterances and previous interventions.
- `LLMProvider`: provider interface; initial implementation is `LMStudioProvider`.
- `ASRProvider`: provider interface; initial implementation is `FasterWhisperProvider`.
- `Deduplicator`: optional product-like filter for cooldown and repeated suggestions.
- `EventLogger`: writes `meeting.json`, `utterances.jsonl`, `decisions.jsonl`, `metrics.json`.

The replay path sends utterances one at a time and builds a fresh snapshot after each utterance. Future utterances are never included in prompts.

