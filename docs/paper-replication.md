# Paper Replication Notes

Hu et al. construct Meeting Delegate benchmark cases from transcript snapshots. A model sees only the transcript up to a current utterance and must decide whether a delegate should respond.

This POC mirrors that setup:

- Replay mode feeds utterances progressively.
- Prompt `intervention_v1` receives delegate intent, shareable information, previous interventions and transcript up to the current moment.
- Decisions use four categories: `EXPLICIT_CUE`, `IMPLICIT_CUE`, `CHIME_IN`, `KEEP_SILENCE`.
- Temperature is fixed at `0`.
- Logs store prompt version, model, category, confidence, response, reason and latencies.

The paper highlights response/silence tradeoffs and issues with irrelevant or repetitive responses. The POC therefore includes an optional deduplication/cooldown filter that can be disabled for raw benchmark behavior.

