# Future Teams Native Integration

The core should remain unchanged.

Replace `BrowserTabAudioSource` with a future `TeamsNativeMediaSource` that emits the same internal events:

- `audio.chunk`
- `speech.started`
- `speech.ended`
- `transcript.partial`
- `utterance.final`
- `meeting.ended`

Expected final utterance event:

```json
{
  "type": "utterance.final",
  "speaker": "REMOTE",
  "text": "Can backend confirm the integration status?",
  "timestamp": "2026-08-19T12:00:00Z"
}
```

Components that remain intact:

- `MeetingEngine`
- `ContextManager`
- `InterventionDecider` through `LLMProvider`
- `Deduplicator`
- `EventLogger`
- UI decision rendering

The native Teams source would only replace media ingestion and possibly provide stronger speaker metadata.

