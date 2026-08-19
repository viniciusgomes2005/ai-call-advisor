# Google Meet Testing

The MVP does not use the Google Meet API.

1. Open Google Meet in Chrome.
2. Join a test meeting.
3. Open the frontend at `http://localhost:5173`.
4. Click `Start live meeting`.
5. Select the Meet tab specifically.
6. Enable tab audio sharing.
7. Allow microphone access.

The browser sends:

- `REMOTE_AUDIO`: audio captured from the Meet tab.
- `LOCAL_MIC_AUDIO`: local microphone.

The backend can label these as `REMOTE` and `ME`. It does not attempt full diarization.

