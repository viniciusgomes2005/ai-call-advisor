import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Mic, MonitorUp, PauseCircle, Play, RotateCcw, Send, Upload } from "lucide-react";
import "./styles.css";

type Category = "EXPLICIT_CUE" | "IMPLICIT_CUE" | "CHIME_IN" | "KEEP_SILENCE";

type ShareableInfo = { context: string; information: string };
type Delegate = {
  name: string;
  role: string;
  meeting_intents: string[];
  shareable_information: ShareableInfo[];
};
type Utterance = { id: number; speaker: string; text: string; timestamp?: string; source?: string };
type Decision = {
  utterance_id: number;
  category: Category;
  should_intervene: boolean;
  confidence: number;
  response: string | null;
  reason: string;
  llm_latency_ms?: number;
  pipeline_latency_ms?: number;
  stale?: boolean;
  displayed?: boolean;
  filtered?: boolean;
};
type Health = { backend: "ok"; lm_studio: "ok" | "error"; model?: string; asr: string; error?: string };

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const WS = API.replace(/^http/, "ws");

const defaultDelegate: Delegate = {
  name: "Bob",
  role: "Backend Engineer",
  meeting_intents: ["Understand the status of the voice feature"],
  shareable_information: [
    {
      context: "When backend integration is discussed",
      information: "The authentication integration was completed last week",
    },
  ],
};

const defaultReplay = JSON.stringify(
  {
    delegate: defaultDelegate,
    utterances: [
      { id: 1, speaker: "Alice", text: "The voice UI is ready for integration testing." },
      { id: 2, speaker: "Carol", text: "There are still latency spikes when people talk over each other." },
      { id: 3, speaker: "Alice", text: "Bob, what does backend think about this?" },
    ],
  },
  null,
  2,
);

function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [delegate, setDelegate] = useState<Delegate>(defaultDelegate);
  const [meetingId, setMeetingId] = useState("");
  const [mode, setMode] = useState<"REPLAY" | "LIVE">("REPLAY");
  const [utterances, setUtterances] = useState<Utterance[]>([]);
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const [showSilence, setShowSilence] = useState(true);
  const [replayJson, setReplayJson] = useState(defaultReplay);
  const [manualSpeaker, setManualSpeaker] = useState("ME");
  const [manualText, setManualText] = useState("");
  const [debug, setDebug] = useState<Record<string, string | number>>({});
  const wsRef = useRef<WebSocket | null>(null);
  const recorders = useRef<MediaRecorder[]>([]);
  const flushTimers = useRef<number[]>([]);

  const lastIntervention = useMemo(
    () => [...decisions].reverse().find((item) => item.should_intervene && item.response),
    [decisions],
  );

  async function refreshStatus() {
    const [healthRes, modelsRes] = await Promise.all([
      fetch(`${API}/health`).then((r) => r.json()),
      fetch(`${API}/models`).then((r) => (r.ok ? r.json() : { models: [] })),
    ]);
    setHealth(healthRes);
    const ids = (modelsRes.models ?? []).map((item: { id: string }) => item.id);
    setModels(ids);
    setSelectedModel((current) => current || healthRes.model || ids[0] || "");
  }

  useEffect(() => {
    refreshStatus().catch(console.error);
  }, []);

  function reset() {
    stopLive();
    setMeetingId("");
    setUtterances([]);
    setDecisions([]);
    setDebug({});
  }

  function appendDecision(decision: Decision) {
    setDecisions((prev) => [...prev, decision]);
    setDebug((prev) => ({
      ...prev,
      "LLM latency": decision.llm_latency_ms ?? "-",
      "Pipeline latency": decision.pipeline_latency_ms ?? "-",
      "Last intervention": decision.response ?? "KEEP_SILENCE",
      "LLM queue size": 0,
    }));
  }

  async function startReplay() {
    setMode("REPLAY");
    setUtterances([]);
    setDecisions([]);
    const payload = JSON.parse(replayJson);
    const res = await fetch(`${API}/replay?model=${encodeURIComponent(selectedModel)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    setMeetingId(data.meeting_id);
    setUtterances(payload.utterances);
    setDecisions(data.decisions);
  }

  async function createMeetingIfNeeded(): Promise<string> {
    if (meetingId) return meetingId;
    const res = await fetch(`${API}/meetings?model=${encodeURIComponent(selectedModel)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ delegate }),
    });
    const data = await res.json();
    setMeetingId(data.meeting_id);
    return data.meeting_id;
  }

  async function sendManual() {
    const id = await createMeetingIfNeeded();
    if (!manualText.trim()) return;
    const res = await fetch(`${API}/meetings/${id}/utterances`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ speaker: manualSpeaker, text: manualText, source: "MANUAL" }),
    });
    const data = await res.json();
    setUtterances((prev) => [...prev, data.utterance]);
    appendDecision(data.decision);
    setManualText("");
  }

  async function uploadAudio(file: File) {
    const id = await createMeetingIfNeeded();
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${API}/meetings/${id}/audio?speaker=UNKNOWN`, { method: "POST", body: form });
    const data = await res.json();
    for (const item of data.segments ?? []) {
      setUtterances((prev) => [...prev, item.utterance]);
      appendDecision(item.decision);
    }
  }

  async function startLive() {
    setMode("LIVE");
    setUtterances([]);
    setDecisions([]);
    const socket = new WebSocket(`${WS}/ws/live`);
    wsRef.current = socket;
    socket.onopen = () => {
      socket.send(JSON.stringify({ type: "meeting.start", delegate, model: selectedModel || undefined }));
    };
    socket.onmessage = (message) => {
      const event = JSON.parse(message.data);
      if (event.type === "meeting.state.updated" && event.payload.meeting_id) {
        setMeetingId(event.payload.meeting_id);
      }
      if (event.type === "utterance.final") {
        setUtterances((prev) => [...prev, event.payload]);
      }
      if (event.type === "intervention.decided") {
        appendDecision(event.payload);
      }
      if (event.type === "audio.chunk") {
        setDebug((prev) => ({ ...prev, "Last audio chunk": `${event.payload.source} ${event.payload.bytes} bytes` }));
      }
      if (event.type === "transcript.error") {
        setDebug((prev) => ({ ...prev, "ASR error": event.payload.error }));
      }
    };
    await new Promise((resolve) => setTimeout(resolve, 300));
    await captureAudio(socket);
  }

  async function captureAudio(socket: WebSocket) {
    alert(
      "Selecione especificamente a aba do Google Meet ou Teams no Chrome e habilite o compartilhamento de áudio da aba. Em seguida autorize o microfone local.",
    );
    const display = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
    const mic = await navigator.mediaDevices.getUserMedia({ audio: true });
    const remoteAudio = new MediaStream(display.getAudioTracks());
    const micAudio = new MediaStream(mic.getAudioTracks());
    startRecorder(socket, remoteAudio, "REMOTE_AUDIO");
    startRecorder(socket, micAudio, "LOCAL_MIC_AUDIO");
    setDebug((prev) => ({ ...prev, "Audio buffer": "capturing", "Current model": selectedModel || "-" }));
  }

  function startRecorder(socket: WebSocket, stream: MediaStream, source: "REMOTE_AUDIO" | "LOCAL_MIC_AUDIO") {
    if (stream.getAudioTracks().length === 0) {
      setDebug((prev) => ({ ...prev, [`${source} status`]: "no track" }));
      return;
    }
    const recorder = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });
    recorder.ondataavailable = async (event) => {
      if (!event.data.size || socket.readyState !== WebSocket.OPEN) return;
      const data = await blobToBase64(event.data);
      socket.send(JSON.stringify({ type: "audio.chunk", source, data }));
    };
    recorder.start(1000);
    recorders.current.push(recorder);
    const timer = window.setInterval(() => {
      if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "audio.flush", source }));
    }, 8000);
    flushTimers.current.push(timer);
  }

  function stopLive() {
    for (const recorder of recorders.current) {
      if (recorder.state !== "inactive") recorder.stop();
      for (const track of recorder.stream.getTracks()) track.stop();
    }
    for (const timer of flushTimers.current) window.clearInterval(timer);
    recorders.current = [];
    flushTimers.current = [];
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "meeting.stop" }));
      wsRef.current.close();
    }
    wsRef.current = null;
    setDebug((prev) => ({ ...prev, "Audio buffer": "stopped" }));
  }

  return (
    <main>
      <section className="panel config">
        <div className="titleRow">
          <h1>Meeting Delegate POC</h1>
          <button className="iconButton" onClick={refreshStatus} title="Refresh status">
            <RotateCcw size={18} />
          </button>
        </div>

        <div className="statusGrid">
          <Status label="LM Studio" value={health?.lm_studio ?? "unknown"} ok={health?.lm_studio === "ok"} />
          <Status label="ASR" value={health?.asr ?? "unknown"} ok={health?.asr === "ok"} />
        </div>

        <label>
          Model
          <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
            <option value="">Select loaded model</option>
            {models.map((model) => (
              <option key={model}>{model}</option>
            ))}
          </select>
        </label>

        <div className="modeSwitch">
          <button className={mode === "REPLAY" ? "selected" : ""} onClick={() => setMode("REPLAY")}>
            REPLAY
          </button>
          <button className={mode === "LIVE" ? "selected" : ""} onClick={() => setMode("LIVE")}>
            LIVE
          </button>
        </div>

        <label>
          Delegate name
          <input value={delegate.name} onChange={(e) => setDelegate({ ...delegate, name: e.target.value })} />
        </label>
        <label>
          Role
          <input value={delegate.role} onChange={(e) => setDelegate({ ...delegate, role: e.target.value })} />
        </label>
        <label>
          Meeting intents
          <textarea
            value={delegate.meeting_intents.join("\n")}
            onChange={(e) => setDelegate({ ...delegate, meeting_intents: e.target.value.split("\n").filter(Boolean) })}
          />
        </label>
        <label>
          Shareable information
          <textarea
            value={delegate.shareable_information.map((item) => `${item.context} => ${item.information}`).join("\n")}
            onChange={(e) =>
              setDelegate({
                ...delegate,
                shareable_information: e.target.value
                  .split("\n")
                  .filter(Boolean)
                  .map((line) => {
                    const [context, information] = line.split("=>");
                    return { context: context?.trim() ?? "", information: information?.trim() ?? "" };
                  }),
              })
            }
          />
        </label>

        <div className="actions">
          <button onClick={startReplay}>
            <Play size={16} /> Start replay
          </button>
          <button onClick={startLive}>
            <MonitorUp size={16} /> Start live meeting
          </button>
          <button onClick={stopLive}>
            <PauseCircle size={16} /> Stop
          </button>
          <button onClick={reset}>
            <RotateCcw size={16} /> Reset
          </button>
        </div>
      </section>

      <section className="panel transcript">
        <h2>Transcription</h2>
        <div className="manualRow">
          <input value={manualSpeaker} onChange={(e) => setManualSpeaker(e.target.value)} aria-label="Speaker" />
          <input
            value={manualText}
            onChange={(e) => setManualText(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendManual()}
            placeholder="Send utterance"
          />
          <button className="iconButton" onClick={sendManual} title="Send utterance">
            <Send size={18} />
          </button>
        </div>

        <label className="upload">
          <Upload size={18} />
          Upload WAV/MP3/M4A
          <input type="file" accept="audio/*" onChange={(e) => e.target.files?.[0] && uploadAudio(e.target.files[0])} />
        </label>

        {mode === "REPLAY" && (
          <textarea className="replayBox" value={replayJson} onChange={(e) => setReplayJson(e.target.value)} />
        )}

        <div className="utteranceList">
          {utterances.map((item) => (
            <article key={`${item.id}-${item.text}`} className="utterance">
              <div className="speaker">
                <span>{item.id.toString().padStart(2, "0")}</span>
                <strong>{item.speaker}</strong>
                <small>{item.source ?? ""}</small>
              </div>
              <p>{item.text}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="panel decisions">
        <div className="titleRow">
          <h2>Decisions</h2>
          <label className="check">
            <input type="checkbox" checked={showSilence} onChange={(e) => setShowSilence(e.target.checked)} />
            Show silence
          </label>
        </div>

        <div className="debugGrid">
          <Status label="Utterances" value={utterances.length.toString()} ok />
          <Status label="Prompt" value="intervention_v1" ok />
          <Status label="Model" value={selectedModel || "-"} ok={Boolean(selectedModel)} />
          <Status label="Last intervention" value={lastIntervention?.category ?? "-"} ok={Boolean(lastIntervention)} />
          {Object.entries(debug).map(([key, value]) => (
            <Status key={key} label={key} value={String(value)} ok />
          ))}
        </div>

        <div className="decisionList">
          {decisions
            .filter((item) => showSilence || item.category !== "KEEP_SILENCE")
            .map((item, index) => (
              <article
                key={`${item.utterance_id}-${index}`}
                className={`decision ${item.category === "KEEP_SILENCE" ? "silence" : "intervention"}`}
              >
                <div className="decisionHeader">
                  <strong>{item.category}</strong>
                  <span>{item.confidence.toFixed(2)}</span>
                </div>
                {item.response && <blockquote>{item.response}</blockquote>}
                <p>{item.reason}</p>
                <small>
                  u{item.utterance_id} · LLM {item.llm_latency_ms ?? "-"}ms · pipeline{" "}
                  {item.pipeline_latency_ms ?? "-"}ms {item.filtered ? "· filtered" : ""}
                </small>
              </article>
            ))}
        </div>
      </section>
    </main>
  );
}

function Status({ label, value, ok }: { label: string; value: string; ok: boolean }) {
  return (
    <div className="status">
      <span className={ok ? "dot ok" : "dot"} />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",")[1] ?? "");
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
