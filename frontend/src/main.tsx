import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { MonitorUp, PauseCircle, PictureInPicture2, Play, RotateCcw, Send, Upload } from "lucide-react";
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
type ChatMessage = {
  id: string;
  role: "assistant" | "user" | "meeting" | "insight";
  text: string;
  timestamp: string;
  kind?: string;
};
type MeetingInsight = {
  id: string;
  type: "OPEN_QUESTION" | "ACTION_ITEM" | "DECISION";
  utterance_id: number;
  speaker: string;
  text: string;
  reason: string;
  confidence: number;
};
type DocumentPictureInPictureController = {
  requestWindow: (options?: { width?: number; height?: number; disallowReturnToOpener?: boolean }) => Promise<Window>;
  window?: Window | null;
};

declare global {
  interface Window {
    documentPictureInPicture?: DocumentPictureInPictureController;
  }
}

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
  const [liveActive, setLiveActive] = useState(false);
  const [floatingSupported] = useState(() => Boolean(window.documentPictureInPicture));
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: crypto.randomUUID(),
      role: "assistant",
      text: "Abra uma reunião live ou rode um replay para eu acompanhar o contexto. Você pode perguntar sobre o que já foi dito.",
      timestamp: new Date().toISOString(),
    },
  ]);
  const [askingQuestion, setAskingQuestion] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const pipWindowRef = useRef<Window | null>(null);
  const pipRootRef = useRef<ReturnType<typeof createRoot> | null>(null);
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
    setChatMessages([
      {
        id: crypto.randomUUID(),
        role: "assistant",
        text: "Contexto limpo. Abra uma reunião live ou rode um replay para começar de novo.",
        timestamp: new Date().toISOString(),
      },
    ]);
  }

  function addChatMessage(role: ChatMessage["role"], text: string) {
    setChatMessages((prev) => [
      ...prev.slice(-49),
      {
        id: crypto.randomUUID(),
        role,
        text,
        timestamp: new Date().toISOString(),
      },
    ]);
  }

  function addUtteranceUpdate(utterance: Utterance) {
    addChatMessage("meeting", `${utterance.speaker}: ${utterance.text}`);
  }

  function addInsightUpdate(insight: MeetingInsight) {
    const label =
      insight.type === "OPEN_QUESTION"
        ? "Open question"
        : insight.type === "ACTION_ITEM"
          ? "Action item"
          : "Decision";
    setChatMessages((prev) => [
      ...prev.slice(-49),
      {
        id: insight.id,
        role: "insight",
        kind: insight.type,
        text: `${label}: ${insight.text}`,
        timestamp: new Date().toISOString(),
      },
    ]);
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
    if (decision.response) {
      addChatMessage("assistant", decision.response);
    }
  }

  async function startReplay() {
    setMode("REPLAY");
    setUtterances([]);
    setDecisions([]);
    setChatMessages([]);
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
    for (const utterance of payload.utterances as Utterance[]) {
      addUtteranceUpdate(utterance);
    }
    for (const decision of data.decisions as Decision[]) {
      if (decision.response) addChatMessage("assistant", decision.response);
    }
    for (const insight of data.insights ?? []) {
      addInsightUpdate(insight);
    }
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
    addUtteranceUpdate(data.utterance);
    for (const insight of data.insights ?? []) {
      addInsightUpdate(insight);
    }
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
      addUtteranceUpdate(item.utterance);
      for (const insight of item.insights ?? []) {
        addInsightUpdate(insight);
      }
      appendDecision(item.decision);
    }
  }

  async function startLive() {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    setMode("LIVE");
    setUtterances([]);
    setDecisions([]);
    setChatMessages([
      {
        id: crypto.randomUUID(),
        role: "assistant",
        text: "Live iniciada. Vou destacar pontos relevantes e responder perguntas usando o contexto capturado.",
        timestamp: new Date().toISOString(),
      },
    ]);
    const socket = new WebSocket(`${WS}/ws/live`);
    wsRef.current = socket;
    socket.onclose = () => {
      wsRef.current = null;
      setLiveActive(false);
    };
    socket.onopen = () => {
      setLiveActive(true);
      socket.send(JSON.stringify({ type: "meeting.start", delegate, model: selectedModel || undefined }));
    };
    socket.onmessage = (message) => {
      const event = JSON.parse(message.data);
      if (event.type === "meeting.state.updated" && event.payload.meeting_id) {
        setMeetingId(event.payload.meeting_id);
      }
      if (event.type === "utterance.final") {
        setUtterances((prev) => [...prev, event.payload]);
        addUtteranceUpdate(event.payload);
      }
      if (event.type === "intervention.decided") {
        appendDecision(event.payload);
      }
      if (event.type === "meeting.insight.detected") {
        addInsightUpdate(event.payload);
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
    setLiveActive(false);
    setDebug((prev) => ({ ...prev, "Audio buffer": "stopped" }));
  }

  async function askMeetingQuestion(question: string) {
    const cleanQuestion = question.trim();
    if (!cleanQuestion) return;
    addChatMessage("user", cleanQuestion);
    if (!meetingId) {
      addChatMessage("assistant", "Ainda não existe uma reunião ativa. Inicie live, replay ou envie uma utterance manual primeiro.");
      return;
    }
    setAskingQuestion(true);
    try {
      const res = await fetch(`${API}/meetings/${meetingId}/questions?model=${encodeURIComponent(selectedModel)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: cleanQuestion }),
      });
      if (!res.ok) {
        throw new Error(await res.text());
      }
      const data = await res.json();
      addChatMessage("assistant", data.answer);
      setDebug((prev) => ({ ...prev, "Last question latency": data.llm_latency_ms ?? "-" }));
    } catch (error) {
      addChatMessage("assistant", `Não consegui responder agora: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setAskingQuestion(false);
    }
  }

  async function openFloatingWidget() {
    const controller = window.documentPictureInPicture;
    if (!controller) {
      alert("Floating widget requires a Chromium browser with Document Picture-in-Picture support.");
      return;
    }
    if (!pipWindowRef.current || pipWindowRef.current.closed) {
      const pipWindow = await controller.requestWindow({ width: 360, height: 500 });
      pipWindowRef.current = pipWindow;
      pipWindow.document.title = "Meeting Delegate";
      pipWindow.document.body.innerHTML = '<div id="floating-root"></div>';
      const style = pipWindow.document.createElement("style");
      style.textContent = floatingWidgetCss;
      pipWindow.document.head.append(style);
      pipWindow.addEventListener("pagehide", () => {
        pipRootRef.current?.unmount();
        pipRootRef.current = null;
        pipWindowRef.current = null;
      });
      pipRootRef.current = createRoot(pipWindow.document.getElementById("floating-root")!);
    }
    renderFloatingWidget();
  }

  function renderFloatingWidget() {
    if (!pipWindowRef.current || pipWindowRef.current.closed || !pipRootRef.current) return;
    pipRootRef.current.render(
      <FloatingWidget
        health={health}
        selectedModel={selectedModel}
        utteranceCount={utterances.length}
        liveActive={liveActive}
        latestUtterances={utterances.slice(-3)}
        chatMessages={chatMessages}
        askingQuestion={askingQuestion}
        onStartLive={startLive}
        onStopLive={stopLive}
        onAskQuestion={askMeetingQuestion}
      />,
    );
  }

  useEffect(() => {
    renderFloatingWidget();
  });

  useEffect(() => {
    return () => {
      pipRootRef.current?.unmount();
      pipWindowRef.current?.close();
    };
  }, []);

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
          <button
            onClick={openFloatingWidget}
            title={floatingSupported ? "Open floating widget" : "Requires Document Picture-in-Picture support"}
          >
            <PictureInPicture2 size={16} /> Floating widget
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

function FloatingWidget({
  health,
  selectedModel,
  utteranceCount,
  liveActive,
  latestUtterances,
  chatMessages,
  askingQuestion,
  onStartLive,
  onStopLive,
  onAskQuestion,
}: {
  health: Health | null;
  selectedModel: string;
  utteranceCount: number;
  liveActive: boolean;
  latestUtterances: Utterance[];
  chatMessages: ChatMessage[];
  askingQuestion: boolean;
  onStartLive: () => void;
  onStopLive: () => void;
  onAskQuestion: (question: string) => void;
}) {
  const [question, setQuestion] = useState("");

  function submitQuestion(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!question.trim() || askingQuestion) return;
    onAskQuestion(question);
    setQuestion("");
  }

  return (
    <div className="floatingWidget">
      <header>
        <div>
          <strong>Meeting Delegate</strong>
          <span>{liveActive ? "Live meeting" : "Standby"}</span>
        </div>
        <span className={liveActive ? "pill live" : "pill"}>{liveActive ? "LIVE" : "OFF"}</span>
      </header>

      <div className="miniStatusGrid">
        <MiniStatus label="LLM" value={health?.lm_studio ?? "unknown"} ok={health?.lm_studio === "ok"} />
        <MiniStatus label="ASR" value={health?.asr ?? "unknown"} ok={health?.asr === "ok"} />
        <MiniStatus label="Model" value={selectedModel || "-"} ok={Boolean(selectedModel)} />
        <MiniStatus label="Utterances" value={String(utteranceCount)} ok />
      </div>

      <section className="chatFeed" aria-label="Meeting chat">
        <span>Meeting chat</span>
        <div className="messages">
          {chatMessages.map((message) => (
            <article key={message.id} className={`message ${message.role} ${message.kind ?? ""}`}>
              <p>{message.text}</p>
            </article>
          ))}
          {askingQuestion && (
            <article className="message assistant">
              <p>Thinking...</p>
            </article>
          )}
        </div>
      </section>

      <section className="miniTranscript">
        <span>Recent transcript</span>
        {latestUtterances.length ? (
          latestUtterances.map((item) => (
            <article key={`${item.id}-${item.text}`}>
              <strong>{item.speaker}</strong>
              <p>{item.text}</p>
            </article>
          ))
        ) : (
          <p className="muted">No utterances captured.</p>
        )}
      </section>

      <footer>
        <div className="liveControls">
          <button onClick={onStartLive} disabled={liveActive}>
            <MonitorUp size={16} /> Start
          </button>
          <button onClick={onStopLive} disabled={!liveActive}>
            <PauseCircle size={16} /> Stop
          </button>
        </div>
        <form onSubmit={submitQuestion}>
          <input
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask about the meeting"
            aria-label="Ask about the meeting"
          />
          <button type="submit" disabled={askingQuestion || !question.trim()}>
            <Send size={16} />
          </button>
        </form>
      </footer>
    </div>
  );
}

function MiniStatus({ label, value, ok }: { label: string; value: string; ok: boolean }) {
  return (
    <div className="miniStatus">
      <span className={ok ? "miniDot ok" : "miniDot"} />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

const floatingWidgetCss = `
  :root {
    color: #1f2933;
    background: #f7f9fb;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }

  * { box-sizing: border-box; }

  body {
    margin: 0;
    min-width: 300px;
    background: #f7f9fb;
  }

  button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 7px;
    min-height: 36px;
    border: 1px solid #b8c3d0;
    border-radius: 6px;
    background: #ffffff;
    color: #1f2933;
    padding: 8px 11px;
    font: inherit;
    font-weight: 700;
    cursor: pointer;
  }

  button:disabled {
    cursor: not-allowed;
    opacity: 0.55;
  }

  .floatingWidget {
    display: grid;
    gap: 12px;
    min-height: 100vh;
    padding: 14px;
  }

  header,
  .liveControls,
  form {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }

  footer {
    display: grid;
    gap: 8px;
  }

  header div {
    display: grid;
    gap: 2px;
    min-width: 0;
  }

  header strong {
    font-size: 16px;
  }

  header span,
  section > span {
    color: #52616f;
    font-size: 12px;
    font-weight: 700;
  }

  .pill {
    border: 1px solid #d1d8e2;
    border-radius: 999px;
    padding: 5px 8px;
    background: #ffffff;
    font-size: 11px;
    font-weight: 800;
  }

  .pill.live {
    border-color: #86efac;
    background: #dcfce7;
    color: #166534;
  }

  .miniStatusGrid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }

  .miniStatus {
    display: grid;
    grid-template-columns: 8px 1fr;
    gap: 3px 7px;
    min-width: 0;
    border: 1px solid #e0e7ef;
    border-radius: 8px;
    background: #ffffff;
    padding: 8px;
    font-size: 11px;
  }

  .miniStatus strong {
    grid-column: 2;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .miniDot {
    grid-row: span 2;
    align-self: center;
    width: 7px;
    height: 7px;
    border-radius: 999px;
    background: #ef4444;
  }

  .miniDot.ok {
    background: #16a34a;
  }

  input {
    width: 100%;
    min-width: 0;
    border: 1px solid #c9d3df;
    border-radius: 6px;
    padding: 9px 10px;
    color: #1f2933;
    background: #ffffff;
    font: inherit;
  }

  .chatFeed,
  .miniTranscript {
    display: grid;
    gap: 8px;
    min-width: 0;
    border: 1px solid #d8dee7;
    border-radius: 8px;
    background: #ffffff;
    padding: 10px;
  }

  .chatFeed {
    min-height: 150px;
    max-height: 240px;
  }

  .messages {
    display: grid;
    align-content: start;
    gap: 7px;
    overflow: auto;
  }

  .message {
    max-width: 92%;
    border: 1px solid #e0e7ef;
    border-radius: 8px;
    padding: 8px 9px;
    background: #f8fafc;
  }

  .message.user {
    justify-self: end;
    border-color: #bfdbfe;
    background: #eff6ff;
  }

  .message.assistant {
    justify-self: start;
    border-color: #c7d2fe;
    background: #eef2ff;
  }

  .message.meeting {
    justify-self: stretch;
    max-width: 100%;
    border-color: #e5e7eb;
    background: #ffffff;
    color: #4b5563;
    font-size: 12px;
  }

  .message.insight {
    justify-self: stretch;
    max-width: 100%;
    border-color: #fed7aa;
    background: #fff7ed;
    color: #7c2d12;
    font-weight: 700;
  }

  .message.insight.DECISION {
    border-color: #bbf7d0;
    background: #f0fdf4;
    color: #14532d;
  }

  .message.insight.OPEN_QUESTION {
    border-color: #fde68a;
    background: #fffbeb;
    color: #713f12;
  }

  p {
    margin: 0;
    line-height: 1.42;
  }

  small,
  .muted {
    color: #6b7280;
  }

  .miniTranscript {
    align-content: start;
    overflow: auto;
  }

  .miniTranscript article {
    display: grid;
    gap: 3px;
    border-top: 1px solid #eef2f7;
    padding-top: 8px;
  }

  .miniTranscript article:first-of-type {
    border-top: 0;
    padding-top: 0;
  }

  .miniTranscript article strong {
    color: #52616f;
    font-size: 12px;
  }

  .liveControls button {
    flex: 1 1 0;
  }

  form input {
    flex: 1 1 auto;
  }

  form button {
    width: 38px;
    padding: 0;
  }
`;

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
