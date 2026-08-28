import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Brain,
  CheckCircle2,
  Circle,
  Loader2,
  Mic,
  MonitorUp,
  PauseCircle,
  RotateCcw,
  Send,
  SlidersHorizontal,
  Volume2,
  VolumeX,
} from "lucide-react";
import "./styles.css";

type Category = "EXPLICIT_CUE" | "IMPLICIT_CUE" | "CHIME_IN" | "KEEP_SILENCE";
type AudioSource = "TAB_AUDIO" | "MIC";

type ShareableInfo = { context: string; information: string };
type Delegate = {
  name: string;
  role: string;
  meeting_intents: string[];
  shareable_information: ShareableInfo[];
};
type Utterance = {
  id: number;
  speaker: string;
  text: string;
  timestamp?: string;
  source?: string;
  start?: number;
  end?: number;
  language?: string | null;
  asr_latency_ms?: number | null;
};
type Decision = {
  utterance_id: number;
  category: Category;
  should_intervene: boolean;
  confidence: number;
  response: string | null;
  reason: string;
  llm_latency_ms?: number;
  pipeline_latency_ms?: number;
  total_suggestion_latency_ms?: number;
  stale?: boolean;
  displayed?: boolean;
  filtered?: boolean;
};
type Health = { backend: "ok"; lm_studio: "ok" | "error"; model?: string; asr: string; error?: string };
type ASRStatus = {
  status: "idle" | "loading" | "ready" | "error" | "unavailable";
  provider: string;
  model: string;
  device?: string;
  compute_type?: string;
  language?: string | null;
  detail?: string;
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
type LiveEvent = {
  type: string;
  meeting_id?: string | null;
  payload: Record<string, any>;
};

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const WS = API.replace(/^http/, "ws");

const defaultDelegate: Delegate = {
  name: "Advisor",
  role: "Senior product and engineering advisor for the AI Call Advisor project",
  meeting_intents: [
    "Help evaluate product and engineering tradeoffs during live calls",
    "Identify risks around privacy, consent, latency, transcription errors, and user trust",
    "Suggest pragmatic next implementation steps for a meeting delegate MVP",
  ],
  shareable_information: [
    {
      context: "When the team discusses live meetings, Teams, Google Meet, audio capture, or transcription",
      information:
        "The implementation captures tab audio and local microphone, sends audio chunks to the backend, transcribes with faster-whisper, and can optionally process utterances through LM Studio.",
    },
    {
      context: "When the team discusses LLM integration or local models",
      information: "The backend talks to LM Studio through OpenAI-compatible endpoints.",
    },
  ],
};

const emptyLevels: Record<AudioSource, number> = { TAB_AUDIO: 0, MIC: 0 };
const emptySpeech: Record<AudioSource, boolean> = { TAB_AUDIO: false, MIC: false };
const emptyTrackStatus: Record<AudioSource, string> = { TAB_AUDIO: "idle", MIC: "idle" };

function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [asrStatus, setAsrStatus] = useState<ASRStatus | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [delegate, setDelegate] = useState<Delegate>(defaultDelegate);
  const [meetingId, setMeetingId] = useState("");
  const [utterances, setUtterances] = useState<Utterance[]>([]);
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const [insights, setInsights] = useState<MeetingInsight[]>([]);
  const [debug, setDebug] = useState<Record<string, string | number>>({});
  const [liveActive, setLiveActive] = useState(false);
  const [llmEnabled, setLlmEnabled] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [captureStage, setCaptureStage] = useState<"idle" | "screen" | "microphone" | "connecting" | "live" | "error">(
    "idle",
  );
  const [captureError, setCaptureError] = useState("");
  const [asrProcessing, setAsrProcessing] = useState(0);
  const [lastTranscriptEmpty, setLastTranscriptEmpty] = useState(false);
  const [previewStream, setPreviewStream] = useState<MediaStream | null>(null);
  const [audioLevels, setAudioLevels] = useState<Record<AudioSource, number>>(emptyLevels);
  const [speechActive, setSpeechActive] = useState<Record<AudioSource, boolean>>(emptySpeech);
  const [trackStatus, setTrackStatus] = useState<Record<AudioSource, string>>(emptyTrackStatus);
  const [manualSpeaker, setManualSpeaker] = useState("ME");
  const [manualText, setManualText] = useState("");
  const [question, setQuestion] = useState("");
  const [askingQuestion, setAskingQuestion] = useState(false);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const liveStreams = useRef<MediaStream[]>([]);
  const audioContexts = useRef<AudioContext[]>([]);
  const audioNodes = useRef<AudioNode[]>([]);
  const meterUpdateAt = useRef<Record<AudioSource, number>>({ TAB_AUDIO: 0, MIC: 0 });

  const latestDecision = useMemo(
    () => [...decisions].reverse().find((item) => item.should_intervene && item.response),
    [decisions],
  );
  const latestAsrLatency = useMemo(() => {
    const latencies = utterances.map((item) => item.asr_latency_ms).filter((value): value is number => typeof value === "number");
    return latencies.at(-1) ?? null;
  }, [utterances]);
  const activeSources = useMemo(
    () => (Object.keys(speechActive) as AudioSource[]).filter((source) => speechActive[source]),
    [speechActive],
  );

  async function refreshStatus() {
    const [healthRes, modelsRes, asrRes] = await Promise.all([
      fetch(`${API}/health`).then((r) => r.json()),
      fetch(`${API}/models`).then((r) => (r.ok ? r.json() : { models: [] })),
      fetch(`${API}/api/asr/status`).then((r) => r.json()),
    ]);
    setHealth(healthRes);
    setAsrStatus(asrRes);
    const ids = (modelsRes.models ?? []).map((item: { id: string }) => item.id);
    setModels(ids);
    setSelectedModel((current) => current || healthRes.model || ids[0] || "");
  }

  async function refreshAsrStatus() {
    const asrRes = await fetch(`${API}/api/asr/status`).then((r) => r.json());
    setAsrStatus(asrRes);
  }

  useEffect(() => {
    refreshStatus().catch(console.error);
    const interval = window.setInterval(() => {
      refreshAsrStatus().catch(console.error);
    }, 3500);
    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!videoRef.current) return;
    videoRef.current.srcObject = previewStream;
  }, [previewStream]);

  useEffect(() => {
    return () => stopLive();
  }, []);

  function resetSession() {
    stopLive();
    setMeetingId("");
    setUtterances([]);
    setDecisions([]);
    setInsights([]);
    setDebug({});
    setAsrProcessing(0);
    setLastTranscriptEmpty(false);
    setManualText("");
    setQuestion("");
    setCaptureError("");
  }

  function updateLlmEnabled(enabled: boolean) {
    setLlmEnabled(enabled);
    const socket = wsRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "llm.set_enabled", enabled }));
    }
  }

  async function startLive() {
    if (wsRef.current?.readyState === WebSocket.OPEN || captureStage === "screen" || captureStage === "microphone") return;
    setCaptureError("");
    setUtterances([]);
    setDecisions([]);
    setInsights([]);
    setDebug({});
    setAsrProcessing(0);
    setLastTranscriptEmpty(false);

    let display: MediaStream | null = null;
    let mic: MediaStream | null = null;
    try {
      setCaptureStage("screen");
      const displayStream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
      display = displayStream;
      setPreviewStream(displayStream);

      setCaptureStage("microphone");
      const micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mic = micStream;

      setCaptureStage("connecting");
      const socket = await openLiveSocket();
      wsRef.current = socket;
      liveStreams.current.push(displayStream, micStream);
      displayStream.getVideoTracks()[0]?.addEventListener("ended", () => stopLive(), { once: true });

      socket.send(
        JSON.stringify({
          type: "meeting.start",
          delegate,
          model: selectedModel || undefined,
          llm_enabled: llmEnabled,
        }),
      );

      startPcmSender(socket, new MediaStream(displayStream.getAudioTracks()), "TAB_AUDIO");
      startPcmSender(socket, new MediaStream(micStream.getAudioTracks()), "MIC");
      setTrackStatus((prev) => ({
        ...prev,
        TAB_AUDIO: displayStream.getAudioTracks().length ? "capturing" : "no track",
        MIC: micStream.getAudioTracks().length ? "capturing" : "no track",
      }));
      setLiveActive(true);
      setCaptureStage("live");
    } catch (error) {
      display?.getTracks().forEach((track) => track.stop());
      mic?.getTracks().forEach((track) => track.stop());
      setCaptureStage("error");
      setCaptureError(error instanceof Error ? error.message : String(error));
      stopLive();
    }
  }

  function openLiveSocket(): Promise<WebSocket> {
    const socket = new WebSocket(`${WS}/ws/live`);
    socket.onmessage = (message) => handleLiveEvent(JSON.parse(message.data));
    socket.onclose = () => {
      wsRef.current = null;
      setLiveActive(false);
      setCaptureStage((current) => (current === "live" ? "idle" : current));
    };
    return new Promise((resolve, reject) => {
      socket.onopen = () => resolve(socket);
      socket.onerror = () => reject(new Error("Nao foi possivel conectar ao backend live."));
    });
  }

  function handleLiveEvent(event: LiveEvent) {
    if (event.type === "meeting.state.updated") {
      if (event.payload.meeting_id) setMeetingId(String(event.payload.meeting_id));
      if (typeof event.payload.llm_enabled === "boolean") setLlmEnabled(event.payload.llm_enabled);
      if (event.payload.last_asr_latency_ms) {
        setDebug((prev) => ({ ...prev, ASR: `${event.payload.last_asr_latency_ms} ms` }));
      }
      if (typeof event.payload.last_asr_queue_latency_ms === "number") {
        setDebug((prev) => ({ ...prev, "ASR espera": `${event.payload.last_asr_queue_latency_ms} ms` }));
      }
      if (typeof event.payload.last_transcript_empty === "boolean") {
        setLastTranscriptEmpty(event.payload.last_transcript_empty);
      }
    }
    if (event.type === "asr.started") {
      setAsrProcessing((count) => count + 1);
      setDebug((prev) => ({
        ...prev,
        "ASR fila": String(event.payload.queue_size ?? 0),
        "ASR trecho": `${event.payload.source ?? "-"} ${event.payload.duration_ms ?? "-"} ms`,
      }));
    }
    if (event.type === "asr.completed") {
      setAsrProcessing((count) => Math.max(0, count - 1));
      setDebug((prev) => ({
        ...prev,
        "ASR RTF": typeof event.payload.real_time_factor === "number" ? event.payload.real_time_factor.toFixed(2) : "-",
        "ASR segmentos": String(event.payload.segment_count ?? 0),
        "ASR espera": `${event.payload.asr_queue_latency_ms ?? 0} ms`,
      }));
    }
    if (event.type === "transcript.empty") {
      setLastTranscriptEmpty(true);
    }
    if (event.type === "utterance.final") {
      setUtterances((prev) => [...prev, event.payload as Utterance]);
    }
    if (event.type === "intervention.decided") {
      setDecisions((prev) => [...prev, event.payload as Decision]);
    }
    if (event.type === "intervention.error") {
      setDebug((prev) => ({ ...prev, LLM: String(event.payload.error ?? "erro") }));
    }
    if (event.type === "meeting.insight.detected") {
      setInsights((prev) => [...prev, event.payload as MeetingInsight]);
    }
    if (event.type === "audio.chunk") {
      const source = event.payload.source as AudioSource;
      setDebug((prev) => ({ ...prev, [`${source} chunk`]: `${event.payload.bytes} bytes` }));
    }
    if (event.type === "speech.started" || event.type === "speech.ended") {
      const source = event.payload.source as AudioSource;
      setSpeechActive((prev) => ({ ...prev, [source]: event.type === "speech.started" }));
    }
    if (event.type === "transcript.error") {
      setAsrProcessing((count) => Math.max(0, count - 1));
      setCaptureError(String(event.payload.error ?? "Erro de transcricao"));
    }
  }

  function startPcmSender(socket: WebSocket, stream: MediaStream, source: AudioSource) {
    if (stream.getAudioTracks().length === 0) {
      setTrackStatus((prev) => ({ ...prev, [source]: "no track" }));
      return;
    }

    const context = new AudioContext();
    const input = context.createMediaStreamSource(stream);
    const processor = context.createScriptProcessor(4096, 1, 1);
    const muted = context.createGain();
    muted.gain.value = 0;

    input.connect(processor);
    processor.connect(muted);
    muted.connect(context.destination);

    processor.onaudioprocess = (event) => {
      const samples = event.inputBuffer.getChannelData(0);
      updateAudioMeter(source, samples);
      if (socket.readyState !== WebSocket.OPEN) return;
      const pcm = downsampleToPcm16(samples, context.sampleRate, 16000);
      if (!pcm.byteLength) return;
      socket.send(JSON.stringify({ type: "audio.chunk", source, sample_rate: 16000, data: arrayBufferToBase64(pcm.buffer) }));
    };

    audioContexts.current.push(context);
    audioNodes.current.push(input, processor, muted);
    void context.resume();
  }

  function updateAudioMeter(source: AudioSource, samples: Float32Array) {
    let sum = 0;
    for (let index = 0; index < samples.length; index += 1) sum += samples[index] * samples[index];
    const rms = Math.sqrt(sum / samples.length);
    const level = Math.min(1, rms * 12);
    const now = performance.now();
    if (now - meterUpdateAt.current[source] < 80) return;
    meterUpdateAt.current[source] = now;
    setAudioLevels((prev) => ({ ...prev, [source]: level }));
  }

  function stopLive() {
    for (const node of audioNodes.current) {
      try {
        node.disconnect();
      } catch {
        // Already disconnected by the browser.
      }
    }
    for (const context of audioContexts.current) void context.close();
    for (const stream of liveStreams.current) {
      for (const track of stream.getTracks()) track.stop();
    }
    liveStreams.current = [];
    audioContexts.current = [];
    audioNodes.current = [];
    setPreviewStream(null);
    setAudioLevels(emptyLevels);
    setSpeechActive(emptySpeech);
    setTrackStatus(emptyTrackStatus);
    setAsrProcessing(0);
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "meeting.stop" }));
      wsRef.current.close();
    }
    wsRef.current = null;
    setLiveActive(false);
    setCaptureStage("idle");
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
    if (!manualText.trim()) return;
    const id = await createMeetingIfNeeded();
    const res = await fetch(`${API}/meetings/${id}/utterances`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ speaker: manualSpeaker, text: manualText, source: "MANUAL" }),
    });
    const data = await res.json();
    setUtterances((prev) => [...prev, data.utterance]);
    setInsights((prev) => [...prev, ...(data.insights ?? [])]);
    setDecisions((prev) => [...prev, data.decision]);
    setManualText("");
  }

  async function askMeetingQuestion(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const cleanQuestion = question.trim();
    if (!cleanQuestion || !meetingId || askingQuestion || !llmEnabled) return;
    setAskingQuestion(true);
    try {
      const res = await fetch(`${API}/meetings/${meetingId}/questions?model=${encodeURIComponent(selectedModel)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: cleanQuestion }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setDecisions((prev) => [
        ...prev,
        {
          utterance_id: utterances.at(-1)?.id ?? 0,
          category: "CHIME_IN",
          should_intervene: true,
          confidence: 1,
          response: data.answer,
          reason: cleanQuestion,
          llm_latency_ms: data.llm_latency_ms,
        },
      ]);
      setQuestion("");
    } finally {
      setAskingQuestion(false);
    }
  }

  const screenStatus =
    captureStage === "live"
      ? "Transmitindo"
      : captureStage === "screen"
        ? "Escolhendo tela"
        : captureStage === "microphone"
          ? "Autorizando mic"
          : captureStage === "connecting"
            ? "Conectando"
            : "Pronto";
  const asrLabel =
    asrProcessing > 0
      ? `processando ${asrProcessing}`
      : asrStatus?.status === "idle"
        ? "idle"
        : asrStatus?.status === "loading"
          ? "carregando"
          : asrStatus?.status ?? "unknown";

  return (
    <main className={llmEnabled ? "appShell" : "appShell llmMuted"}>
      <aside className="controlRail">
        <div className="brandBlock">
          <div>
            <span className="eyebrow">AI Call Advisor</span>
            <h1>Live transcription</h1>
          </div>
          <button className="iconButton" onClick={refreshStatus} title="Atualizar status">
            <RotateCcw size={18} />
          </button>
        </div>

        <div className="primaryControls">
          <button className="startButton" onClick={startLive} disabled={liveActive || captureStage === "screen" || captureStage === "microphone"}>
            {captureStage === "screen" || captureStage === "microphone" || captureStage === "connecting" ? (
              <Loader2 size={18} className="spin" />
            ) : (
              <MonitorUp size={18} />
            )}
            Iniciar captura
          </button>
          <button className="stopButton" onClick={stopLive} disabled={!liveActive && captureStage === "idle"}>
            <PauseCircle size={18} />
            Parar
          </button>
        </div>

        <section className="controlGroup">
          <div className="switchRow">
            <div>
              <strong>LM Studio</strong>
              <span>{health?.lm_studio === "ok" ? selectedModel || "online" : "offline"}</span>
            </div>
            <button
              className={llmEnabled ? "switch isOn" : "switch"}
              onClick={() => updateLlmEnabled(!llmEnabled)}
              role="switch"
              aria-checked={llmEnabled}
              title="Ligar ou desligar chamadas ao LLM"
            >
              <span />
            </button>
          </div>
          <select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)} disabled={!llmEnabled}>
            <option value="">Modelo local</option>
            {models.map((model) => (
              <option key={model}>{model}</option>
            ))}
          </select>
        </section>

        <section className="statusStack">
          <StatusLine label="Backend" ok={health?.backend === "ok"} value={health?.backend ?? "unknown"} />
          <StatusLine
            label="Whisper"
            ok={Boolean(asrStatus && !["error", "unavailable"].includes(asrStatus.status))}
            value={asrStatus?.status ?? "unknown"}
          />
          <StatusLine label="Modelo ASR" ok={Boolean(asrStatus?.model)} value={asrStatus?.model ?? "-"} />
          <StatusLine label="Sessao" ok={Boolean(meetingId)} value={meetingId ? meetingId.slice(0, 8) : "-"} />
        </section>

        <section className="controlGroup">
          <button className="secondaryButton" onClick={() => setSettingsOpen((open) => !open)}>
            <SlidersHorizontal size={17} />
            Perfil do advisor
          </button>
          {settingsOpen && (
            <div className="settingsPanel">
              <label>
                Nome
                <input value={delegate.name} onChange={(event) => setDelegate({ ...delegate, name: event.target.value })} />
              </label>
              <label>
                Papel
                <input value={delegate.role} onChange={(event) => setDelegate({ ...delegate, role: event.target.value })} />
              </label>
              <label>
                Intencoes
                <textarea
                  value={delegate.meeting_intents.join("\n")}
                  onChange={(event) =>
                    setDelegate({ ...delegate, meeting_intents: event.target.value.split("\n").filter(Boolean) })
                  }
                />
              </label>
            </div>
          )}
        </section>

        <section className="manualBox">
          <div className="manualHeader">
            <span>Manual</span>
            <Brain size={15} />
          </div>
          <div className="manualRow">
            <input value={manualSpeaker} onChange={(event) => setManualSpeaker(event.target.value)} aria-label="Speaker" />
            <input
              value={manualText}
              onChange={(event) => setManualText(event.target.value)}
              onKeyDown={(event) => event.key === "Enter" && void sendManual()}
              placeholder="Utterance"
            />
            <button className="iconButton" onClick={sendManual} title="Enviar utterance">
              <Send size={17} />
            </button>
          </div>
        </section>
      </aside>

      <section className="stage">
        <header className="stageHeader">
          <div>
            <span className="eyebrow">Screen preview</span>
            <h2>{screenStatus}</h2>
          </div>
          <div className={liveActive ? "livePill on" : "livePill"}>
            {liveActive ? <CheckCircle2 size={15} /> : <Circle size={15} />}
            {liveActive ? "LIVE" : "IDLE"}
          </div>
        </header>

        <div className="videoFrame">
          <video ref={videoRef} autoPlay muted playsInline />
          {!previewStream && (
            <div className="emptyPreview">
              <MonitorUp size={42} />
              <span>Nenhuma tela selecionada</span>
            </div>
          )}
        </div>

        <div className="meterDock">
          <AudioMeter
            source="TAB_AUDIO"
            label="Tela"
            level={audioLevels.TAB_AUDIO}
            speechActive={speechActive.TAB_AUDIO}
            status={trackStatus.TAB_AUDIO}
          />
          <AudioMeter
            source="MIC"
            label="Microfone"
            level={audioLevels.MIC}
            speechActive={speechActive.MIC}
            status={trackStatus.MIC}
          />
        </div>

        <section className="signalPanel">
          <Metric label="Utterances" value={String(utterances.length)} />
          <Metric label="Whisper" value={asrLabel} muted={asrStatus?.status === "idle"} />
          <Metric label="ASR" value={latestAsrLatency ? `${latestAsrLatency} ms` : "-"} />
          <Metric label="Ouvindo" value={activeSources.length ? activeSources.map(sourceLabel).join(" + ") : "-"} />
          <Metric label="LLM" value={llmEnabled ? "on" : "off"} muted={!llmEnabled} />
          {Object.entries(debug).map(([key, value]) => (
            <Metric key={key} label={key} value={String(value)} />
          ))}
          {lastTranscriptEmpty && <div className="noticeLine">Audio processado, mas o Whisper nao reconheceu texto nesse trecho.</div>}
          {captureError && <div className="errorLine">{captureError}</div>}
        </section>
      </section>

      <aside className="transcriptRail">
        <header className="railHeader">
          <div>
            <span className="eyebrow">Transcript</span>
            <h2>Ao vivo</h2>
          </div>
          <button className="iconButton" onClick={resetSession} title="Limpar sessao">
            <RotateCcw size={18} />
          </button>
        </header>

        <div className="transcriptList">
          {utterances.length === 0 && (
            <div className="emptyTranscript">
              <Mic size={24} />
              <span>Aguardando fala detectada</span>
            </div>
          )}
          {utterances.map((item) => (
            <article key={`${item.id}-${item.source}-${item.text}`} className="transcriptItem">
              <div className="utteranceMeta">
                <strong>{item.speaker}</strong>
                <span>{sourceLabel(item.source)}</span>
                {typeof item.asr_latency_ms === "number" && <small>{item.asr_latency_ms} ms</small>}
              </div>
              <p>{item.text}</p>
            </article>
          ))}
          {activeSources.map((source) => (
            <article key={`active-${source}`} className="transcriptItem pending">
              <div className="utteranceMeta">
                <strong>{sourceLabel(source)}</strong>
                <span>captando</span>
              </div>
              <div className="typingDots">
                <span />
                <span />
                <span />
              </div>
            </article>
          ))}
        </div>

        {llmEnabled && (
          <section className="advisorPanel">
            <div className="advisorHeader">
              <span>Advisor</span>
              <strong>{latestDecision?.category ?? "Sem resposta"}</strong>
            </div>
            {latestDecision?.response && <blockquote>{latestDecision.response}</blockquote>}
            {insights.slice(-3).map((insight) => (
              <article key={insight.id} className="insightItem">
                <strong>{insight.type.replace("_", " ")}</strong>
                <p>{insight.text}</p>
              </article>
            ))}
            <form onSubmit={askMeetingQuestion} className="questionForm">
              <input
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Perguntar ao contexto"
                disabled={!meetingId || askingQuestion}
              />
              <button className="iconButton" type="submit" disabled={!meetingId || !question.trim() || askingQuestion}>
                {askingQuestion ? <Loader2 size={17} className="spin" /> : <Send size={17} />}
              </button>
            </form>
          </section>
        )}
      </aside>
    </main>
  );
}

function StatusLine({ label, value, ok }: { label: string; value: string; ok: boolean }) {
  return (
    <div className="statusLine">
      <span className={ok ? "statusDot ok" : "statusDot"} />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Metric({ label, value, muted = false }: { label: string; value: string; muted?: boolean }) {
  return (
    <div className={muted ? "metric mutedMetric" : "metric"}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function AudioMeter({
  source,
  label,
  level,
  speechActive,
  status,
}: {
  source: AudioSource;
  label: string;
  level: number;
  speechActive: boolean;
  status: string;
}) {
  const bars = [0.18, 0.32, 0.5, 0.7, 0.9];
  const active = speechActive || level > 0.08;
  return (
    <div className={active ? "audioMeter active" : "audioMeter"} data-source={source}>
      <div className="audioIcon">{status === "no track" ? <VolumeX size={18} /> : <Volume2 size={18} />}</div>
      <div>
        <strong>{label}</strong>
        <span>{status === "no track" ? "sem faixa" : active ? "ouvindo" : "silencio"}</span>
      </div>
      <div className="bars" aria-hidden="true">
        {bars.map((threshold) => (
          <span key={threshold} style={{ transform: `scaleY(${Math.max(0.12, Math.min(1, level / threshold))})` }} />
        ))}
      </div>
    </div>
  );
}

function sourceLabel(source?: string | null): string {
  if (source === "MIC" || source === "LOCAL_MIC" || source === "LOCAL_MIC_AUDIO") return "Mic";
  if (source === "TAB_AUDIO" || source === "REMOTE_AUDIO" || source === "REMOTE") return "Tela";
  if (source === "MANUAL") return "Manual";
  if (source === "FILE") return "Arquivo";
  return source || "-";
}

function downsampleToPcm16(input: Float32Array, sourceRate: number, targetRate: number): Int16Array {
  if (sourceRate === targetRate) return floatToPcm16(input);
  const ratio = sourceRate / targetRate;
  const outputLength = Math.max(1, Math.floor(input.length / ratio));
  const output = new Float32Array(outputLength);
  for (let index = 0; index < outputLength; index += 1) {
    const start = Math.floor(index * ratio);
    const end = Math.min(input.length, Math.floor((index + 1) * ratio));
    let sum = 0;
    for (let cursor = start; cursor < end; cursor += 1) sum += input[cursor];
    output[index] = sum / Math.max(1, end - start);
  }
  return floatToPcm16(output);
}

function floatToPcm16(input: Float32Array): Int16Array {
  const output = new Int16Array(input.length);
  for (let index = 0; index < input.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, input[index]));
    output[index] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return output;
}

function arrayBufferToBase64(buffer: ArrayBufferLike): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunkSize = 0x8000;
  for (let index = 0; index < bytes.length; index += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(index, index + chunkSize));
  }
  return btoa(binary);
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
