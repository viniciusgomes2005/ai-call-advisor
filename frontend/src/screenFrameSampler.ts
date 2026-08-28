// Visual context capture: shared screen -> frame sampling -> change detection ->
// relevant screenshot. No vision model reads these frames yet (see backend
// app/vision/provider.py) - this module only builds the synchronized capture
// pipeline for one to plug into later.
//
// The pure functions below (meanAbsoluteDifference, toGrayscale, shouldAcceptFrame,
// clampToMaxDimension) contain all the actual change-detection logic and take no DOM
// input, so they stay testable in isolation even though this project has no frontend
// test runner yet (see README "Visual Context Pipeline").

export type ChangeDetectionConfig = {
  sampleIntervalMs: number;
  forceCaptureIntervalMs: number;
  changeThreshold: number; // mean abs diff on a grayscale analysis frame, normalized 0..1
  maxDimension: number; // largest edge of the JPEG sent to the backend
  jpegQuality: number;
  analysisWidth: number;
  analysisHeight: number;
};

export const DEFAULT_CHANGE_DETECTION_CONFIG: ChangeDetectionConfig = {
  sampleIntervalMs: 4000,
  forceCaptureIntervalMs: 30000,
  changeThreshold: 0.04,
  maxDimension: 1280,
  jpegQuality: 0.7,
  analysisWidth: 64,
  analysisHeight: 36,
};

export type AcceptReason = "no_previous_frame" | "change_above_threshold" | "force_capture_interval";
export type SkipReason = "below_threshold";

export type FrameDecision = { accept: true; reason: AcceptReason } | { accept: false; reason: SkipReason };

/** Mean absolute pixel difference between two equal-length grayscale buffers,
 * normalized to 0..1. Buffers of mismatched (or zero) length are treated as maximally
 * different rather than throwing, since a resolution change is itself a real change. */
export function meanAbsoluteDifference(previous: ArrayLike<number>, current: ArrayLike<number>): number {
  if (previous.length === 0 || current.length === 0 || previous.length !== current.length) return 1;
  let sum = 0;
  for (let index = 0; index < current.length; index += 1) {
    sum += Math.abs(current[index] - previous[index]);
  }
  return sum / current.length / 255;
}

/** RGBA (e.g. from canvas ImageData.data) -> single-channel grayscale luminance. */
export function toGrayscale(rgba: ArrayLike<number>): Uint8ClampedArray {
  const pixelCount = Math.floor(rgba.length / 4);
  const gray = new Uint8ClampedArray(pixelCount);
  for (let pixel = 0; pixel < pixelCount; pixel += 1) {
    const offset = pixel * 4;
    gray[pixel] = 0.299 * rgba[offset] + 0.587 * rgba[offset + 1] + 0.114 * rgba[offset + 2];
  }
  return gray;
}

/** Deterministic accept/skip decision - no I/O, easy to unit test directly. */
export function shouldAcceptFrame(params: {
  changeScore: number;
  hasPreviousFrame: boolean;
  msSinceLastAccepted: number;
  config: ChangeDetectionConfig;
}): FrameDecision {
  const { changeScore, hasPreviousFrame, msSinceLastAccepted, config } = params;
  if (!hasPreviousFrame) return { accept: true, reason: "no_previous_frame" };
  if (changeScore >= config.changeThreshold) return { accept: true, reason: "change_above_threshold" };
  if (msSinceLastAccepted >= config.forceCaptureIntervalMs) return { accept: true, reason: "force_capture_interval" };
  return { accept: false, reason: "below_threshold" };
}

/** Scale width/height down (never up) so the largest edge is at most maxDimension. */
export function clampToMaxDimension(width: number, height: number, maxDimension: number): { width: number; height: number } {
  if (width <= 0 || height <= 0) return { width: 0, height: 0 };
  const largest = Math.max(width, height);
  if (largest <= maxDimension) return { width: Math.round(width), height: Math.round(height) };
  const scale = maxDimension / largest;
  return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) };
}

export type ScreenFrameSample = {
  timestamp: number; // meeting-elapsed seconds - same clock family as TranscriptSegment.start/end
  capturedAt: string; // ISO UTC, for logs only
  mimeType: string;
  width: number;
  height: number;
  changeScore: number | null;
  blob: Blob;
};

export type ScreenFrameSamplerStats = {
  sampled: number;
  skipped: number;
  accepted: number;
  lastChangeScore: number | null;
};

const emptyStats: ScreenFrameSamplerStats = { sampled: 0, skipped: 0, accepted: 0, lastChangeScore: null };

export type ScreenFrameSamplerOptions = {
  /** Read at every tick, so the sampler always draws from whatever <video> element
   * currently shows the shared-screen MediaStream - no second getDisplayMedia call. */
  getVideoElement: () => HTMLVideoElement | null;
  videoTrack: MediaStreamTrack;
  getElapsedSeconds: () => number;
  config?: Partial<ChangeDetectionConfig>;
  onAccepted: (sample: ScreenFrameSample) => void;
  onStats?: (stats: ScreenFrameSamplerStats) => void;
};

/** Samples the shared screen on an interval, applies cheap deterministic change
 * detection, and reports only relevant frames via onAccepted. Never requests a second
 * screen-share permission - it only ever reads whatever video element the caller hands
 * it via getVideoElement(). */
export class ScreenFrameSampler {
  private readonly config: ChangeDetectionConfig;
  private readonly getVideoElement: () => HTMLVideoElement | null;
  private readonly videoTrack: MediaStreamTrack;
  private readonly getElapsedSeconds: () => number;
  private readonly onAccepted: (sample: ScreenFrameSample) => void;
  private readonly onStats: (stats: ScreenFrameSamplerStats) => void;

  private readonly captureCanvas: HTMLCanvasElement;
  private readonly analysisCanvas: HTMLCanvasElement;

  private intervalId: number | null = null;
  private stopped = true;
  private ticking = false;
  private previousAnalysisFrame: Uint8ClampedArray | null = null;
  private lastAcceptedAtMs: number | null = null;
  private stats: ScreenFrameSamplerStats = { ...emptyStats };

  constructor(options: ScreenFrameSamplerOptions) {
    this.config = { ...DEFAULT_CHANGE_DETECTION_CONFIG, ...options.config };
    this.getVideoElement = options.getVideoElement;
    this.videoTrack = options.videoTrack;
    this.getElapsedSeconds = options.getElapsedSeconds;
    this.onAccepted = options.onAccepted;
    this.onStats = options.onStats ?? (() => {});
    this.captureCanvas = document.createElement("canvas");
    this.analysisCanvas = document.createElement("canvas");
  }

  start(): void {
    if (!this.stopped) return;
    this.stopped = false;
    this.intervalId = window.setInterval(() => {
      void this.tick();
    }, this.config.sampleIntervalMs);
  }

  /** Idempotent: safe to call more than once (meeting.stop, track "ended", and
   * WebSocket close can all race to stop the same sampler). */
  stop(): void {
    this.stopped = true;
    if (this.intervalId !== null) {
      window.clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }

  private async tick(): Promise<void> {
    if (this.stopped || this.ticking) return;
    if (this.videoTrack.readyState !== "live") {
      this.stop();
      return;
    }
    const video = this.getVideoElement();
    if (!video || !video.videoWidth || !video.videoHeight) return; // dimension still zero

    this.ticking = true;
    try {
      this.stats = { ...this.stats, sampled: this.stats.sampled + 1 };

      const analysisContext = this.analysisCanvas.getContext("2d", { willReadFrequently: true });
      if (!analysisContext) return;
      this.analysisCanvas.width = this.config.analysisWidth;
      this.analysisCanvas.height = this.config.analysisHeight;
      analysisContext.drawImage(video, 0, 0, this.config.analysisWidth, this.config.analysisHeight);
      const analysisFrame = toGrayscale(
        analysisContext.getImageData(0, 0, this.config.analysisWidth, this.config.analysisHeight).data,
      );

      const hasPreviousFrame = this.previousAnalysisFrame !== null;
      const changeScore = hasPreviousFrame
        ? meanAbsoluteDifference(this.previousAnalysisFrame as Uint8ClampedArray, analysisFrame)
        : 1;
      this.previousAnalysisFrame = analysisFrame;

      const decision = shouldAcceptFrame({
        changeScore,
        hasPreviousFrame,
        msSinceLastAccepted: this.lastAcceptedAtMs === null ? Number.POSITIVE_INFINITY : performance.now() - this.lastAcceptedAtMs,
        config: this.config,
      });

      this.stats = { ...this.stats, lastChangeScore: changeScore };

      if (!decision.accept) {
        this.stats = { ...this.stats, skipped: this.stats.skipped + 1 };
        this.onStats({ ...this.stats });
        return;
      }

      const { width, height } = clampToMaxDimension(video.videoWidth, video.videoHeight, this.config.maxDimension);
      const captureContext = this.captureCanvas.getContext("2d");
      if (!captureContext || width === 0 || height === 0) return;
      this.captureCanvas.width = width;
      this.captureCanvas.height = height;
      captureContext.drawImage(video, 0, 0, width, height);

      const blob = await new Promise<Blob | null>((resolve) => {
        this.captureCanvas.toBlob(resolve, "image/jpeg", this.config.jpegQuality);
      });
      if (!blob || this.stopped) return; // sampler was stopped mid-encode

      this.lastAcceptedAtMs = performance.now();
      this.stats = { ...this.stats, accepted: this.stats.accepted + 1 };
      this.onStats({ ...this.stats });
      this.onAccepted({
        timestamp: this.getElapsedSeconds(),
        capturedAt: new Date().toISOString(),
        mimeType: "image/jpeg",
        width,
        height,
        changeScore,
        blob,
      });
    } finally {
      this.ticking = false;
    }
  }
}
