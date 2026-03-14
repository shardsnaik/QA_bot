// hooks/useVoiceSocket.js
// Handles all WebSocket communication, PCM capture, VAD, and audio playback.

import { useRef, useState, useCallback, useEffect } from "react";

// ── WebSocket URL resolution ─────────────────────────────────
// Priority: VITE_WS_URL env var → auto-detect from window.location
//
// In your .env.production set the plain URL — no extra wss:// prefix:
//   VITE_WS_URL=wss://qa-bot-voice-to-voice.onrender.com/ws/voice
//
// In your .env.local (dev):
//   VITE_WS_URL=ws://localhost:8000/ws/voice

const _rawEnvUrl = "wss://qa-bot-voice-to-voice.onrender.com/ws/voice"

// Fix 1: strip accidental double-protocol  wss://wss//... → wss://...
const _deduped = _rawEnvUrl.replace(/^(wss?:\/\/)(wss?:\/+)/i, "$1");

// Fix 2: ensure path ends with /ws/voice
//   accepts bare domain:  wss://host.onrender.com
//   accepts full URL:     wss://host.onrender.com/ws/voice
//   accepts trailing /:   wss://host.onrender.com/
const _withPath = _deduped
  ? _deduped.replace(/\/ws\/voice$/, "").replace(/\/$/, "") + "/ws/voice"
  : "";

const WS_URL = _withPath ||
  (window.location.protocol === "https:" ? "wss" : "ws") +
  "://" +
  (window.location.hostname === "localhost"
    ? "localhost:8001"
    : window.location.host) +
  "/ws/voice";

// Log resolved URL once so you can verify in the browser console
console.debug("[VoiceSocket] WS_URL resolved to:", WS_URL);

// ── Audio playback queue ─────────────────────────────────────
class AudioQueue {
  constructor() {
    this.queue    = [];
    this.playing  = false;
    this.ctx      = new (window.AudioContext || window.webkitAudioContext)();
    this.onPlay   = null;   // callback when playback starts/ends
  }

  async resume() {
    if (this.ctx.state === "suspended") await this.ctx.resume();
  }

  enqueue(base64mp3) {
    this.queue.push(base64mp3);
    if (!this.playing) this._drain();
  }

  async _drain() {
    if (!this.queue.length) {
      this.playing = false;
      this.onPlay?.(false);
      return;
    }
    this.playing = true;
    this.onPlay?.(true);

    const b64  = this.queue.shift();
    const buf  = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)).buffer;

    try {
      const decoded = await this.ctx.decodeAudioData(buf);
      const src     = this.ctx.createBufferSource();
      src.buffer    = decoded;
      src.connect(this.ctx.destination);
      src.onended   = () => this._drain();
      src.start();
    } catch (e) {
      console.warn("Audio decode error:", e);
      this._drain();
    }
  }

  clear() {
    this.queue   = [];
    this.playing = false;
  }
}

// ── PCM float32 → int16 base64 ──────────────────────────────
function pcmToBase64(float32Array) {
  const int16 = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    int16[i] = Math.max(-32768, Math.min(32767, float32Array[i] * 32768));
  }
  let bin = "";
  const bytes = new Uint8Array(int16.buffer);
  for (let i = 0; i < bytes.byteLength; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

// ── Hook ─────────────────────────────────────────────────────
export default function useVoiceSocket() {
  const [status, setStatus]           = useState("disconnected"); // disconnected|idle|listening|processing|responding|speaking
  const [messages, setMessages]       = useState([]);
  const [streamingText, setStreaming] = useState("");

  const wsRef          = useRef(null);
  const audioQueueRef  = useRef(null);
  const mediaStreamRef = useRef(null);
  const scriptProcRef  = useRef(null);
  const analyserRef    = useRef(null);
  const isListeningRef = useRef(false);

  // Init audio queue once
  if (!audioQueueRef.current) {
    audioQueueRef.current = new AudioQueue();
  }

  // ── WebSocket helpers ──────────────────────────────────────
  const sendWS = useCallback((obj) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(obj));
    }
  }, []);

  // ── Commit streaming buffer into messages list ─────────────
  const streamBufferRef = useRef("");

  const finaliseAiMessage = useCallback(() => {
    const text = streamBufferRef.current.trim();
    if (text) {
      setMessages((prev) => [...prev, { role: "ai", text }]);
    }
    streamBufferRef.current = "";
    setStreaming("");
  }, []);

  // ── Connect ───────────────────────────────────────────────
  const connect = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState < 2) wsRef.current.close();
    setStatus("connecting");

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => setStatus("idle");

    ws.onmessage = ({ data }) => {
      const msg = JSON.parse(data);
      switch (msg.event) {
        case "transcript":
          setMessages((prev) => [...prev, { role: "user", text: msg.text }]);
          streamBufferRef.current = "";
          setStreaming("");
          setStatus("responding");
          break;

        case "token":
          streamBufferRef.current += msg.text;
          setStreaming(streamBufferRef.current);
          break;

        case "audio_chunk":
          audioQueueRef.current.onPlay = (playing) =>
            setStatus(playing ? "speaking" : "idle");
          audioQueueRef.current.enqueue(msg.data);
          break;

        case "turn_end":
          finaliseAiMessage();
          setStatus("idle");
          break;

        case "processing":
          if (msg.stage === "stt")  setStatus("processing");
          if (msg.stage === "llm")  setStatus("responding");
          if (msg.stage === "idle") setStatus("idle");
          break;

        case "reset_ack":
          setMessages([]);
          setStreaming("");
          streamBufferRef.current = "";
          audioQueueRef.current.clear();
          setStatus("idle");
          break;

        case "error":
          console.error("Server error:", msg.message);
          setStatus("idle");
          break;

        default:
          break;
      }
    };

    ws.onclose = () => {
      setStatus("disconnected");
      // Auto-reconnect after 3s (handles Render spin-down / network blip)
      setTimeout(() => {
        if (wsRef.current === ws) connect();
      }, 3000);
    };
    ws.onerror = (e) => console.error("WS error", e);
  }, [finaliseAiMessage]);

  // ── Mic start ─────────────────────────────────────────────
  const startListening = useCallback(async () => {
    if (isListeningRef.current) return;
    isListeningRef.current = true;

    await audioQueueRef.current.resume();

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: 16000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });
    mediaStreamRef.current = stream;

    const ctx    = audioQueueRef.current.ctx;
    const source = ctx.createMediaStreamSource(stream);

    // Analyser for visualiser
    const analyser  = ctx.createAnalyser();
    analyser.fftSize = 64;
    source.connect(analyser);
    analyserRef.current = analyser;

    // ScriptProcessor → PCM → WebSocket
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    source.connect(proc);
    proc.connect(ctx.destination);
    proc.onaudioprocess = (e) => {
      const pcm = e.inputBuffer.getChannelData(0);
      const b64 = pcmToBase64(pcm);
      sendWS({ event: "audio_chunk", data: b64, sample_rate: 16000 });
    };
    scriptProcRef.current = proc;

    setStatus("listening");
  }, [sendWS]);

  // ── Mic stop ──────────────────────────────────────────────
  const stopListening = useCallback(() => {
    if (!isListeningRef.current) return;
    isListeningRef.current = false;

    scriptProcRef.current?.disconnect();
    scriptProcRef.current = null;

    mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
    mediaStreamRef.current = null;

    analyserRef.current = null;
    sendWS({ event: "end_of_speech" });
    setStatus("processing");
  }, [sendWS]);

  // ── Reset chat ────────────────────────────────────────────
  const resetChat = useCallback(() => {
    sendWS({ event: "reset" });
  }, [sendWS]);

  // ── Expose analyser for visualiser ────────────────────────
  const getAnalyser = useCallback(() => analyserRef.current, []);

  // ── Auto-connect on mount ─────────────────────────────────
  // React 18 Strict Mode double-invokes effects in dev — the ignore
  // flag prevents a second socket being opened then immediately closed.
  useEffect(() => {
    let ignore = false;
    if (!ignore) connect();
    return () => {
      ignore = true;
      const ws = wsRef.current;
      if (ws && ws.readyState < 2) ws.close();
      wsRef.current = null;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return {
    status,
    messages,
    streamingText,
    connect,
    startListening,
    stopListening,
    resetChat,
    getAnalyser,
  };
}