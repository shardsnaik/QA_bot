// VoiceApp.jsx
// Root component — assembles all pieces.

import { useState } from "react";
import useVoiceSocket from "../../hooks/useVoiceSocket";
import AudioVisualiser from "../AudioVisualiser";
import MessageList    from "../MessageList";
import MicButton      from "../streaming cursor/MicButton";
import StatusBar      from "../streaming cursor/StatusBar";
import styles         from "./VoiceApp.module.css";

export default function VoiceApp() {
  const {
    status,
    messages,
    streamingText,
    connect,
    startListening,
    stopListening,
    resetChat,
    getAnalyser,
  } = useVoiceSocket();

  const isListening   = status === "listening";
  const isBusy        = status === "processing" || status === "responding" || status === "speaking";
  const isDisconnected = status === "disconnected" || status === "connecting";

  return (
    <div className={styles.shell}>

      {/* Background orbs */}
      <div className={styles.orb1} />
      <div className={styles.orb2} />

      {/* ── Header ── */}
      <header className={styles.header}>
        <div className={styles.logo}>
          voice<span className={styles.logoAccent}>AI</span>
        </div>
        <StatusBar status={status} />
      </header>

      {/* ── Conversation ── */}
      <main className={styles.main}>
        <MessageList messages={messages} streamingText={streamingText} />
      </main>

      {/* ── Controls ── */}
      <footer className={styles.footer}>

        {/* Visualiser */}
        <div className={styles.vizWrap}>
          <AudioVisualiser getAnalyser={getAnalyser} active={isListening} />
        </div>

        {/* Mic button */}
        <MicButton
          onStart={startListening}
          onStop={stopListening}
          active={isListening}
          disabled={isBusy || isDisconnected}
        />

        {/* Hint text */}
        <p className={styles.hint}>
          {isListening  ? "release when done speaking"  :
           isBusy       ? status + "…"                  :
           isDisconnected ? "not connected"             :
           "hold to speak"}
        </p>

        {/* Action row */}
        <div className={styles.actionRow}>
          <button
            className={styles.actionBtn}
            onClick={resetChat}
            disabled={isDisconnected}
          >
            ↺ reset
          </button>
          <button
            className={styles.actionBtn}
            onClick={connect}
          >
            {isDisconnected ? "⟳ connect" : "⟳ reconnect"}
          </button>
        </div>
      </footer>

    </div>
  );
}
