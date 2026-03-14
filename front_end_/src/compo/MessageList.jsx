// components/MessageList.jsx
// Renders conversation history + live streaming AI bubble.

import { useEffect, useRef } from "react";
import styles from "./streaming cursor/MessageList.module.css";

export default function MessageList({ messages, streamingText }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingText]);

  const isEmpty = messages.length === 0 && !streamingText;

  return (
    <div className={styles.list}>
      {isEmpty && (
        <div className={styles.empty}>
          <span className={styles.emptyIcon}>◎</span>
          <p>Hold the button and speak.<br />Release when you're done.</p>
        </div>
      )}

      {messages.map((msg, i) => (
        <div
          key={i}
          className={`${styles.row} ${msg.role === "user" ? styles.user : styles.ai}`}
        >
          <span className={styles.label}>{msg.role === "user" ? "you" : "ai"}</span>
          <div className={styles.bubble}>{msg.text}</div>
        </div>
      ))}

      {streamingText && (
        <div className={`${styles.row} ${styles.ai}`}>
          <span className={styles.label}>ai</span>
          <div className={`${styles.bubble} ${styles.streaming}`}>
            {streamingText}
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
