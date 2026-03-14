// components/MicButton.jsx
// Push-to-talk mic button with ripple animation.

import styles from "./MicButton.module.css";

export default function MicButton({ onStart, onStop, active, disabled }) {
  return (
    <button
      className={`${styles.btn} ${active ? styles.active : ""}`}
      onMouseDown={onStart}
      onMouseUp={onStop}
      onMouseLeave={() => { if (active) onStop(); }}
      onTouchStart={(e) => { e.preventDefault(); onStart(); }}
      onTouchEnd={(e)   => { e.preventDefault(); onStop(); }}
      disabled={disabled}
      aria-label={active ? "Release to send" : "Hold to speak"}
    >
      <span className={styles.ring} />
      <span className={styles.ring2} />
      <MicIcon active={active} />
    </button>
  );
}

function MicIcon({ active }) {
  return (
    <svg
      className={styles.icon}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <rect
        x="9" y="2" width="6" height="12"
        rx="3"
        fill={active ? "#f76a8a" : "white"}
        style={{ transition: "fill 0.2s" }}
      />
      <path
        d="M5 11C5 15.418 8.134 19 12 19C15.866 19 19 15.418 19 11"
        stroke={active ? "#f76a8a" : "white"}
        strokeWidth="2"
        strokeLinecap="round"
        style={{ transition: "stroke 0.2s" }}
      />
      <line x1="12" y1="19" x2="12" y2="22"
        stroke={active ? "#f76a8a" : "white"}
        strokeWidth="2" strokeLinecap="round"
        style={{ transition: "stroke 0.2s" }}
      />
      <line x1="8" y1="22" x2="16" y2="22"
        stroke={active ? "#f76a8a" : "white"}
        strokeWidth="2" strokeLinecap="round"
        style={{ transition: "stroke 0.2s" }}
      />
    </svg>
  );
}
