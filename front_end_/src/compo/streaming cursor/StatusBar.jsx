// components/StatusBar.jsx
// Shows connection state and current pipeline stage.

import styles from "./StatusBar.module.css";

const STATUS_MAP = {
  disconnected: { label: "disconnected", cls: "off"        },
  connecting:   { label: "connecting…",  cls: "off"        },
  idle:         { label: "ready",        cls: "ready"      },
  listening:    { label: "listening",    cls: "listening"  },
  processing:   { label: "transcribing", cls: "processing" },
  responding:   { label: "responding",   cls: "responding" },
  speaking:     { label: "speaking",     cls: "speaking"   },
};

export default function StatusBar({ status }) {
  const { label, cls } = STATUS_MAP[status] || STATUS_MAP.disconnected;
  return (
    <div className={`${styles.pill} ${styles[cls]}`}>
      <span className={styles.dot} />
      <span className={styles.label}>{label}</span>
    </div>
  );
}
