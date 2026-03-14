// components/AudioVisualiser.jsx
// Animated frequency bars using requestAnimationFrame + canvas.

import { useRef, useEffect } from "react";
import styles from "./AudioVisualiser.module.css";

const BAR_COUNT = 28;

export default function AudioVisualiser({ getAnalyser, active }) {
  const canvasRef = useRef(null);
  const rafRef    = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    function draw() {
      const analyser  = getAnalyser();
      const W         = canvas.width;
      const H         = canvas.height;
      ctx.clearRect(0, 0, W, H);

      if (!analyser || !active) {
        // Idle sine wave
        const t     = Date.now() / 1000;
        const barW  = W / BAR_COUNT;
        for (let i = 0; i < BAR_COUNT; i++) {
          const h = 4 + Math.abs(Math.sin((i / BAR_COUNT) * Math.PI * 2 + t * 1.2)) * 8;
          const x = i * barW + barW * 0.2;
          ctx.fillStyle = "rgba(139,120,255,0.25)";
          ctx.beginPath();
          ctx.roundRect(x, (H - h) / 2, barW * 0.6, h, 3);
          ctx.fill();
        }
      } else {
        const data = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(data);
        const step = Math.floor(data.length / BAR_COUNT);
        const barW = W / BAR_COUNT;

        for (let i = 0; i < BAR_COUNT; i++) {
          const val = data[i * step] || 0;
          const h   = 4 + (val / 255) * (H - 8);
          const x   = i * barW + barW * 0.15;

          // Gradient bar
          const grad = ctx.createLinearGradient(0, (H - h) / 2, 0, (H + h) / 2);
          grad.addColorStop(0,   "rgba(139,120,255,0.9)");
          grad.addColorStop(0.5, "rgba(247,106,138,0.85)");
          grad.addColorStop(1,   "rgba(139,120,255,0.4)");

          ctx.fillStyle = grad;
          ctx.beginPath();
          ctx.roundRect(x, (H - h) / 2, barW * 0.7, h, 4);
          ctx.fill();
        }
      }

      rafRef.current = requestAnimationFrame(draw);
    }

    draw();
    return () => cancelAnimationFrame(rafRef.current);
  }, [getAnalyser, active]);

  // DPI-aware canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr  = window.devicePixelRatio || 1;
    const rect  = canvas.getBoundingClientRect();
    canvas.width  = rect.width  * dpr;
    canvas.height = rect.height * dpr;
    canvas.getContext("2d").scale(dpr, dpr);
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className={styles.canvas}
      aria-hidden="true"
    />
  );
}
