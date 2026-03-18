import React from 'react';
import './DemoPopup.css';

function DemoPopup({ onContinue, onCancel }) {
  return (
    <div className="popup-overlay">
      <div className="popup-card">

        {/* Header */}
        <div className="popup-header">
          <span className="popup-icon">🤖</span>
          <h1 className="popup-title">Hold up, human! ✋</h1>
          <p className="popup-subtitle">Your AI is still putting on its shoes.</p>
        </div>

        <div className="popup-divider" />

        {/* Cold Start Notice */}
        <div className="popup-section">
          <h2 className="popup-section-title">⚙️ Demo Environment Notice</h2>
          <p className="popup-text">
            This AI runs on <strong>zero-cost cloud infrastructure</strong> — yes, completely free. 
            Because of that, the backend takes a little nap 😴 when nobody's using it 
            (it's on a <strong>serverless free tier</strong>, it literally sleeps on the job).
          </p>
          <div className="popup-alert">
            <span className="popup-alert-icon">⏰</span>
            <span>
              <strong>Expected startup time:</strong> ~<strong>30–60 seconds</strong> on first load.
              Go grab a ☕ coffee. You've earned it.
            </span>
          </div>
        </div>

        {/* Things you can test */}
        <div className="popup-section">
          <h2 className="popup-section-title">🧪 What Can You Actually Do Here?</h2>
          <ul className="popup-feature-list">
            <li>
              <span className="feature-badge">💬</span>
              <span><strong>Multimodal Chat</strong> — Talk to the AI like a civilized human.</span>
            </li>
            <li>
              <span className="feature-badge">👁️</span>
              <span><strong>Vision Understanding</strong> — Show it a picture, ask weird questions.</span>
            </li>
            <li>
              <span className="feature-badge">🎙️</span>
              <span><strong>Voice Interaction</strong> — Yell at the AI and it'll yell back (politely).</span>
            </li>
            <li>
              <span className="feature-badge">📚</span>
              <span><strong>Knowledge-based Responses</strong> — AI with actual context. Fancy stuff (RAG).</span>
            </li>
          </ul>
        </div>

        {/* Why the delay */}
        <div className="popup-section popup-section--muted">
          <p className="popup-text-small">
            🏦 <strong>Why the wait?</strong> This entire platform runs on <strong>free-tier infrastructure</strong> 
            to prove that a full AI stack can be built with <em>zero operational cost</em>. 
            In production, the server never sleeps. But here? Budget life. 💸
          </p>
        </div>

        {/* CTA */}
        <div className="popup-section">
          <p className="popup-text-small popup-cta-note">
            🚀 If this is your <strong>first request</strong>, give the system up to <strong>1 minute</strong> to wake up. 
            After that, it'll be blazing fast. Promise.
          </p>
        </div>

        {/* Buttons */}
        <div className="popup-actions">
          <button className="popup-btn popup-btn--cancel" onClick={onCancel}>
            ❌ Nope, take me out
          </button>
          <button className="popup-btn popup-btn--continue" onClick={onContinue}>
            🚀 Start Demo — Let's Go!
          </button>
        </div>

      </div>
    </div>
  );
}

export default DemoPopup;
