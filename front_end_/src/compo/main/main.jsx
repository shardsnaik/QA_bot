import React, { useState, useRef, useEffect } from 'react';
import VoiceApp from '../VoiceApp/VoiceApp';
import './main.css';
import { ToastContainer, toast, Bounce } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

const CHAT_URL   = process.env.REACT_APP_CHAT_URL   || 'http://localhost:6001/api/v1';
const VISION_URL = process.env.REACT_APP_VISION_URL || 'http://localhost:6002/api/v1';
const VOICE_URL  = process.env.REACT_APP_VOICE_URL  || 'http://localhost:6003/api/v1';

// ─── Avatar ─────────────────────────────────────────────────────────────
const Avatar = ({ role }) => {
  if (role === 'user') {
    return <div className="avatar avatar-user">U</div>;
  }
  return <div className ="avatar avatar-bot">🤖</div>;
};

// ─── Message Bubble ──────────────────────────────────────────────────────
const MessageBubble = ({ msg }) => {
  const isUser = msg.sender === 'user';
  return (
    <div className={`message-row ${isUser ? 'message-row-user' : 'message-row-bot'}`}>
      {!isUser && <Avatar role="bot" />}
      <div className={`bubble ${isUser ? 'bubble-user' : 'bubble-bot'}`}>
        {msg.file && (
          <div className="bubble-file-preview">
            {msg.fileType?.startsWith('image/') ? (
              <img src={msg.file} alt="uploaded" className="bubble-image" />
            ) : (
              <div className="bubble-file-chip">
                <span className="bubble-file-icon">📎</span>
                <span>{msg.fileName}</span>
              </div>
            )}
          </div>
        )}
        <div className="bubble-text">{msg.text}</div>
        {msg.sources && msg.sources.length > 0 && (
          <div className="bubble-sources">
            <div className="sources-label">Sources</div>
            {msg.sources.map((s, i) => (
              <div key={i} className="source-chip">
                {typeof s === 'string' ? s : s.doc_id || 'Unknown source'}
              </div>
            ))}
          </div>
        )}
        {msg.meta && (
          <div className="bubble-meta">{msg.meta}</div>
        )}
      </div>
      {isUser && <Avatar role="user" />}
    </div>
  );
};

// ─── Typing Indicator ────────────────────────────────────────────────────
const TypingIndicator = () => (
  <div className="message-row message-row-bot">
    <Avatar role="bot" />
    <div className="bubble bubble-bot typing-bubble">
      <span /><span /><span />
    </div>
  </div>
);

// ─── Welcome Screen ──────────────────────────────────────────────────────
const WelcomeScreen = ({ onSuggestion }) => {
  const suggestions = [
    'Summarise the uploaded documents',
    'What are the key takeaways?',
    'Explain in simple terms',
    'Give me 5 bullet points',
  ];

  return (
    <div className="welcome">
      <div className="welcome-icon">🤖</div>
      <h2 className="welcome-title">How can I help you today?</h2>
      <div className="suggestions-grid">
        {suggestions.map((s, i) => (
          <button key={i} className="suggestion-card" onClick={() => onSuggestion(s)}>
            {s}
          </button>
        ))}
      </div>
    </div>
  );
};

// ─── Main Component ──────────────────────────────────────────────────────
const Main = ({ onToggleSidebar, onConversationStart }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [attachedFile, setAttachedFile] = useState(null);
  const [isTyping, setIsTyping] = useState(false);
  const [showAttachMenu, setShowAttachMenu] = useState(false);
  const [voiceMode, setVoiceMode] = useState(false);

  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  // Handle clicks outside the attach menu to close it
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (showAttachMenu && !e.target.closest('.attach-menu-container')) {
        setShowAttachMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showAttachMenu]);

  // Auto-grow textarea
  const autoGrow = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 180) + 'px';
  };

  const handleInputChange = (e) => {
    setInput(e.target.value);
    autoGrow();
  };

  const handleSuggestion = (text) => {
    setInput(text);
    textareaRef.current?.focus();
  };

  // ── File Handlers ──────────────────────────────────────────────────────
  const triggerFileSelect = (modeKey, acceptFilter) => {
    // Store the intended mode so onChange knows what to do
    fileInputRef.current.dataset.mode = modeKey;
    fileInputRef.current.accept = acceptFilter;
    fileInputRef.current.click();
    setShowAttachMenu(false);
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const internalMode = e.target.dataset.mode || 'upload';
    const previewUrl = file.type.startsWith('image/') ? URL.createObjectURL(file) : null;
    
    setAttachedFile({ 
      file, 
      name: file.name, 
      type: file.type, 
      previewUrl,
      internalMode 
    });
    
    e.target.value = '';
    toast.info(`📎 Attached: ${file.name}`, {
      position: 'top-center',
      autoClose: 2500,
      theme: 'dark',
      transition: Bounce,
    });
    
    textareaRef.current?.focus();
  };

  const removeFile = () => {
    if (attachedFile?.previewUrl) URL.revokeObjectURL(attachedFile.previewUrl);
    setAttachedFile(null);
  };

  // ── Send / submit ──────────────────────────────────────────────────────
  const handleSend = async () => {
    const text = input.trim();
    if (!text && !attachedFile) return;

    // Use internalMode if file is attached, else standard chat
    const sendingMode = attachedFile ? attachedFile.internalMode : 'chat';
    const currentAttachment = attachedFile; // snapshot for the async call
    
    // Trigger history event if this is the first message
    if (messages.length === 0 && onConversationStart) {
      const title = text ? text.slice(0, 30) : `Shared ${currentAttachment.name}`;
      onConversationStart(title);
    }

    // Build user message
    const userMsg = {
      id: Date.now(),
      sender: 'user',
      text: text || '',
      file: currentAttachment?.previewUrl || null,
      fileName: currentAttachment?.name || null,
      fileType: currentAttachment?.type || null,
    };
    
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setAttachedFile(null); // Clear input bar immediately
    setShowAttachMenu(false);
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
    setIsTyping(true);

    try {
      let data;

      if (sendingMode === 'chat') {
        const res = await fetch(`${CHAT_URL}/chat-direct`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text }),
        });
        if (!res.ok) throw new Error(`Server error ${res.status}`);
        data = await res.json();
        setMessages((prev) => [
          ...prev,
          {
            id: Date.now() + 1,
            sender: 'bot',
            text: data.answer || 'No answer returned.',
            sources: data.sources || [],
          },
        ]);

      } else if (sendingMode === 'vision') {
        const form = new FormData();
        form.append('file', currentAttachment.file);
        form.append('message', text || 'Describe this image in detail.');
        const res = await fetch(`${VISION_URL}/vision`, { method: 'POST', body: form });
        if (!res.ok) throw new Error(`Vision error ${res.status}`);
        data = await res.json();
        setMessages((prev) => [
          ...prev,
          {
            id: Date.now() + 1,
            sender: 'bot',
            text: data.answer || 'No description returned.',
            meta: `Model: ${data.model}`,
          },
        ]);

      } else if (sendingMode === 'voice') {
        const form = new FormData();
        form.append('file', currentAttachment.file);
        form.append('message', text || '');
        const res = await fetch(`${VOICE_URL}/voice`, { method: 'POST', body: form });
        if (!res.ok) throw new Error(`Voice error ${res.status}`);
        data = await res.json();
        const resultText = data.result || data.transcript || 'No result returned.';
        setMessages((prev) => [
          ...prev,
          {
            id: Date.now() + 1,
            sender: 'bot',
            text: resultText,
            meta: `Task: ${data.task} · Model: ${data.model}`,
          },
        ]);

      } else if (sendingMode === 'upload') {
        toast.info('📤 Uploading file…', {
          position: 'top-center',
          autoClose: 8000,
          theme: 'dark',
          transition: Bounce,
        });
        const form = new FormData();
        form.append('file', currentAttachment.file);
        if (text) form.append('message', text);
        const res = await fetch(`${CHAT_URL}/upload-direct`, { method: 'POST', body: form });
        if (!res.ok) throw new Error(`Upload error ${res.status}`);
        data = await res.json();
        toast.success('✅ File uploaded successfully!', {
          position: 'top-center',
          autoClose: 3000,
          theme: 'dark',
          transition: Bounce,
        });
        const replyText = data.data?.answer
          || data.data?.message
          || `✅ "${currentAttachment.name}" has been uploaded and ingested successfully.`;
        setMessages((prev) => [
          ...prev,
          { id: Date.now() + 1, sender: 'bot', text: replyText },
        ]);
      }

    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 2,
          sender: 'bot',
          text: `⚠️ ${err.message || 'Something went wrong. Please try again.'}`,
        },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const showWelcome = messages.length === 0 && !isTyping;

  return (
    <div className="chat-shell">
      <ToastContainer />

      {/* ── Top bar ── */}
      <div className="chat-topbar">
        <button className="topbar-menu-btn" onClick={onToggleSidebar}>☰</button>
        <div className="topbar-spacer" />
        <button
          className={`voice-toggle-btn${voiceMode ? ' voice-toggle-active' : ''}`}
          onClick={() => setVoiceMode(v => !v)}
          title={voiceMode ? 'Switch to text chat' : 'Switch to voice chat'}
        >
          🎙️ {voiceMode ? 'Text' : 'Voice'}
        </button>
      </div>

      {/* ── Voice mode — full VoiceApp ── */}
      {voiceMode ? (
        <div className="voice-mode-wrapper">
          <VoiceApp />
        </div>
      ) : showWelcome ? (
        <WelcomeScreen onSuggestion={handleSuggestion} />
      ) : (
        <div className="messages-list">
          {messages.map((msg) => (
            <MessageBubble key={msg.id} msg={msg} />
          ))}
          {isTyping && <TypingIndicator />}
          <div ref={messagesEndRef} />
        </div>
      )}

      {/* ── Input area — hidden in voice mode ── */}
      {!voiceMode && (
        <div className="input-area-wrapper">
        <div className="input-box">
          {/* Attached file preview */}
          {attachedFile && (
            <div className="attached-preview">
              {attachedFile.previewUrl ? (
                <img src={attachedFile.previewUrl} alt="preview" className="attached-thumb" />
              ) : (
                <span className="attached-chip">📎 {attachedFile.name}</span>
              )}
              <button className="attached-remove" onClick={removeFile} title="Remove">✕</button>
            </div>
          )}

          <div className="input-row">
            {/* Hidden file input */}
            <input
              ref={fileInputRef}
              type="file"
              onChange={handleFileChange}
              style={{ display: 'none' }}
              id="global-file-input"
            />
            
            {/* Plus attachment button and popup menu */}
            <div className="attach-menu-container">
              <button
                className={`attach-plus-btn ${showAttachMenu ? 'active' : ''}`}
                onClick={() => setShowAttachMenu((prev) => !prev)}
                title="Attach file"
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="12" y1="5" x2="12" y2="19"></line>
                  <line x1="5" y1="12" x2="19" y2="12"></line>
                </svg>
              </button>
              
              {showAttachMenu && (
                <div className="attach-dropdown">
                  <button className="attach-option" onClick={() => triggerFileSelect('upload', '.pdf,text/plain,text/markdown,text/csv,application/pdf')}>
                    <span className="opt-icon">📂</span> Upload Document
                  </button>
                  <button className="attach-option" onClick={() => triggerFileSelect('vision', 'image/*')}>
                    <span className="opt-icon">🖼️</span> Upload Image
                  </button>
                  <button className="attach-option" onClick={() => triggerFileSelect('voice', 'audio/*')}>
                    <span className="opt-icon">🎙️</span> Upload Audio
                  </button>
                </div>
              )}
            </div>

            {/* Text input */}
            <textarea
              ref={textareaRef}
              className="input-textarea"
              placeholder="Message QA Bot..."
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              rows={1}
            />

            {/* Send button */}
            <button
              className={`send-btn${(input.trim() || attachedFile) ? ' send-btn-active' : ''}`}
              onClick={handleSend}
              disabled={!input.trim() && !attachedFile}
              aria-label="Send message"
            >
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
              </svg>
            </button>
          </div>
        </div>
        <p className="input-hint">
          QA Bot uses RAG, Vision, and Audio AI. Press <kbd>Enter</kbd> to send.
        </p>
        </div>
      )}
    </div>
  );
};

export default Main;