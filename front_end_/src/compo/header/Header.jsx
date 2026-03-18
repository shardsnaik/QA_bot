import React from 'react';
import './header.css';

const Sidebar = ({ history = [], onNewChat, onSelect, isOpen, onClose }) => {
  const todayHistory = history.filter(h => h.time === 'Today');
  const yesterdayHistory = history.filter(h => h.time === 'Yesterday');
  const olderHistory = history.filter(h => h.time === 'Previous 7 Days');

  const handleSelect = (id) => {
    if (onSelect) onSelect(id);
    if (isOpen) onClose();
  };

  return (
    <>
      {isOpen && <div className="sidebar-overlay" onClick={onClose} />}
      <aside className={`sidebar${isOpen ? ' open' : ''}`}>
        {/* Brand */}
        <div className="sidebar-brand">
          <div className="sidebar-brand-icon">🤖</div>
          <div className="sidebar-brand-text">
            <h2>QA Bot</h2>
            <span>Multimodal AI Assistant</span>
          </div>
        </div>

        {/* New Chat */}
        <button className="new-chat-btn" onClick={() => {
          onNewChat();
          if (isOpen) onClose();
        }}>
          <span>✦</span> New Conversation
        </button>

        {/* Chat History Section */}
        <div className="sidebar-scroll-area">
          {todayHistory.length > 0 && (
            <>
              <div className="sidebar-section-label">Today</div>
              <nav className="sidebar-nav">
                {todayHistory.map((item) => (
                  <button key={item.id} className="history-item" onClick={() => handleSelect(item.id)}>
                    <span className="history-icon">💬</span>
                    <span className="history-label">{item.label}</span>
                  </button>
                ))}
              </nav>
            </>
          )}

          {yesterdayHistory.length > 0 && (
            <>
              <div className="sidebar-section-label" style={{ marginTop: '12px' }}>Yesterday</div>
              <nav className="sidebar-nav">
                {yesterdayHistory.map((item) => (
                  <button key={item.id} className="history-item" onClick={() => handleSelect(item.id)}>
                    <span className="history-icon">💬</span>
                    <span className="history-label">{item.label}</span>
                  </button>
                ))}
              </nav>
            </>
          )}

          {olderHistory.length > 0 && (
            <>
              <div className="sidebar-section-label" style={{ marginTop: '12px' }}>Previous 7 Days</div>
              <nav className="sidebar-nav">
                {olderHistory.map((item) => (
                  <button key={item.id} className="history-item" onClick={() => handleSelect(item.id)}>
                    <span className="history-icon">💬</span>
                    <span className="history-label">{item.label}</span>
                  </button>
                ))}
              </nav>
            </>
          )}
        </div>

        {/* Footer */}
        <div className="sidebar-footer">
          <a
            href="https://github.com/shardsnaik"
            target="_blank"
            rel="noopener noreferrer"
            className="sidebar-footer-link"
          >
            <span>⚙</span> GitHub Repos
          </a>
        </div>
      </aside>
    </>
  );
};

export default Sidebar;
