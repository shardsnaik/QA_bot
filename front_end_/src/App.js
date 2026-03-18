import { useState, useCallback, useEffect } from 'react';
import './App.css';
import Sidebar from './compo/header/Header';
import Main from './compo/main/main';
import DemoPopup from './compo/DemoPopup';

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  
  // Load initial conversations from localStorage
  const [conversations, setConversations] = useState(() => {
    const saved = localStorage.getItem('qa_bot_conversations');
    return saved ? JSON.parse(saved) : [];
  });
  
  const [activeChatId, setActiveChatId] = useState(null);
  const [chatKey, setChatKey] = useState(0);

  // Sync conversations to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem('qa_bot_conversations', JSON.stringify(conversations));
  }, [conversations]);

  // Show popup once per browser session
  const [showPopup, setShowPopup] = useState(
    () => sessionStorage.getItem('demo_popup_dismissed') !== 'true'
  );

  const handleToggleSidebar = useCallback(() => {
    setSidebarOpen((prev) => !prev);
  }, []);

  const handleNewChat = useCallback(() => {
    setSidebarOpen(false);
    setActiveChatId(null);
    setChatKey(prev => prev + 1);
  }, []);

  const handleSelectConversation = useCallback((id) => {
    setActiveChatId(id);
    setChatKey(id); // Use ID as key to force remount of Main with that chat's data
  }, []);

  const handleConversationStart = useCallback((title) => {
    const newId = Date.now();
    setConversations(prev => [
      { id: newId, label: title, time: 'Today', messages: [] },
      ...prev
    ]);
    setActiveChatId(newId);
  }, []);

  const handleMessagesUpdate = useCallback((newMessages) => {
    if (!activeChatId) return;
    setConversations(prev => prev.map(conv => 
      conv.id === activeChatId ? { ...conv, messages: newMessages } : conv
    ));
  }, [activeChatId]);

  const handlePopupContinue = useCallback(() => {
    sessionStorage.setItem('demo_popup_dismissed', 'true');
    setShowPopup(false);
  }, []);

  const handlePopupCancel = useCallback(() => {
    window.close();
    setShowPopup(false);
  }, []);

  const activeConversation = conversations.find(c => c.id === activeChatId);

  return (
    <div className="App">
      {showPopup && (
        <DemoPopup
          onContinue={handlePopupContinue}
          onCancel={handlePopupCancel}
        />
      )}
      <Sidebar
        history={conversations}
        onNewChat={handleNewChat}
        onSelect={handleSelectConversation}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />
      <Main
        key={chatKey}
        onToggleSidebar={handleToggleSidebar}
        onConversationStart={handleConversationStart}
        initialMessages={activeConversation?.messages || []}
        onMessagesUpdate={handleMessagesUpdate}
      />
    </div>
  );
}

export default App;
