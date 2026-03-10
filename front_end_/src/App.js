import { useState, useCallback } from 'react';
import './App.css';
import Sidebar from './compo/header/Header';
import Main from './compo/main/main';

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [conversations, setConversations] = useState([]); 
  const [chatKey, setChatKey] = useState(0);

  const handleToggleSidebar = useCallback(() => {
    setSidebarOpen((prev) => !prev);
  }, []);

  const handleNewChat = useCallback(() => {
    setSidebarOpen(false);
    setChatKey(prev => prev + 1); // Force remount of Main component
  }, []);

  const handleConversationStart = useCallback((title) => {
    // Add new conversation to history securely
    setConversations(prev => [
      { id: Date.now(), label: title, time: 'Today' },
      ...prev
    ]);
  }, []);

  return (
    <div className="App">
      <Sidebar
        history={conversations}
        onNewChat={handleNewChat}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />
      <Main
        key={chatKey}
        onToggleSidebar={handleToggleSidebar}
        onConversationStart={handleConversationStart}
      />
    </div>
  );
}

export default App;
