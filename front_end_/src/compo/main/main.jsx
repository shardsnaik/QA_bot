// filepath: /C:/Users/Public/Gen_AI/New folder (2)/QA_bot/front_end_/src/compo/main/main.jsx
import React, { useState } from 'react';
import './main.css'; // Assuming you have a CSS file for styling
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faPlay } from '@fortawesome/free-solid-svg-icons'

const Main = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [pdfname, setpdfname] = useState('No file uploaded')

  const handleInputChange = (e) => {
    setInput(e.target.value);
  };

  const handleSendMessage = async () => {
    if (input.trim() === '') return;

    const newMessage = { text: input, sender: 'user' };
    setMessages([...messages, newMessage]);
    // https://qa-bot-ijyw.onrender.com/chat
    try {
      const response = await fetch('https://qa-bot-ijyw.onrender.com/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: input }), // Ensure the payload matches the expected format
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      const botMessage = { text: data.answer, sender: 'bot' };
      setMessages((prevMessages) => [...prevMessages, botMessage]);
      console.log(data);
    } catch (error) {
      console.error('Error communicating with the API:', error);
    }

    setInput('');
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    console.log(file);
    if (file){
      setpdfname(file.name);
    }
  }
  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }}

  return (

    <div className='centr-con' >
    <div className="main-container">
      <div className="upload-section">
        <div className="pdf-name">{pdfname}</div>
        <div className="upload-button">

        <input type="file" accept='.pdf' onChange={handleFileUpload} style={{opacity:0, position: 'absolute'}} id='file-upload' />
         <button  className='upl-btn' >Upload</button>
        </div>
      </div>

      <div className="chat-section">
        <div className={pdfname == 'No file uploaded' ? 'temp-header' : 'chat-header'}>RAG Model Integrated with Chat-Gpt and {pdfname.length > 10 ? `${pdfname.substring(0, 10)}...` : pdfname}</div>

        <div className="chat-messages">
          {messages.map((message, index) => (
            <div key={index} className={`message-${message.sender}`}>
              {message.text}
            </div>
          ))}
        </div>
        <div className="message-input">
          <input
            type="text"
            placeholder="Type your message..."
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
          />
  
          <div className='send-icon' onClick={handleSendMessage} >
          <FontAwesomeIcon icon={faPlay} size="2xl" />
          </div>
        </div>
      </div>
      </div>
    </div>
  );
};

export default Main;