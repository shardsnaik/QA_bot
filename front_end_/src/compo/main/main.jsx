// filepath: /C:/Users/Public/Gen_AI/New folder (2)/QA_bot/front_end_/src/compo/main/main.jsx
import React, { useState } from 'react';
import './main.css'; // Assuming you have a CSS file for styling
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faPlay } from '@fortawesome/free-solid-svg-icons'



import {ToastContainer, toast, Bounce } from 'react-toastify'
import 'react-toastify/dist/ReactToastify.css';


const Main = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [pdfname, setpdfname] = useState('No file uploaded')
  const [data_available, setdata_available] = useState(false)

  const [isProcessing, setIsProcessing] = useState(false); // New state for processing

  
const tostload =()=>{
  toast.info('PDF Uploading....', {
    position: "top-center",
    autoClose: 10000,
    hideProgressBar: false,
    closeOnClick: true,
    pauseOnHover: true,
    draggable: true,
    progress: undefined,
    theme: "colored",
    transition: Bounce,
    } )
}

const tostmes =()=>{
  toast.success('PDF Uploaded Successfully', {
    position: "top-center",
    autoClose: 3000,
    hideProgressBar: false,
    closeOnClick: true,
    pauseOnHover: true,
    draggable: true,
    progress: undefined,
    theme: "colored",
    transition: Bounce,
    } )
}

  const handleInputChange = (e) => {
    setInput(e.target.value);
  };

  const handleSendMessage = async () => {
    if (input.trim() === '') return;

    const newMessage = { text: input, sender: 'user' };
    setMessages([...messages, newMessage]);
    // https://qa-bot-ijyw.onrender.com/chat
    // https://qa-bot-wfjj.onrender.com  working 
    try {
      const response = await fetch('http://127.0.0.1:8000/chat', {
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

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    console.log(file);
    if (file){
      setpdfname(file.name);
      tostload()
      setIsProcessing(true)

      const formData = new FormData();
      formData.append('file', file);

      try {

        const res = await fetch('http://127.0.0.1:8000/upload', {
          method: 'POST',
          body: formData,
        });
        console.log(res);
        
        if (!res.ok){
          throw new Error('Network response was not ok');
        }
       

        const data = await res.json();
        // if (!data){
        //   tostprocessing()
        // }
  
        console.log('pdf uploaded', data);

        setdata_available(true)
        setIsProcessing(false)
        tostmes()

      }
      catch (error){
        console.error('Error communicating with the API:', error);  
    }}
  }
  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }}

  return (

    <div className='centr-con' >
      <ToastContainer />

    <div className="main-container">
      <div className="upload-section">
        <div className="pdf-name">{pdfname}</div>
        <div className="upload-button">

        <input type="file" accept='.pdf' onChange={handleFileUpload} style={{opacity:0, position: 'absolute'}} id='file-upload' />
         <button  className='upl-btn' >Upload</button>
        </div>
      </div>

      <div className="chat-section">
        {
          isProcessing ? (
            <div className='temp-header'>✨ Processing ✨

            (Just a moment, please! ⏳)
            
            I'm working on your request. 😊
            
            
            </div>
          ):(<div className={!data_available ? 'temp-heade' : 'chat-header'}>RAG Model Integrated with Chat-Gpt and {pdfname.length > 10 ? `${pdfname.substring(0, 10)}...` : pdfname}</div>)
        }
        

        <div className="chat-messages">
          {messages.map((message, index) => (
            <div key={index} className={`message-${message.sender}`}>
              {message.text}
            </div>
          ))}
        </div>
      </div>
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
  );
};

export default Main;