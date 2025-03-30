import React, { useState, useEffect, useRef } from 'react';
import '../styles/ChatInterface.scss';

const ChatInterface = ({ username }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [loading, setLoading] = useState(false);
  const messageEndRef = useRef(null);

  // API endpoint base URL
  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  useEffect(() => {
    // Start a new chat session when the component mounts
    const startSession = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/chat/start_session?user_id=${username}`);
        const data = await response.json();
        setSessionId(data.session_id);
        
        // Add welcome message
        setMessages([
          {
            role: 'assistant',
            content: `Hello! I'm your movie recommendation assistant. I can suggest films based on your Letterboxd ratings and watchlist. What kind of movies are you in the mood for today?`,
            timestamp: Date.now(),
          },
        ]);
      } catch (error) {
        console.error('Error starting chat session:', error);
      }
    };

    if (username) {
      startSession();
    }
  }, [username, API_BASE_URL]);

  useEffect(() => {
    // Scroll to bottom on new messages
    messageEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!input.trim() || !sessionId) return;
    
    // Add user message to the chat
    const userMessage = {
      role: 'user',
      content: input,
      timestamp: Date.now(),
    };
    
    setMessages([...messages, userMessage]);
    setInput('');
    setLoading(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/chat/message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: username,
          message: input,
          session_id: sessionId,
        }),
      });
      
      const data = await response.json();
      
      if (data.success) {
        // Add assistant response to the chat
        setMessages((prevMessages) => [
          ...prevMessages,
          {
            role: 'assistant',
            content: data.message,
            timestamp: Date.now(),
          },
        ]);
      } else {
        // Add error message
        setMessages((prevMessages) => [
          ...prevMessages,
          {
            role: 'assistant',
            content: `Sorry, I encountered an error: ${data.error || 'Unknown error'}`,
            timestamp: Date.now(),
          },
        ]);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      // Add error message
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          role: 'assistant',
          content: 'Sorry, I encountered an error while processing your request. Please try again later.',
          timestamp: Date.now(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  return (
    <div className="chat-interface">
      <div className="chat-container">
        <div className="chat-header">
          <h2>Personal Movie Recommendations</h2>
          <p>Chat with our AI assistant to get tailored movie suggestions</p>
        </div>
        
        <div className="messages-container">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`message ${message.role === 'user' ? 'user-message' : 'assistant-message'}`}
            >
              <div className="message-content">{message.content}</div>
              {message.timestamp && (
                <div className="message-timestamp">{formatTimestamp(message.timestamp)}</div>
              )}
            </div>
          ))}
          {loading && (
            <div className="message assistant-message">
              <div className="message-content loading">
                <span className="dot"></span>
                <span className="dot"></span>
                <span className="dot"></span>
              </div>
            </div>
          )}
          <div ref={messageEndRef} />
        </div>
        
        <form className="input-container" onSubmit={handleSubmit}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask for movie recommendations..."
            disabled={loading || !sessionId}
          />
          <button type="submit" disabled={loading || !input.trim() || !sessionId}>
            {loading ? 'Sending...' : 'Send'}
          </button>
        </form>
      </div>
    </div>
  );
};

export default ChatInterface;