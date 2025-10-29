import os
import json
import requests
from typing import List, Dict, Any, Optional
import time
import pandas as pd
import uuid

# You'll need to add an environment variable for your LLM API key
API_KEY = os.environ.get("OPENAI_API_KEY", "")

# In production, use a proper database instead of in-memory storage
chat_history = {}  # user_id -> list of messages

# Load movie data once
movie_data = pd.read_csv("data_processing/data/movie_data.csv")

def get_movie_details(movie_id):
    """Get details for a specific movie"""
    movie = movie_data[movie_data["id"] == movie_id]
    if len(movie) == 0:
        return None
    
    return movie.iloc[0].to_dict()

def create_chat_session(user_id: str) -> str:
    """Create a new chat session for a user"""
    session_id = str(uuid.uuid4())
    
    if user_id not in chat_history:
        chat_history[user_id] = {}
    
    chat_history[user_id][session_id] = []
    
    return session_id

def get_chat_sessions(user_id: str) -> List[str]:
    """Get all chat sessions for a user"""
    if user_id not in chat_history:
        return []
    
    return list(chat_history[user_id].keys())

def get_chat_history(user_id: str, session_id: str) -> List[Dict[str, Any]]:
    """Get chat history for a specific session"""
    if user_id not in chat_history or session_id not in chat_history[user_id]:
        return []
    
    return chat_history[user_id][session_id]

def add_message(user_id: str, session_id: str, role: str, content: str) -> None:
    """Add a message to the chat history"""
    if user_id not in chat_history:
        chat_history[user_id] = {}
    
    if session_id not in chat_history[user_id]:
        chat_history[user_id][session_id] = []
    
    chat_history[user_id][session_id].append({
        "role": role,
        "content": content,
        "timestamp": time.time()
    })

def get_llm_recommendations(
    user_id: str, 
    session_id: str, 
    user_message: str, 
    user_ratings: Optional[Dict[str, float]] = None, 
    user_watchlist: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Get personalized recommendations using LLM"""
    # Add user message to history
    add_message(user_id, session_id, "user", user_message)
    
    # Create context for the LLM
    context = ""
    if user_ratings:
        context += "User's rated movies:\n"
        # Include top 10 highly-rated and bottom 5 lowest-rated movies for context
        sorted_ratings = sorted(user_ratings.items(), key=lambda x: x[1], reverse=True)
        top_movies = sorted_ratings[:10]
        bottom_movies = sorted_ratings[-5:]
        
        for movie_id, rating in top_movies:
            movie = get_movie_details(movie_id)
            if movie:
                context += f"- {movie['title']} ({movie['year']}): {rating}/5 stars\n"
        
        context += "\nLower rated movies:\n"
        for movie_id, rating in bottom_movies:
            movie = get_movie_details(movie_id)
            if movie:
                context += f"- {movie['title']} ({movie['year']}): {rating}/5 stars\n"
    
    if user_watchlist:
        context += "\nUser's watchlist (movies they want to watch):\n"
        for i, movie_id in enumerate(user_watchlist[:10]):  # Limit to first 10
            movie = get_movie_details(movie_id)
            if movie:
                context += f"- {movie['title']} ({movie['year']})\n"
    
    # Construct the full conversation history
    messages = [
        {"role": "system", "content": f"""You are a movie recommendation assistant that provides personalized recommendations.
Use the user's ratings and watchlist to inform your suggestions, but also consider the user's current request.
Based on their preferences, suggest movies they might enjoy watching.
Here is information about the user's movie preferences:

{context}

When recommending movies, be specific and explain why you think they would enjoy each one.
If possible, connect recommendations to movies they already liked."""},
    ]
    
    # Add the conversation history
    for msg in get_chat_history(user_id, session_id):
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Make the API call to the LLM
    try:
        response = call_llm_api(messages)
        
        # Add assistant response to history
        add_message(user_id, session_id, "assistant", response)
        
        return {
            "success": True,
            "message": response,
            "session_id": session_id
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "session_id": session_id
        }

def call_llm_api(messages: List[Dict[str, str]]) -> str:
    """Call the LLM API with the given messages"""
    # This is a placeholder for an actual API call
    # Replace with your preferred LLM API (OpenAI, Claude, etc.)
    
    if not API_KEY:
        # For development without an API key, return a mock response
        return "I recommend watching 'The Godfather' based on your preferences for drama and highly-rated classics."
    
    # Example using OpenAI API (replace with your preferred LLM API)
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "temperature": 0.7
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code != 200:
        raise Exception(f"API call failed: {response.text}")
    
    return response.json()["choices"][0]["message"]["content"]