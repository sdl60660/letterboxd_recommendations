from fastapi import FastAPI, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

from typing import Union, List, Dict, Any, Optional
from pydantic import BaseModel

from urllib.parse import urlparse, urlunparse

import pandas as pd
import pickle

from rq import Queue
from rq.exceptions import NoSuchJobError
from rq.job import Job
from rq.registry import DeferredJobRegistry

from worker import conn

from handle_recs import get_client_user_data, build_client_model
from handle_llm import (
    create_chat_session,
    get_chat_sessions,
    get_chat_history,
    get_llm_recommendations
)


app = FastAPI()

origins = [
    "http://localhost",
    "https://localhost",
    "http://localhost:3000",
    "https://localhost:3000",
    "http://letterboxd-recommendations.herokuapp.com",
    "https://letterboxd-recommendations.herokuapp.com",
    "http://letterboxd.samlearner.com",
    "https://letterboxd.samlearner.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

queue_pool = [Queue(channel, connection=conn) for channel in ["high", "default", "low"]]
popularity_thresholds_500k_samples = [2500, 2000, 1500, 1000, 700, 400, 250, 150]


# Pydantic models for chat API
class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[float] = None


class ChatRequest(BaseModel):
    user_id: str
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None
    session_id: str


# A direct link to the heroku site will redirect to new domain
# Should take care of stale link issue
@app.get("/", response_class=HTMLResponse)
def homepage():
    return RedirectResponse("https://letterboxd.samlearner.com")
    # return templates.TemplateResponse("index.html", {})


@app.get("/get_recs")
def get_recs(
    username: str, training_data_size: int, popularity_filter: int, data_opt_in: bool
):
    if popularity_filter >= 0:
        popularity_threshold = popularity_thresholds_500k_samples[popularity_filter]
    else:
        popularity_threshold = None

    num_items = 2000

    ordered_queues = sorted(
        queue_pool, key=lambda queue: DeferredJobRegistry(queue=queue).count
    )
    print([(q, DeferredJobRegistry(queue=q).count) for q in ordered_queues])
    q = ordered_queues[0]

    job_get_user_data = q.enqueue(
        get_client_user_data,
        args=(
            username,
            data_opt_in,
        ),
        description=f"Scraping user data for {username} (sample: {training_data_size}, popularity_filter: {popularity_threshold}, data_opt_in: {data_opt_in})",
        result_ttl=45,
        ttl=200,
    )
    job_build_model = q.enqueue(
        build_client_model,
        args=(
            username,
            training_data_size,
            popularity_threshold,
            num_items,
        ),
        depends_on=job_get_user_data,
        description=f"Building model for {username} (sample: {training_data_size}, popularity_filter: {popularity_threshold})",
        result_ttl=30,
        ttl=200,
    )

    return JSONResponse(
        {
            "redis_get_user_data_job_id": job_get_user_data.get_id(),
            "redis_build_model_job_id": job_build_model.get_id(),
        }
    )


@app.get("/results")
def get_results(redis_build_model_job_id: str, redis_get_user_data_job_id: str):
    job_ids = {
        "redis_build_model_job_id": redis_build_model_job_id,
        "redis_get_user_data_job_id": redis_get_user_data_job_id,
    }

    job_statuses = {}
    for key, job_id in job_ids.items():
        try:
            job_statuses[key.replace("_id", "_status")] = Job.fetch(
                job_id, connection=conn
            ).get_status()
        except NoSuchJobError:
            job_statuses[key.replace("_id", "_status")] = "finished"

    end_job = Job.fetch(job_ids["redis_build_model_job_id"], connection=conn)
    execution_data = {"build_model_stage": end_job.meta.get("stage")}

    try:
        user_job = Job.fetch(job_ids["redis_get_user_data_job_id"], connection=conn)
        execution_data |= {
            "num_user_ratings": user_job.meta.get("num_user_ratings"),
            "user_watchlist": user_job.meta.get("user_watchlist"),
            "user_status": user_job.meta.get("user_status"),
        }
    except NoSuchJobError:
        pass

    if end_job.is_finished:
        return JSONResponse(
            status_code=200,
            content={
                "statuses": job_statuses,
                "execution_data": execution_data,
                "result": end_job.result,
            },
        )
    else:
        return JSONResponse(
            status_code=202,
            content={"statuses": job_statuses, "execution_data": execution_data},
        )


@app.post("/chat/start_session")
def start_chat_session(user_id: str):
    """Start a new chat session"""
    session_id = create_chat_session(user_id)
    return JSONResponse(
        status_code=200,
        content={"session_id": session_id}
    )


@app.get("/chat/sessions")
def list_chat_sessions(user_id: str):
    """List all chat sessions for a user"""
    sessions = get_chat_sessions(user_id)
    return JSONResponse(
        status_code=200,
        content={"sessions": sessions}
    )


@app.get("/chat/history")
def get_conversation_history(user_id: str, session_id: str):
    """Get the history of a chat session"""
    history = get_chat_history(user_id, session_id)
    return JSONResponse(
        status_code=200,
        content={"history": history}
    )


@app.post("/chat/message", response_model=ChatResponse)
async def send_chat_message(request: ChatRequest):
    """Send a message and get LLM-based recommendations"""
    # If no session_id is provided, create a new one
    session_id = request.session_id
    if not session_id:
        session_id = create_chat_session(request.user_id)
    
    # Get current user ratings and watchlist data if available
    try:
        # Attempt to get the user's data from Redis
        user_data_key = f"user_data:{request.user_id}"
        user_ratings = None
        user_watchlist = None
        
        # Try to retrieve user data from Redis
        user_data_bytes = conn.get(user_data_key)
        if user_data_bytes:
            user_data = pickle.loads(user_data_bytes)
            user_ratings = user_data.get("ratings")
            user_watchlist = user_data.get("watchlist")
        
        response = get_llm_recommendations(
            request.user_id,
            session_id,
            request.message,
            user_ratings,
            user_watchlist
        )
        
        return response
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "session_id": session_id
        }
