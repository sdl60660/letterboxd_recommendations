import hashlib
import json

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from rq import Queue
from rq.exceptions import NoSuchJobError
from rq.job import Job
from rq.registry import DeferredJobRegistry

from handle_recs import (
    build_client_model,
    get_client_user_data,
    get_client_user_data_from_upload,
)
from worker import conn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost(:\d+)?|letterboxd(\.samlearner\.com)?|.*herokuapp\.com)$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

queue_pool = [Queue(channel, connection=conn) for channel in ["high", "default", "low"]]
popularity_thresholds_500k_samples = [2500, 2000, 1500, 1000, 700, 400, 250, 150]

USERDATA_CACHE_TTL = 300
USERDATA_TTL_BUFFER = 10

# Cap on the uploaded CSV bundle (JSON of all export CSVs). Even a power-user's
# full bundle is only a few MB; this leaves generous headroom.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


async def _read_capped_body(request: Request, max_bytes: int) -> bytes | None:
    """Read the request body, returning None if it exceeds max_bytes. Streams so
    an oversized payload is never fully buffered (the length check on a fully
    buffered body would be too late to prevent a memory blow-up)."""
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > max_bytes:
        return None

    chunks = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > max_bytes:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _ttl_for_job(job: Job) -> int | None:
    """
    Return remaining TTL in seconds for this job's Redis key.
    - >0 : seconds until expiry
    - -1 : key exists with no expiry (e.g., queued/running or no result_ttl)
    - -2 : key doesn't exist (expired)
    Normalize to None for -2 (missing).
    """
    t = conn.ttl(job.key)
    if t == -2:
        return None
    return t


# A direct link to the heroku site will redirect to new domain
# Should take care of stale link issue
@app.get("/", response_class=HTMLResponse)
def homepage():
    return RedirectResponse("https://letterboxd.samlearner.com")
    # return templates.TemplateResponse("index.html", {})


@app.get("/health")
def health():
    return {"ok": True}


def _enqueue_recs_pipeline(
    *,
    user_job_id,
    user_job_callable,
    user_job_args,
    user_job_description,
    model_username,
    training_data_size,
    num_items=2000,
):
    """Shared enqueue + cache logic for the user-data job followed by the
    model-building job. Used by both the username (scrape) and upload flows."""
    ordered_queues = sorted(
        queue_pool, key=lambda queue: DeferredJobRegistry(queue=queue).count
    )

    print([(q, DeferredJobRegistry(queue=q).count) for q in ordered_queues])
    q = ordered_queues[0]

    reused_cache = False
    job_get_user_data = None
    ttl_remaining = None

    try:
        job = Job.fetch(user_job_id, connection=conn)

        ttl_remaining = conn.ttl(job.key)
        reusable = (
            job.is_finished
            and job.result is not None
            and (
                ttl_remaining == -1
                or ttl_remaining is not None
                and ttl_remaining > USERDATA_TTL_BUFFER
            )
        )

        if reusable:
            # Top up the TTL to avoid close-call expiry
            if ttl_remaining not in (-1, None) and ttl_remaining <= (
                USERDATA_TTL_BUFFER + 5
            ):
                conn.expire(job.key, USERDATA_TTL_BUFFER + 30)
                ttl_remaining = conn.ttl(job.key)
            job_get_user_data = job
            reused_cache = True

    except NoSuchJobError:
        pass

    if job_get_user_data is None:
        job_get_user_data = q.enqueue(
            user_job_callable,
            args=user_job_args,
            job_id=user_job_id,
            description=user_job_description,
            result_ttl=120,
            ttl=200,
        )

        # will set to -1, I think, as job won't be finished. can then ingest that on the frontend and use total cache time val
        ttl_remaining = _ttl_for_job(job_get_user_data)

    job_build_model = q.enqueue(
        build_client_model,
        args=(
            model_username,
            training_data_size,
            num_items,
        ),
        depends_on=job_get_user_data,
        description=f"Building model for {model_username} (sample: {training_data_size})",
        result_ttl=45,
        ttl=200,
    )

    job_build_model.meta.update(
        {
            "reused_cache": reused_cache,
            "user_job_id": job_get_user_data.id,
            "user_cache_ttl_at_enqueue": conn.ttl(
                job_get_user_data.key
            ),  # may be -1 or >0
            "userdata_result_ttl": USERDATA_CACHE_TTL,
        }
    )
    job_build_model.save()

    return JSONResponse(
        {
            "redis_get_user_data_job_id": job_get_user_data.id,
            "redis_build_model_job_id": job_build_model.id,
            "user_data_cache": {
                "reused_cache": reused_cache,
                "cached_data_ttl": ttl_remaining,
                "total_cache_time_seconds": USERDATA_CACHE_TTL,
            },
        }
    )


@app.get("/get_recs")
def get_recs(username: str, training_data_size: int, data_opt_in: bool):
    username = username.strip().lower()
    # popularity_threshold = None

    # set deterministic ID for user-data job
    user_job_id = f"username-{username}__optin-{int(bool(data_opt_in))}"

    return _enqueue_recs_pipeline(
        user_job_id=user_job_id,
        user_job_callable=get_client_user_data,
        user_job_args=(username, data_opt_in),
        user_job_description=(
            f"Scraping user data for {username} "
            f"(sample: {training_data_size}, data_opt_in: {data_opt_in})"
        ),
        model_username=username,
        training_data_size=training_data_size,
    )


@app.post("/get_recs_upload")
async def get_recs_upload(
    request: Request,
    training_data_size: int,
    data_opt_in: bool = False,
):
    """Generate recommendations from an uploaded Letterboxd export.

    The browser unzips the export and POSTs the CSV text as JSON
    ({"files": [{"name", "text"}, ...]}); the server never handles a ZIP.
    """
    raw = await _read_capped_body(request, MAX_UPLOAD_BYTES)
    if raw is None:
        return JSONResponse(
            status_code=413,
            content={"error": "Uploaded data is too large."},
        )
    if not raw:
        return JSONResponse(status_code=400, content={"error": "No data uploaded."})

    try:
        files = json.loads(raw)["files"]
        if not isinstance(files, list):
            raise ValueError
    except (ValueError, KeyError, TypeError):
        return JSONResponse(
            status_code=400, content={"error": "Invalid upload payload."}
        )

    # Deterministic id so re-uploading the same file reuses the cached result.
    content_hash = hashlib.sha1(raw).hexdigest()[:16]
    user_job_id = f"upload-{content_hash}__optin-{int(bool(data_opt_in))}"
    model_username = f"upload-{content_hash}"

    # _enqueue_recs_pipeline does blocking Redis I/O; keep it off the event loop.
    return await run_in_threadpool(
        _enqueue_recs_pipeline,
        user_job_id=user_job_id,
        user_job_callable=get_client_user_data_from_upload,
        user_job_args=(files, data_opt_in),
        user_job_description=(
            f"Parsing uploaded export ({len(files)} files, sample: {training_data_size})"
        ),
        model_username=model_username,
        training_data_size=training_data_size,
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

    user_cache_ttl = None  # default if job missing/expired

    try:
        user_job = Job.fetch(job_ids["redis_get_user_data_job_id"], connection=conn)
        execution_data |= {
            "num_user_ratings": user_job.meta.get("num_user_ratings"),
            "user_watchlist": user_job.meta.get("user_watchlist"),
            "user_status": user_job.meta.get("user_status"),
        }
        user_cache_ttl = _ttl_for_job(user_job)
    except NoSuchJobError:
        pass

    payload = {
        "statuses": job_statuses,
        "execution_data": execution_data,
        "user_data_cache": {
            "reused_cache": bool(end_job.meta.get("reused_cache", False)),
            "cached_data_ttl": user_cache_ttl,
            "total_cache_time_seconds": end_job.meta.get("userdata_result_ttl"),
            "cached_data_ttl_at_enque": end_job.meta.get("user_cache_ttl_at_enqueue"),
            "user_job_id": end_job.meta.get("user_job_id"),
        },
    }

    if end_job.is_finished:
        payload["result"] = end_job.result
        return JSONResponse(
            status_code=200,
            content=payload,
        )
    else:
        return JSONResponse(
            status_code=202,
            content=payload,
        )
