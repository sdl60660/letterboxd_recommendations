# (full file content with retry wrapper inserted)
import pickle
import time
import random
from urllib.parse import urlparse, parse_qs

import pandas as pd
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import (
    CollectionInvalid,
    WriteConcernError,
    OperationFailure,
    AutoReconnect,
)

from utils.config import sample_sizes
from utils.db_connect import connect_to_db
from utils.utils import get_rich_movie_data

# --- params (tune these) ---
USER_MIN = 50  # users must have ≥ this many ratings to end up in sampling pool
MOVIE_MIN = 50  # movies must have ≥ this many ratings (this gets adjusted to account for subsampling)
PER_USER_CAP = 300  # at most this many ratings per sampled user

# collection names
ACTIVE_USERS_COLL = "active_users_tmp"
MOVIE_COUNTS_COLL = "movie_counts"
THRESHOLD_MOVIES_COLL = "threshold_movies_tmp"
SAMPLED_USERS_COLL = "sampled_users_tmp"
QUALIFYING_MOVIES_COLL = "qualifying_movies_tmp"
RAW_TRAINING_DATA_SAMPLE_COLL = "training_data_sample_raw"
TRAINING_DATA_SAMPLE_COLL = "training_data_sample"

# --- Retry helper for transient replica / network errors ---------------------
_TRANSIENT_CODES = {11602, 10107, 13435, 189}  # InterruptedDueToReplStateChange etc.


def _get_mongo_error_code(exc):
    """Extract server error code from common PyMongo exception shapes."""
    if getattr(exc, "code", None) is not None:
        return exc.code
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        return details.get("code")
    return None


def run_with_transient_retry(
    fn, *, label, max_attempts=6, base_sleep_s=20.0, max_sleep_s=120.0
):
    """
    Run fn() and retry on transient Mongo errors (replica stepdowns, reconnects).
    Verbose logging + exponential backoff + jitter.

    - fn: zero-arg callable that performs the operation and returns its result (cursor or value)
    - label: human-readable label printed in logs
    - max_attempts: total attempts (including first)
    - base_sleep_s: base sleep for backoff (seconds)
    - max_sleep_s: max cap for backoff (seconds)
    """
    attempt = 1
    while True:
        try:
            if attempt == 1:
                print(f"[{label}] starting")
            else:
                print(f"[{label}] attempt {attempt}/{max_attempts}")
            return fn()
        except (WriteConcernError, OperationFailure, AutoReconnect) as exc:
            code = _get_mongo_error_code(exc)
            code_name = None
            details = getattr(exc, "details", None)
            if isinstance(details, dict):
                code_name = details.get("codeName")
            is_transient = (
                isinstance(exc, AutoReconnect)
                or (code in _TRANSIENT_CODES)
                or (
                    code_name
                    in {
                        "InterruptedDueToReplStateChange",
                        "NotWritablePrimary",
                        "PrimarySteppedDown",
                    }
                )
            )

            # Print concise error info (first part of message)
            print(
                f"[{label}] ERROR: {type(exc).__name__} code={code} codeName={code_name} msg={str(exc)[:300]}"
            )

            if (not is_transient) or attempt >= max_attempts:
                print(
                    f"[{label}] giving up (transient={is_transient}, attempt={attempt})"
                )
                raise

            # Exponential backoff with jitter
            sleep_s = min(max_sleep_s, base_sleep_s * (2 ** (attempt - 1)))
            sleep_s = sleep_s * (
                0.8 + 0.4 * random.random()
            )  # jitter between 0.8x and 1.2x
            print(
                f"[{label}] transient error; sleeping {sleep_s:.1f}s then retrying..."
            )
            time.sleep(sleep_s)
            attempt += 1


# ---------------------------------------------------------------------------


def ensure_empty_collection(db, name, wait_secs=2.0):
    # Drop if present
    if name in db.list_collection_names():
        db[name].drop()
        # wait briefly for namespace to disappear
        deadline = time.time() + wait_secs
        while name in db.list_collection_names() and time.time() < deadline:
            time.sleep(0.1)

    try:
        if name not in db.list_collection_names():
            db.create_collection(name)
    except CollectionInvalid:
        pass
    return db[name]


def get_or_build_collection(db, name, build_fn, use_cache=False, drop_existing=True):
    collections = set(db.list_collection_names())
    if not use_cache or name not in collections:
        if drop_existing and name in collections:
            ensure_empty_collection(db, name)
        coll = build_fn()
        return coll if coll is not None else db[name]
    return db[name]


def filter_active_users(db):
    ratings = db.ratings

    def _agg():
        return ratings.aggregate(
            [
                {"$group": {"_id": "$user_id", "n": {"$sum": 1}}},
                {"$match": {"n": {"$gte": USER_MIN}}},
                {"$project": {"_id": 0, "user_id": "$_id"}},  # one field: user_id
                {"$out": ACTIVE_USERS_COLL},
            ],
            allowDiskUse=True,
        )

    run_with_transient_retry(
        _agg, label="filter_active_users", max_attempts=6, base_sleep_s=20.0
    )

    db[ACTIVE_USERS_COLL].create_index([("user_id", 1)], unique=True)

    return db[ACTIVE_USERS_COLL]


def create_movie_counts(db):
    ratings = db.ratings

    def _agg():
        return ratings.aggregate(
            [
                {"$group": {"_id": "$movie_id", "count": {"$sum": 1}}},
                {
                    "$merge": {
                        "into": MOVIE_COUNTS_COLL,
                        "on": "_id",  # movie_id is in _id from the $group
                        "whenMatched": "replace",  # replace {count} value
                        "whenNotMatched": "insert",
                    }
                },
            ],
            allowDiskUse=True,
            comment="movie_counts_merge",
        )

    run_with_transient_retry(
        _agg, label="create_movie_counts", max_attempts=6, base_sleep_s=20.0
    )

    # (Re)create indexes after the merge completes
    db[MOVIE_COUNTS_COLL].create_index([("count", DESCENDING)])
    db[MOVIE_COUNTS_COLL].create_index([("_id", ASCENDING)])

    return db[MOVIE_COUNTS_COLL]


def filter_threshold_movies(db, threshold_ratings_count):
    def _agg():
        return db[MOVIE_COUNTS_COLL].aggregate(
            [
                {"$match": {"count": {"$gte": threshold_ratings_count}}},
                {"$project": {"_id": 0, "movie_id": "$_id"}},
                {"$out": THRESHOLD_MOVIES_COLL},
            ],
            allowDiskUse=True,
        )

    run_with_transient_retry(
        _agg, label="filter_threshold_movies", max_attempts=6, base_sleep_s=20.0
    )

    db[THRESHOLD_MOVIES_COLL].create_index([("movie_id", 1)], unique=True)

    return db[THRESHOLD_MOVIES_COLL]


def get_sampled_users(db, active_users_coll, target_sample_size):
    target_user_count = int((target_sample_size / (PER_USER_CAP / 1.25)) * 1.05)

    def _agg():
        return active_users_coll.aggregate(
            [{"$sample": {"size": target_user_count}}, {"$out": SAMPLED_USERS_COLL}]
        )

    run_with_transient_retry(
        _agg, label="get_sampled_users", max_attempts=6, base_sleep_s=20.0
    )

    db[SAMPLED_USERS_COLL].create_index([("user_id", 1)], unique=True)

    return db[SAMPLED_USERS_COLL]


def get_raw_final_sample(
    db,
    collection_name,
    sampled_users,
    deterministic_user_cap=True,
    collection_suffix="",
    user_batch_size: int = 500,
):
    raw_final_sample = db[collection_name]
    raw_final_sample.create_index(
        [("user_id", ASCENDING), ("movie_id", ASCENDING)],
        unique=True,
        name="user_movie_ix",
    )

    deterministic_cap_pipeline = [
        {
            "$addFields": {
                "h": {
                    "$toHashedIndexKey": {
                        "$concat": [
                            {"$toString": "$user_id"},
                            ":",
                            {"$toString": "$movie_id"},
                        ]
                    }
                }
            }
        },
        {"$sort": {"h": 1}},
        {"$limit": PER_USER_CAP},
        {"$unset": "h"},
    ]
    non_deterministic_cap_pipeline = [{"$sample": {"size": PER_USER_CAP}}]

    start = time.time()

    # stream user_ids in bounded batches from sampled_users
    uid_cur = sampled_users.find({}, {"_id": 0, "user_id": 1})
    batch = []
    total_users = 0
    for doc in uid_cur:
        batch.append(doc["user_id"])
        if len(batch) >= user_batch_size:
            _run_one_batch(
                db,
                batch,
                collection_name,
                deterministic_user_cap,
                deterministic_cap_pipeline,
                non_deterministic_cap_pipeline,
            )
            total_users += len(batch)
            print(f" … merged ratings for {total_users} users so far")
            batch = []

    if batch:
        _run_one_batch(
            db,
            batch,
            collection_name,
            deterministic_user_cap,
            deterministic_cap_pipeline,
            non_deterministic_cap_pipeline,
        )
        total_users += len(batch)

    elapsed = time.time() - start
    written_docs = raw_final_sample.estimated_document_count()
    written_users = total_users
    print(
        f"Added {written_docs} ratings for {written_users} filtered users in {elapsed:.1f}s "
        f"({(elapsed / max(1, written_users)):.2f} seconds/user)"
    )

    raw_final_sample.create_index([("movie_id", 1)])

    return raw_final_sample


def _run_one_batch(
    db,
    user_ids_batch,
    collection_name,
    deterministic_user_cap,
    deterministic_cap_pipeline,
    non_deterministic_cap_pipeline,
):
    pipeline = [
        # NEW: bound to this user batch
        {"$match": {"user_id": {"$in": user_ids_batch}}},
        {
            "$lookup": {
                "from": "ratings",
                "let": {"uid": "$user_id"},
                "pipeline": [
                    {"$match": {"$expr": {"$eq": ["$user_id", "$$uid"]}}},
                    {
                        "$lookup": {
                            "from": THRESHOLD_MOVIES_COLL,
                            "localField": "movie_id",
                            "foreignField": "movie_id",
                            "as": "pm",
                        }
                    },
                    {"$match": {"pm.0": {"$exists": True}}},
                    {"$unset": "pm"},
                    *(
                        deterministic_cap_pipeline
                        if deterministic_user_cap
                        else non_deterministic_cap_pipeline
                    ),
                ],
                "as": "r",
            }
        },
        {"$unwind": "$r"},
        {"$replaceRoot": {"newRoot": "$r"}},
        {
            "$merge": {
                "into": collection_name,
                "on": ["user_id", "movie_id"],
                "whenMatched": "keepExisting",
                "whenNotMatched": "insert",
            }
        },
    ]

    # Bound each run; keep disk use on to spill if needed.

    def _agg():
        return db[SAMPLED_USERS_COLL].aggregate(
            pipeline,
            allowDiskUse=True,
            comment="training_data_sample_build_chunked",
            # maxTimeMS=0  # optionally leave unlimited; or set a ceiling if CI is touchy
        )

    cursor = run_with_transient_retry(
        _agg,
        label=f"_run_one_batch users({len(user_ids_batch)})",
        max_attempts=6,
        base_sleep_s=20.0,
    )

    # If cursor is consumed by iteration in calling code, this will behave the same as before.
    # We do not convert to list() here to avoid storing large results in memory.
    # The aggregate returns a cursor-like object; PyMongo will lazily fetch as needed.
    # But returning cursor isn't necessary here since the aggregate writes via $merge.
    try:
        # ensure the aggregation completes by exhausting the cursor if the driver requires it
        # (some server-side aggregations will complete immediately but iteration is safe)
        for _ in cursor:
            # we don't need the returned docs; the loop forces completion where necessary
            pass
    except TypeError:
        # Some PyMongo aggregate variants may return None for server-side $out/$merge operations.
        # In that case, we simply ignore iteration.
        pass


def prune_orphan_entries(
    db,
    src,
    dst,
    movie_threshold,
    collection_suffix="",
    user_chunk_size=125,
):
    """
    Prune a ratings sample by removing entries whose movie_id does not meet
    a minimum frequency threshold in the *full* source sample.

    This version chunks the prune/merge step by user_id (not _id), which is
    robust even when _id values are non-ObjectId or mixed types.

    Args:
        db: pymongo database handle
        src: source collection name (raw sample)
        dst: destination collection base name
        movie_threshold: original threshold; code uses movie_threshold // 2
        collection_suffix: optional suffix appended to dst
        user_chunk_size: number of user_ids per prune chunk

    Returns:
        Destination collection handle.
    """
    adjusted_threshold = movie_threshold // 2
    dst_name = f"{dst}{collection_suffix}"

    # --- 1) Build qualifying movies temp collection from FULL src ---
    db[QUALIFYING_MOVIES_COLL].drop()

    def _agg_qm():
        return db[src].aggregate(
            [
                {"$group": {"_id": "$movie_id", "n": {"$sum": 1}}},
                {"$match": {"n": {"$gte": adjusted_threshold}}},
                {"$project": {"_id": 0, "movie_id": "$_id"}},
                {"$out": QUALIFYING_MOVIES_COLL},
            ],
            allowDiskUse=True,
        )

    run_with_transient_retry(
        _agg_qm,
        label="prune:build_qualifying_movies",
        max_attempts=6,
        base_sleep_s=20.0,
    )

    db[QUALIFYING_MOVIES_COLL].create_index([("movie_id", 1)])

    # Start fresh so reruns don’t accumulate results
    db[dst_name].drop()

    # --- 2) Prune + merge pipeline tail (applied per user chunk) ---
    base_pipeline_tail = [
        {
            "$lookup": {
                "from": QUALIFYING_MOVIES_COLL,
                "localField": "movie_id",
                "foreignField": "movie_id",
                "as": "qm",
            }
        },
        {"$match": {"qm.0": {"$exists": True}}},
        {"$unset": "qm"},
        {
            "$merge": {
                "into": dst_name,
                "whenMatched": "replace",
                "whenNotMatched": "insert",
            }
        },
    ]

    # Fetch all user_ids present in src. For your datasets (~4–5k users),
    # this is usually fine; it returns just distinct values, not full docs.
    t0 = time.time()
    user_ids = db[src].distinct("user_id")

    # Make progress stable/repeatable. Mixed types can exist; sort defensively.
    # If user_id is always numeric or always string, this will be stable.
    try:
        user_ids.sort()
    except TypeError:
        # Mixed types: sort by (type-name, stringified value) to avoid crashes.
        user_ids.sort(key=lambda x: (type(x).__name__, str(x)))

    print(
        f"Pruning {src} -> {dst_name} using user_id chunks: "
        f"{len(user_ids):,} users, user_chunk_size={user_chunk_size}, "
        f"movie_threshold={movie_threshold} (adjusted={adjusted_threshold})"
    )

    processed_users = 0

    for i in range(0, len(user_ids), user_chunk_size):
        u_chunk = user_ids[i : i + user_chunk_size]
        processed_users += len(u_chunk)

        chunk_match = {"user_id": {"$in": u_chunk}}

        # Optional: add a stable sort inside each chunk (not required for correctness)
        # but can help with predictable resource usage.
        pipeline = [{"$match": chunk_match}] + base_pipeline_tail

        def _agg_chunk():
            return db[src].aggregate(pipeline, allowDiskUse=True)

        # Retry each chunk if it hits a transient error (auto-retries for stepdowns).
        run_with_transient_retry(
            _agg_chunk,
            label=f"prune_chunk users {processed_users}/{len(user_ids)}",
            max_attempts=6,
            base_sleep_s=20.0,
        )

        elapsed = time.time() - t0
        print(
            f"Prune/merge progress: users processed {processed_users:,}/{len(user_ids):,} "
            f"(chunk={len(u_chunk):,}) in {elapsed:.1f}s"
        )

    original_sample_count = db[src].estimated_document_count()
    pruned_sample_count = db[dst_name].estimated_document_count()
    print(
        f"Original sample count: {original_sample_count:,}, "
        f"Pruned count: {pruned_sample_count:,}"
    )

    return db[dst_name]


def create_training_set(
    db, ratings, sample_size, active_users, use_cached_aggregations=False
):
    sampled_users = get_or_build_collection(
        db,
        SAMPLED_USERS_COLL,
        build_fn=lambda: get_sampled_users(db, active_users, sample_size),
        use_cache=False,
    )

    # We need to adjust the initial threshold on the number of reviews to account for the fact...
    # ..that we'll be taking a subsample of the full ratings collection. so if we want our...
    # ...subsample to end up with movies with a minimum of 20 ratings, but we're only grabbing 10% of users...
    # ...then we want to set an initial threshold of 20 * 10 = 200 ratings/movie to end up approximately in...
    # ...the right place. I'll do a little pruning at the end too, but if it's slightly off, it's not a big issue
    active_users_count = active_users.estimated_document_count()
    sampled_users_count = sampled_users.estimated_document_count()
    adjusted_movie_threshold = (
        MOVIE_MIN * (active_users_count / sampled_users_count) * 1.05
    )

    get_or_build_collection(
        db,
        THRESHOLD_MOVIES_COLL,
        build_fn=lambda: filter_threshold_movies(db, adjusted_movie_threshold),
        use_cache=False,
    )

    raw_sample_coll_name = f"{RAW_TRAINING_DATA_SAMPLE_COLL}_{sample_size}"
    final_sample_coll_name = f"{TRAINING_DATA_SAMPLE_COLL}_{sample_size}"

    get_or_build_collection(
        db,
        raw_sample_coll_name,
        build_fn=lambda: get_raw_final_sample(
            db, raw_sample_coll_name, sampled_users, deterministic_user_cap=False
        ),
        use_cache=False,
    )

    final_training_data_sample = get_or_build_collection(
        db,
        final_sample_coll_name,
        build_fn=lambda: prune_orphan_entries(
            db, raw_sample_coll_name, final_sample_coll_name, MOVIE_MIN
        ),
        use_cache=False,
    )

    return final_training_data_sample, final_sample_coll_name


def create_movie_data_sample(db, threshold=MOVIE_MIN):
    if MOVIE_COUNTS_COLL not in db.list_collection_names():
        create_movie_counts(db)

    pipeline = [
        {
            "$lookup": {
                "from": MOVIE_COUNTS_COLL,
                "localField": "movie_id",
                "foreignField": "_id",
                "as": "mc",
            }
        },
        {"$unwind": "$mc"},
        {"$match": {"mc.count": {"$gte": threshold}}},
        {"$addFields": {"ratings_count": "$mc.count"}},
    ]

    cursor = db["movies"].aggregate(pipeline, allowDiskUse=True)
    movie_df = pd.DataFrame(list(cursor))

    movie_df = movie_df[
        [
            "movie_id",
            "image_url",
            "movie_title",
            "year_released",
            "ratings_count",
            "content_type",
        ]
    ]
    movie_df["image_url"] = (
        movie_df["image_url"]
        .fillna("")
        .str.replace("https://a.ltrbxd.com/resized/", "", regex=False)
    )

    movie_df["image_url"] = (
        movie_df["image_url"]
        .fillna("")
        .str.replace(
            "https://s.ltrbxd.com/static/img/empty-poster-230.c6baa486.png",
            "",
            regex=False,
        )
    )

    return movie_df


def create_review_counts_df(db, threshold=MOVIE_MIN):
    # Create review counts dataframe
    review_count = db.ratings.aggregate(
        [
            {"$group": {"_id": "$movie_id", "review_count": {"$sum": 1}}},
            {"$match": {"review_count": {"$gte": threshold}}},
        ]
    )
    review_counts_df = pd.DataFrame(list(review_count))
    review_counts_df.rename(
        columns={"_id": "movie_id", "review_count": "count"}, inplace=True
    )

    return review_counts_df


def clean_up_temp_collections(db):
    collections_for_removal = [
        ACTIVE_USERS_COLL,
        THRESHOLD_MOVIES_COLL,
        SAMPLED_USERS_COLL,
        QUALIFYING_MOVIES_COLL,
    ]

    for sample_size in sample_sizes:
        raw_sample_coll = f"{RAW_TRAINING_DATA_SAMPLE_COLL}_{sample_size}"
        collections_for_removal.append(raw_sample_coll)

    for temp_collection in collections_for_removal:
        db.drop_collection(temp_collection)


def store_sample_movie_list(db, output_collection_name, sample_size):
    # index on movie id to make next step faster
    db[output_collection_name].create_index("movie_id")
    sample_movie_list = set(db[output_collection_name].distinct("movie_id"))
    with open(f"data/movie_lists/sample_movie_list_{sample_size}.txt", "wb") as fp:
        pickle.dump(sample_movie_list, fp)

    return sample_movie_list


def store_sample_ratings(db, output_collection_name, sample_size):
    # get all files in output collection and load into pandas dataframe, without _id
    proj = {"_id": 0}
    cursor = db[output_collection_name].find({}, proj)
    df = pd.DataFrame(cursor)

    # Export to CSV/Parquet files
    df.to_parquet(
        f"./data/training_data_samples/training_data_{sample_size}.parquet", index=False
    )


def create_and_store_sample(
    db, sample_size, ratings, active_users, use_cached_aggregations
):
    print(f"Starting build for sample size: {sample_size}")
    output_collection_name = f"{TRAINING_DATA_SAMPLE_COLL}_{sample_size}"

    create_training_set(db, ratings, sample_size, active_users, use_cached_aggregations)

    # index on movie id to make next steps faster
    db[output_collection_name].create_index("movie_id")
    sample_movie_list = store_sample_movie_list(db, output_collection_name, sample_size)

    get_rich_movie_data(
        movie_ids=sample_movie_list,
        output_path=f"./data/rich_movie_data/sample_movie_data_{sample_size}.parquet",
    )

    store_sample_ratings(db, output_collection_name, sample_size)


def main(use_cached_aggregations=False, remove_temp_collections=True):
    db_name, client = connect_to_db()
    db = client[db_name]

    ratings = db["ratings"]

    active_users = get_or_build_collection(
        db,
        ACTIVE_USERS_COLL,
        build_fn=lambda: filter_active_users(db),
        use_cache=use_cached_aggregations,
    )

    get_or_build_collection(
        db,
        MOVIE_COUNTS_COLL,
        build_fn=lambda: create_movie_counts(db),
        use_cache=use_cached_aggregations,
    )

    for sample_size in sample_sizes:
        create_and_store_sample(
            db, sample_size, ratings, active_users, use_cached_aggregations
        )

    movie_df = create_movie_data_sample(db, threshold=MOVIE_MIN)
    movie_df.to_csv("../static/data/movie_data.csv", index=False)

    review_counts_df = create_review_counts_df(db, threshold=MOVIE_MIN)
    review_counts_df.to_csv("data/review_counts.csv", index=False)

    if remove_temp_collections:
        clean_up_temp_collections(db)


if __name__ == "__main__":
    main(use_cached_aggregations=False, remove_temp_collections=True)
