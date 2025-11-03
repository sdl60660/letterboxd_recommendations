import pickle
import time

import pandas as pd
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import CollectionInvalid
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

    ratings.aggregate(
        [
            {"$group": {"_id": "$user_id", "n": {"$sum": 1}}},
            {"$match": {"n": {"$gte": USER_MIN}}},
            {"$project": {"_id": 0, "user_id": "$_id"}},  # one field: user_id
            {"$out": ACTIVE_USERS_COLL},
        ],
        allowDiskUse=True,
    )

    db[ACTIVE_USERS_COLL].create_index([("user_id", 1)], unique=True)

    return db[ACTIVE_USERS_COLL]


def create_movie_counts(db):
    ratings = db.ratings

    # Build (or rebuild) movie_counts
    db[MOVIE_COUNTS_COLL].drop()
    ratings.aggregate(
        [
            {"$group": {"_id": "$movie_id", "count": {"$sum": 1}}},
            {"$out": MOVIE_COUNTS_COLL},
        ],
        allowDiskUse=True,
    )

    # Indexes
    db[MOVIE_COUNTS_COLL].create_index([("count", DESCENDING)])
    db[MOVIE_COUNTS_COLL].create_index([("_id", ASCENDING)])

    return db[MOVIE_COUNTS_COLL]


def filter_threshold_movies(db, threshold_ratings_count):
    db[MOVIE_COUNTS_COLL].aggregate(
        [
            {"$match": {"count": {"$gte": threshold_ratings_count}}},
            {"$project": {"_id": 0, "movie_id": "$_id"}},
            {"$out": THRESHOLD_MOVIES_COLL},
        ],
        allowDiskUse=True,
    )

    db[THRESHOLD_MOVIES_COLL].create_index([("movie_id", 1)], unique=True)

    return db[THRESHOLD_MOVIES_COLL]


def get_sampled_users(db, active_users_coll, target_sample_size):
    target_user_count = int((target_sample_size / (PER_USER_CAP / 1.25)) * 1.05)
    active_users_coll.aggregate(
        [{"$sample": {"size": target_user_count}}, {"$out": SAMPLED_USERS_COLL}]
    )

    db[SAMPLED_USERS_COLL].create_index([("user_id", 1)], unique=True)

    return db[SAMPLED_USERS_COLL]


def get_raw_final_sample(
    db,
    collection_name,
    sampled_users,
    deterministic_user_cap=True,
    collection_suffix="",
):
    raw_final_sample = db[collection_name]
    raw_final_sample.create_index(
        [("user_id", ASCENDING), ("movie_id", ASCENDING)],
        unique=True,
        name="user_movie_unique",
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

    pipeline = [
        # {"$match": {"user_id": {"$in": uids}}},
        {
            "$lookup": {
                "from": "ratings",
                "let": {"uid": "$user_id"},
                "pipeline": [
                    # ratings for this user (hits ratings{user_id:1} index)
                    {"$match": {"$expr": {"$eq": ["$user_id", "$$uid"]}}},
                    # keep only movies over threshold
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
                        if deterministic_user_cap == True
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

    sampled_users.aggregate(
        pipeline, allowDiskUse=True, comment="training_data_sample_build"
    )

    elapsed = time.time() - start
    written_docs = raw_final_sample.estimated_document_count()
    written_users = sampled_users.estimated_document_count()
    print(
        f"Added {written_docs} ratings for {written_users} filtered users in {elapsed:.1f}s ({(elapsed / written_users):.2f} seconds/user)"
    )

    raw_final_sample.create_index([("movie_id", 1)])

    return raw_final_sample


def prune_orphan_entries(db, src, dst, movie_threshold, collection_suffix=""):
    #  this is routh, but it seems to be a decent enough ratio to make the hard-cutoff threshold about half of the original filter pass threshold
    adjusted_threshold = movie_threshold // 2

    # create temp movie group collection
    db[QUALIFYING_MOVIES_COLL].drop()
    db[src].aggregate(
        [
            {"$group": {"_id": "$movie_id", "n": {"$sum": 1}}},
            {"$match": {"n": {"$gte": adjusted_threshold}}},
            {"$project": {"_id": 0, "movie_id": "$_id"}},
            {"$out": QUALIFYING_MOVIES_COLL},
        ],
        allowDiskUse=True,
    )
    db[QUALIFYING_MOVIES_COLL].create_index([("movie_id", 1)])

    pipeline = pipeline = [
        # Start from the big collection and just *check existence* in the small set
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
        # Write as we go (so you can poll progress); $merge streams inserts
        {
            "$merge": {
                "into": f"{dst}{collection_suffix}",
                "whenMatched": "replace",
                "whenNotMatched": "insert",
            }
        },
    ]

    db[src].aggregate(pipeline, allowDiskUse=True)

    original_sample_count = db[src].estimated_document_count()
    pruned_sample_count = db[dst].estimated_document_count()
    print(
        f"Original sample count: {original_sample_count}, Pruned count: {pruned_sample_count}"
    )

    return db[dst]


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
