import os
import random
import warnings
from statistics import mean

import pandas as pd
from pymongo import ReplaceOne
from tqdm.auto import tqdm

if os.getcwd().endswith("data_processing"):
    from get_user_ratings import attach_synthetic_ratings, get_user_data
    from utils.db_connect import connect_to_db
    from utils.mongo_utils import safe_commit_ops, safe_commit_ops_chunked

else:
    from data_processing.get_user_ratings import attach_synthetic_ratings, get_user_data
    from data_processing.utils.db_connect import connect_to_db
    from data_processing.utils.mongo_utils import (
        safe_commit_ops,
        safe_commit_ops_chunked,
    )


TEST_USER_COLLECTION_NAME = "test_sample_ratings"
STATIC_FILE_PATH = "./testing/test_user_data.parquet"


def prepare_test_sample_ratings(db, name="test_sample_ratings"):
    """
    Drops and recreates the test_sample_ratings collection,
    then adds indexes:
        - movie_id (asc)
        - user_id (asc)
        - compound_unique_key on (movie_id, user_id), unique
    """
    # Drop if it exists (removes data + indexes)
    if name in db.list_collection_names():
        db[name].drop()

    # Recreate and add indexes
    coll = db.create_collection(name)
    coll.create_index([("movie_id", 1)])
    coll.create_index([("user_id", 1)])
    coll.create_index(
        [("movie_id", 1), ("user_id", 1)],
        unique=True,
        name="compound_unique_key",
    )
    return coll


def get_global_mean_liked_rated(all_ratings):
    global_liked_rated = [
        r["rating_val"]
        for r in all_ratings
        if r.get("liked") is True
        and isinstance(r.get("rating_val"), (int, float))
        and r["rating_val"] >= 0
    ]
    global_mean = mean(global_liked_rated) if global_liked_rated else None
    return global_mean


def is_keepable(r: dict) -> bool:
    """Keep if explicit rating exists, or if liked."""
    rv = r.get("rating_val")
    return (isinstance(rv, (int, float)) and rv >= 0) or (r.get("liked") is True)


def store_test_ratings(collection, outfile_path=STATIC_FILE_PATH):
    collection_doc_count = collection.count_documents({})

    existing_data = pd.read_parquet(outfile_path)
    existing_row_count = existing_data.shape[0]

    # Just adding a safety check in case something goes wrong, so that...
    # we don't overwrite the working test user file if the collection ends up empty/near-empty
    if collection_doc_count / existing_row_count < 0.5:
        warnings.warn(
            f"WARNING: not overwriting existing user test set because new data count ({collection_doc_count:,}) is too small relative to existing data count ({existing_row_count:,}). Must be at least half as large."
        )
        return

    # get all files in output collection and load into pandas dataframe, without _id
    proj = {"_id": 0}
    cursor = collection.find({}, proj)
    df = pd.DataFrame(cursor)

    collection_doc_count = collection.count_documents({})

    # Export to CSV/Parquet files
    df.to_parquet(outfile_path, index=False)


def get_user_sample(users, num_users, review_range=[150, 1200]):
    user_sample = list(
        users.aggregate(
            [
                {
                    "$match": {
                        "num_reviews": {
                            "$gte": review_range[0],
                            "$lte": review_range[1],
                        },
                        "scrape_status": "ok",
                    }
                },
                {"$sample": {"size": num_users}},
                {"$project": {"_id": 0, "username": 1}},
            ]
        )
    )
    return user_sample


def cycle_out_users(test_sample_ratings, num_to_drop, current_users):
    users_to_drop = random.sample(current_users, num_to_drop)
    delete_result = test_sample_ratings.delete_many({"user_id": {"$in": users_to_drop}})
    print(
        f"Dropped ratings for {num_to_drop} users ({delete_result.deleted_count} docs)."
    )


def get_new_sample_ratings(user_sample):
    all_test_sample_ratings = []
    for user in tqdm(user_sample, desc="Fetching user data"):
        user_ratings, _status = get_user_data(user["username"])
        all_test_sample_ratings += user_ratings

    global_mean_liked_rated = get_global_mean_liked_rated(all_test_sample_ratings)
    attach_synthetic_ratings(
        all_test_sample_ratings, global_mean_liked_rated, global_weight=5
    )

    keepable = [r for r in all_test_sample_ratings if is_keepable(r)]

    # Create ops to upsert all sampled ratings into the fresh collection
    ops = [
        ReplaceOne(
            {"user_id": r["user_id"], "movie_id": r["movie_id"]},
            r,
            upsert=True,
        )
        for r in keepable
    ]

    return ops


def main(total_sample_size=1500, cycled_users=150):
    # Connect to MongoDB client
    db_name, client = connect_to_db()

    # Find letterboxd database and user collection
    db = client[db_name]
    users = db.users

    # If collection already exists, find the total number of new user ratings we need
    # and cycle out some set of stale ratings
    if TEST_USER_COLLECTION_NAME in db.list_collection_names():
        test_sample_ratings = db[TEST_USER_COLLECTION_NAME]
        total_current_users = list(test_sample_ratings.distinct("user_id"))
        sample_size = cycled_users + (total_sample_size - len(total_current_users))

        cycle_out_users(test_sample_ratings, sample_size, total_current_users)
    # If it doesn't exist, create collection + indices
    else:
        test_sample_ratings = prepare_test_sample_ratings(
            db, name=TEST_USER_COLLECTION_NAME
        )
        sample_size = total_sample_size

    # Get new set of users
    user_sample = get_user_sample(users, sample_size)
    # Get ratings upsert ops from those users
    ops = get_new_sample_ratings(user_sample)

    # Send all update ops to collection
    safe_commit_ops_chunked(
        test_sample_ratings,
        ops,
        batch_size=5000,
        desc="Adding ratings (wtih likes) to test user collection",
    )

    store_test_ratings(test_sample_ratings)


if __name__ == "__main__":
    main()
