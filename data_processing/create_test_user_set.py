import os
from pprint import pprint
from statistics import mean

import pandas as pd
from pymongo import ReplaceOne
from pymongo.errors import BulkWriteError
from tqdm.auto import tqdm

if os.getcwd().endswith("data_processing"):
    from get_user_ratings import attach_synthetic_ratings, get_user_data
    from utils.db_connect import connect_to_db

else:
    from data_processing.get_user_ratings import attach_synthetic_ratings, get_user_data
    from data_processing.utils.db_connect import connect_to_db


def prepare_test_sample_ratings(db):
    """
    Drops and recreates the test_sample_ratings collection,
    then adds indexes:
        - movie_id (asc)
        - user_id (asc)
        - compound_unique_key on (movie_id, user_id), unique
    """
    name = "test_sample_ratings"

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


def store_test_ratings(collection):
    # get all files in output collection and load into pandas dataframe, without _id
    proj = {"_id": 0}
    cursor = collection.find({}, proj)
    df = pd.DataFrame(cursor)

    # Export to CSV/Parquet files
    df.to_parquet("./testing/test_user_data.parquet", index=False)


def main(sample_size=1500):
    # Connect to MongoDB client
    db_name, client = connect_to_db()

    # Find letterboxd database and user collection
    db = client[db_name]
    users = db.users
    test_sample_ratings = prepare_test_sample_ratings(db)

    user_sample = list(
        users.aggregate(
            [
                {
                    "$match": {
                        "num_reviews": {"$gte": 150, "$lte": 2000},
                        "scrape_status": "ok",
                    }
                },
                {"$sample": {"size": sample_size}},
                {"$project": {"_id": 0, "username": 1}},
            ]
        )
    )

    all_test_sample_ratings = []
    for user in tqdm(user_sample, desc="Fetching user data"):
        user_ratings, _status = get_user_data(user["username"])
        all_test_sample_ratings += user_ratings

    global_mean_liked_rated = get_global_mean_liked_rated(all_test_sample_ratings)
    attach_synthetic_ratings(
        all_test_sample_ratings, global_mean_liked_rated, global_weight=5
    )

    keepable = [r for r in all_test_sample_ratings if is_keepable(r)]

    # Upsert all sampled ratings into the fresh collection
    ops = [
        ReplaceOne(
            {"user_id": r["user_id"], "movie_id": r["movie_id"]},
            r,
            upsert=True,
        )
        for r in keepable
    ]

    try:
        test_sample_ratings.bulk_write(ops, ordered=False)
        store_test_ratings(test_sample_ratings)
    except BulkWriteError as bwe:
        pprint(bwe.details)


if __name__ == "__main__":
    main()
