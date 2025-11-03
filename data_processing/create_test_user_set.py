import os
from pprint import pprint

from pymongo import ReplaceOne
from pymongo.errors import BulkWriteError
from tqdm.auto import tqdm

if os.getcwd().endswith("data_processing"):
    from db_connect import connect_to_db
    from get_user_ratings import get_user_data

else:
    from data_processing.db_connect import connect_to_db
    from data_processing.get_user_ratings import get_user_data


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


def main(sample_size=2000):
    # Connect to MongoDB client
    db_name, client = connect_to_db()

    # Find letterboxd database and user collection
    db = client[db_name]
    users = db.users
    test_sample_ratings = prepare_test_sample_ratings(db)

    user_sample = list(
        users.aggregate(
            [
                {"$match": {"num_reviews": {"$gte": 300}, "scrape_status": "ok"}},
                {"$sample": {"size": sample_size}},
                {"$project": {"_id": 0, "username": 1}},
            ]
        )
    )

    all_test_sample_ratings = []
    for user in tqdm(user_sample, desc="Fetching user data"):
        user_ratings, _status = get_user_data(user["username"])
        all_test_sample_ratings += user_ratings

    # Upsert all sampled ratings into the fresh collection
    ops = [
        ReplaceOne(
            {"user_id": r["user_id"], "movie_id": r["movie_id"]},
            r,
            upsert=True,
        )
        for r in all_test_sample_ratings
    ]

    try:
        test_sample_ratings.bulk_write(ops, ordered=False)
    except BulkWriteError as bwe:
        pprint(bwe.details)


if __name__ == "__main__":
    main()
