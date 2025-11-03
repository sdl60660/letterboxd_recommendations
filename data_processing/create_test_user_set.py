import os
from pprint import pprint

from pymongo import ReplaceOne
from pymongo.errors import BulkWriteError

if os.getcwd().endswith("data_processing"):
    from db_connect import connect_to_db
    from get_user_ratings import get_user_data

else:
    from data_processing.db_connect import connect_to_db
    from data_processing.get_user_ratings import get_user_data


def main(sample_size=2000):
    # Connect to MongoDB client
    db_name, client = connect_to_db()

    # Find letterboxd database and user collection
    db = client[db_name]
    users = db.users
    test_sample_ratings = db.test_sample_ratings

    user_sample = list(
        users.aggregate(
            [
                {"$match": {"num_reviews": {"$gte": 300}, "scrape_status": "ok"}},
                {"$sample": {"size": sample_size}},
            ]
        )
    )

    all_test_sample_ratings = []
    for user in user_sample:
        user_ratings, _ = get_user_data(user["username"])
        all_test_sample_ratings += user_ratings

    upsert_ratings_operations = []
    for rating in all_test_sample_ratings:
        upsert_ratings_operations.append(
            ReplaceOne(
                {"user_id": rating["user_id"], "movie_id": rating["movie_id"]},
                rating,
                upsert=True,
            )
        )

    try:
        if len(upsert_ratings_operations) > 0:
            test_sample_ratings.bulk_write(upsert_ratings_operations, ordered=False)
    except BulkWriteError as bwe:
        pprint(bwe.details)


if __name__ == "__main__":
    main()
