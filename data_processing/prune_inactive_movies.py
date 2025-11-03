#!/usr/local/bin/python3.12
"""prune_inactive_movies.py: Find inactive/dead movie links in database and remove entries/corresponding ratings entries (migrate them to retired db)"""

import datetime

from utils.db_connect import connect_to_db


def get_filter_exp(inactive_fail_count=3):
    return {"fail_count": {"$gte": inactive_fail_count}, "scrape_status": "failed"}


def get_inactive_movies(movies_collection, filter_exp):
    return list(movies_collection.find(filter_exp))


def migrate_inactive_movies(movies_collection, retired_movies_collection, filter_exp):
    filter_stage = {"$match": filter_exp}
    now = datetime.datetime.now(datetime.timezone.utc)

    pipeline = [
        filter_stage,
        {"$set": {"migrated_at": now}},
        {
            "$merge": {
                "into": {
                    "db": retired_movies_collection.database.name,
                    "coll": retired_movies_collection.name,
                },
                "whenMatched": "keepExisting",
                "whenNotMatched": "insert",
            }
        },
    ]

    movies_collection.aggregate(pipeline, allowDiskUse=True)


def migrate_inactive_ratings(
    movies_coll, ratings_coll, retired_ratings_coll, filter_exp
):
    now = datetime.datetime.now(datetime.timezone.utc)

    pipeline = [
        {"$match": filter_exp},
        {"$project": {"_id": 0, "movie_id": 1}},
        {
            "$lookup": {
                "from": ratings_coll.name,
                "localField": "movie_id",
                "foreignField": "movie_id",
                "as": "r",
            }
        },
        {"$unwind": "$r"},
        {"$replaceRoot": {"newRoot": "$r"}},
        {"$set": {"migrated_at": now}},
        {
            "$merge": {
                "into": {
                    "db": retired_ratings_coll.database.name,
                    "coll": retired_ratings_coll.name,
                },
                "whenMatched": "keepExisting",
                "whenNotMatched": "insert",
            }
        },
    ]

    # Run on the DB where `movies_coll` lives
    movies_coll.aggregate(
        pipeline, allowDiskUse=True, comment="retire_ratings_by_inactive_movies"
    )


def delete_inactive_movies(movies_coll, filter_exp):
    return movies_coll.delete_many(filter_exp).deleted_count


def delete_inactive_ratings(ratings_coll, movies_coll, filter_exp):
    movie_ids = movies_coll.distinct("movie_id", filter=filter_exp)

    if not movie_ids:
        return 0

    res = ratings_coll.delete_many({"movie_id": {"$in": movie_ids}})
    return res.deleted_count


def main():
    # Connect to MongoDB client
    db_name, client = connect_to_db()
    db = client[db_name]
    movies = db.movies
    ratings = db.ratings

    retired_db = client["retired_entries"]
    retired_movies = retired_db.movies
    retired_ratings = retired_db.ratings

    filter_exp = get_filter_exp(inactive_fail_count=5)

    # 1) Copy movies
    migrate_inactive_movies(movies, retired_movies, filter_exp)

    # 2) Copy ratings linked to those movies
    migrate_inactive_ratings(movies, ratings, retired_ratings, filter_exp)

    # 3) Delete from active DBs
    deleted_ratings = delete_inactive_ratings(ratings, movies, filter_exp)
    deleted_movies = delete_inactive_movies(movies, filter_exp)

    print(
        f"Deleted from live DB/migrated to retired DB - ratings: {deleted_ratings:,}, movies: {deleted_movies:,}"
    )


if __name__ == "__main__":
    main()
