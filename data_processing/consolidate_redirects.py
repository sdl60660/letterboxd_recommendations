#!/usr/local/bin/python3.12

import argparse
import datetime as datetime

from pymongo import ASCENDING
from pymongo.errors import PyMongoError
from tqdm import tqdm

# Use your existing DB helper
from utils.db_connect import connect_to_db


def archive_old_ratings(db, retired_db, old_id: str, new_id: str) -> None:
    """Copy old-id ratings into retired_entries.ratings with audit fields."""
    pipeline = [
        {"$match": {"movie_id": old_id}},
        {
            "$addFields": {
                "migrated_at": "$$NOW",
                "redirect_from": old_id,
                "redirect_to": new_id,
            }
        },
        {
            "$merge": {
                "into": {"db": retired_db.name, "coll": "ratings"},
                "whenMatched": "keepExisting",
                "whenNotMatched": "insert",
            }
        },
    ]
    db.ratings.aggregate(pipeline, allowDiskUse=True)


def archive_old_movie(db, retired_db, old_id: str, new_id: str) -> None:
    """Copy old movie doc into retired_entries.movies with redirect markers."""
    pipeline = [
        {"$match": {"movie_id": old_id}},
        {
            "$addFields": {
                "migrated_at": "$$NOW",
                "redirected_at": "$$NOW",
                "redirect_to": new_id,
                "scrape_status": "redirected",
            }
        },
        {
            "$merge": {
                "into": {"db": retired_db.name, "coll": "movies"},
                "whenMatched": "replace",
                "whenNotMatched": "insert",
            }
        },
    ]
    db.movies.aggregate(pipeline, allowDiskUse=True)


def iter_redirect_pairs(redirects_coll, only_status="pending"):
    q = {"new_id": {"$type": "string"}, "old_id": {"$type": "string"}}
    if only_status is not None:
        q["status"] = only_status

    for doc in redirects_coll.find(q, {"_id": 1, "old_id": 1, "new_id": 1}):
        old_id, new_id = doc.get("old_id"), doc.get("new_id")
        if old_id and new_id and old_id != new_id:
            yield (doc["_id"], old_id, new_id)  # keep ObjectId


def ensure_indexes(db):
    # movie_redirects typically small; index old_id helps a bit
    db.movie_redirects.create_index([("old_id", ASCENDING)])


def migrate_ratings_one_pair(db, retired_db, old_id: str, new_id: str):
    """
    Archive old-id ratings -> retired_entries.ratings, then remap in-place to new_id,
    then delete the old-id rows from main db.
    """
    # Archive the originals (to keep a record of what was removed)
    archive_old_ratings(db, retired_db, old_id, new_id)

    # Merge remapped rows back to main ratings under new_id (deduped by user_id+movie_id)
    pipeline = [
        {"$match": {"movie_id": old_id}},
        {"$addFields": {"movie_id": new_id}},
        {"$unset": "_id"},  # avoid duplicate _id on insert
        {
            "$merge": {
                "into": "ratings",
                "on": ["user_id", "movie_id"],
                "whenMatched": "keepExisting",  # don't overwrite if user already has a rating for new_id
                "whenNotMatched": "insert",
            }
        },
    ]
    db.ratings.aggregate(pipeline, allowDiskUse=True)

    # Delete old_id rows from main db
    deleted = db.ratings.delete_many({"movie_id": old_id}).deleted_count
    return (None, deleted)


def mark_movie_redirected(db, old_id: str, new_id: str) -> None:
    now = datetime.datetime.now(datetime.timezone.utc)
    db.movies.update_one(
        {"movie_id": old_id},
        {
            "$set": {
                "scrape_status": "redirected",
                "redirect_to": new_id,
                "redirected_at": now,
            }
        },
    )


def delete_old_movie(db, retired_db, old_id: str, new_id: str) -> int:
    """
    Archive old movie doc to retired_entries.movies, then delete from main DB.
    Returns delete count (0/1).
    """
    archive_old_movie(db, retired_db, old_id, new_id)
    return db.movies.delete_one({"movie_id": old_id}).deleted_count


def complete_redirect_row(redirects_coll, doc_id, stats: dict) -> None:
    now = datetime.datetime.now(datetime.timezone.utc)
    redirects_coll.update_one(
        {"_id": doc_id},  # <- ObjectId now
        {"$set": {"status": "merged", "merged_at": now, **stats}},
    )


def process_all_redirects(db, batch_limit: int) -> None:
    redirects = db.movie_redirects
    retired_db = db.client["retired_entries"]

    db.movie_redirects.create_index([("old_id", ASCENDING)])

    pairs = list(iter_redirect_pairs(redirects, only_status="pending"))
    if batch_limit and batch_limit > 0:
        pairs = pairs[:batch_limit]

    print(f"Found {len(pairs)} pending redirects.")
    for doc_id, old_id, new_id in tqdm(pairs, desc="Merging"):
        try:
            inserted_kept, deleted = migrate_ratings_one_pair(
                db, retired_db, old_id, new_id
            )
            mark_movie_redirected(db, old_id, new_id)
            deleted_movies = delete_old_movie(db, retired_db, old_id, new_id)

            stats = {
                "ratings_migrated": deleted,
                "movie_migrated": deleted_movies,
                "redirect_to": new_id,
            }
            complete_redirect_row(redirects, doc_id, stats)
        except PyMongoError as e:
            print(f"[WARN] Failed to merge {old_id} → {new_id}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate movie ID redirects and archive inactive records."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N redirects (for testing).",
    )
    args = parser.parse_args()

    db_name, client = connect_to_db()
    db = client[db_name]

    process_all_redirects(db, batch_limit=args.limit)


if __name__ == "__main__":
    main()
