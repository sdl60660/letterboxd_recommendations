from pymongo import MongoClient, ASCENDING, DESCENDING
from string import ascii_lowercase, digits
from time import time

from db_connect import connect_to_db

# Connect
db_name, client, tmdb_key = connect_to_db()
db = client[db_name]
src = db.ratings
tmp = db.ratings_normalized_tmp

# Clean slate & pre-create temp so counts work immediately
tmp.drop()
db.create_collection("ratings_normalized_tmp")

# ---- Bucketing ----
# Buckets: a..z, 0..9, and "other"
BUCKETS = list(ascii_lowercase) + list(digits) + ["other"]

def match_for_bucket(ch):
    # First codepoint of lowercased string
    first = {"$substrCP": [{"$toLower": {"$toString": "$user_id"}}, 0, 1]}
    if ch == "other":
        # Not in a..z or 0..9
        return {
            "$expr": {
                "$not": [{"$in": [first, BUCKETS[:-1]]}]
            }
        }
    else:
        return {"$expr": {"$eq": [first, ch]}}

def pipeline_for_bucket(ch):
    return [
        {"$match": match_for_bucket(ch)},

        {"$addFields": {
            "user_lower": {"$toLower": {"$toString": "$user_id"}}
        }},

        # Pick canonical doc per (lower(user_id), movie_id)
        {"$group": {
            "_id": {"ul": "$user_lower", "m": "$movie_id"},
            "doc": {
                "$top": {
                    "sortBy": {"updated_at": DESCENDING, "_id": DESCENDING},
                    "output": "$$ROOT"
                }
            }
        }},

        # Force user_id to lowercase on the kept doc
        {"$replaceRoot": {
            "newRoot": {"$mergeObjects": ["$doc", {"user_id": "$_id.ul"}]}
        }},

        {"$unset": ["user_lower"]},

        # Write/replace into the temp collection
        {"$merge": {
            "into": "ratings_normalized_tmp",
            "whenMatched": "replace",
            "whenNotMatched": "insert"
        }}
    ]

# ---- Run buckets sequentially (easy to read + steady progress) ----
overall_start = time()
total_written_prev = 0

for i, ch in enumerate(BUCKETS, 1):
    bucket_start = time()
    src.aggregate(
        pipeline_for_bucket(ch),
        allowDiskUse=True,
        comment=f"ratings_normalize_bucket_{ch}"
    )
    # Progress after each bucket finishes
    n = tmp.estimated_document_count()
    delta = n - total_written_prev
    total_written_prev = n
    dur = time() - bucket_start
    print(f"[{i}/{len(BUCKETS)}] bucket={ch!r} +{delta:,} docs (bucket {dur:0.1f}s)  total={n:,}")

print(f"All buckets done in {(time()-overall_start):0.1f}s.")

# Build unique index AFTER ingest (fastest)
print("Building unique index on temp...")
tmp.create_index([("user_id", ASCENDING), ("movie_id", ASCENDING)],
                 unique=True, name="compound_key")
print("Index build complete.")

# Swap when ready (uncomment when you plan the cutover window)
# db.command("renameCollection", f"{db.name}.ratings", to=f"{db.name}.ratings_backup")
# db.command("renameCollection", f"{db.name}.ratings_normalized_tmp", to=f"{db.name}.ratings")
# print("Swap complete.")
