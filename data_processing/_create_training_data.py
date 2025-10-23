import pickle
import time
from math import ceil

from pymongo import ASCENDING
from pymongo.errors import CollectionInvalid

from db_connect import connect_to_db


# --- params (tune these) ---
USER_MIN = 50               # users must have ≥ this many ratings to end up in sampling pool
MOVIE_MIN = 25              # movies must have ≥ this many ratings (this gets adjusted to account for subsampling)

TARGET_EDGES = 1_500_000    # aim ~this many ratings in final sample (±5–10%)
PER_USER_CAP = 300          # at most this many ratings per sampled user

# temp/final collection names
ACTIVE_USERS_COLL = "active_users_tmp"
POPULAR_MOVIES_COLL = "popular_movies_tmp"
DENSE_POOL_COLL = "ratings_dense_pool"
SAMPLED_USERS_COLL = "sampled_users_tmp"
FINAL_SAMPLE_COLL = "training_data_sample"


def ensure_empty_collection(db, name, wait_secs=2.0):
    # Drop if present
    if name in db.list_collection_names():
        db[name].drop()
        # wait briefly for namespace to disappear
        deadline = time.time() + wait_secs
        while name in db.list_collection_names() and time.time() < deadline:
            time.sleep(0.05)
    # Create if absent (ignore if someone else just created it)
    try:
        if name not in db.list_collection_names():
            db.create_collection(name)
    except CollectionInvalid:
        pass
    return db[name]

def get_or_build_collection(db, name, build_fn, use_cache=False, drop_existing=True):
  """
  Return db[name] if it exists and use_cache=True; otherwise run build_fn()
  to (re)build the collection and return it.
  """
  names = set(db.list_collection_names())
  if not use_cache or name not in names:
      if drop_existing and name in names:
          db[name].drop()
      coll = build_fn()
      return coll if coll is not None else db[name]
  return db[name]

def filter_active_users(db, ratings):
  ratings.aggregate([
      {"$group": {"_id": "$user_id", "n": {"$sum": 1}}},
      {"$match": {"n": {"$gte": USER_MIN}}},
      {"$project": {"_id": 0, "user_id": "$_id"}},   # one field: user_id
      {"$out": ACTIVE_USERS_COLL}
  ], allowDiskUse=True)

  db[ACTIVE_USERS_COLL].create_index([("user_id", 1)], unique=True)

  return db[ACTIVE_USERS_COLL]

def filter_popular_movies(db, ratings, threshold_ratings_count):
  ratings.aggregate([
      {"$group": {"_id": "$movie_id", "n": {"$sum": 1}}},
      {"$match": {"n": {"$gte": threshold_ratings_count}}},
      {"$project": {"_id": 0, "movie_id": "$_id"}},  # one field: movie_id
      {"$out": POPULAR_MOVIES_COLL}
  ], allowDiskUse=True)

  db[POPULAR_MOVIES_COLL].create_index([("movie_id", 1)], unique=True)

  return db[POPULAR_MOVIES_COLL]


def get_sampled_users(db, active_users_coll):
  target_user_count = int((TARGET_EDGES / (PER_USER_CAP / 1.25)) * 1.1)
  active_users_coll.aggregate([
      {"$sample": {"size": target_user_count}},
      {"$out": SAMPLED_USERS_COLL}
  ])

  db[SAMPLED_USERS_COLL].create_index([("user_id", 1)], unique=True)

  return db[SAMPLED_USERS_COLL]


def get_final_sample(db, sampled_users, deterministic_user_cap = True):
  final_sample = ensure_empty_collection(db, FINAL_SAMPLE_COLL)
  final_sample.create_index(
      [("user_id", ASCENDING), ("movie_id", ASCENDING)],
      unique=True,
      name="user_movie_unique"
  )

  deterministic_cap_pipeline = [
    { "$addFields": {
      "h": { "$toHashedIndexKey": {
        "$concat": [ { "$toString": "$user_id" }, ":", { "$toString": "$movie_id" } ]
      }}
    }},
    { "$sort": { "h": 1 } },
    { "$limit": PER_USER_CAP },
    { "$unset": "h" }
  ]
  non_deterministic_cap_pipeline = [{ "$sample": { "size": PER_USER_CAP } }]

  start = time.time()

  pipeline = [
    # {"$match": {"user_id": {"$in": uids}}},
    {
      "$lookup": {
        "from": "ratings",
        "let": { "uid": "$user_id" },
        "pipeline": [
          # ratings for this user (hits ratings{user_id:1} index)
          { "$match": { "$expr": { "$eq": ["$user_id", "$$uid"] } } },

          # keep only movies over threshold
          {
            "$lookup": {
              "from": POPULAR_MOVIES_COLL,
              "localField": "movie_id",
              "foreignField": "movie_id",
              "as": "pm"
            }
          },
          { "$match": { "pm.0": { "$exists": True } } },
          { "$unset": "pm" },
          *(deterministic_cap_pipeline if deterministic_user_cap == True else non_deterministic_cap_pipeline)
        ],
        "as": "r"
      }
    },
    { "$unwind": "$r" },
    { "$replaceRoot": { "newRoot": "$r" } },
    {
      "$merge": {
        "into": FINAL_SAMPLE_COLL,
        "on": ["user_id", "movie_id"],
        "whenMatched": "keepExisting",
        "whenNotMatched": "insert"
      }
    }
  ]

  sampled_users.aggregate(pipeline, allowDiskUse=True, comment="training_data_sample_build")

  elapsed = time.time() - start
  written_docs = final_sample.estimated_document_count()
  written_users = sampled_users.estimated_document_count()
  print(f"Added {written_docs} ratings for {written_users} filtered users in {elapsed:.1f}s ({(elapsed/written_users):.2f} seconds/user)")

  return final_sample


def main(use_cached_aggregations=False):
  db_name, client, tmdb_key = connect_to_db()
  db = client[db_name]
  ratings = db["ratings"]

  active_users = get_or_build_collection(
        db, ACTIVE_USERS_COLL,
        build_fn=lambda: filter_active_users(db, ratings),
        use_cache=use_cached_aggregations
    )
  
  sampled_users = get_or_build_collection(
    db, SAMPLED_USERS_COLL, build_fn=lambda: get_sampled_users(db, active_users),
      use_cache=use_cached_aggregations
  )

  # We need to adjust the initial threshold on the number of reviews to account for the fact...
  # ..that we'll be taking a subsample of the full ratings collection. so if we want our...
  # ...subsample to end up with movies with a minimum of 20 ratings, but we're only grabbing 10% of users...
  # ...then we want to set an initial threshold of 20 * 10 = 200 ratings/movie to end up approximately in...
  # ...the right place. I'll do a little pruning at the end too, but if it's slightly off, it's not a big issue
  active_users_count = active_users.estimated_document_count()
  sampled_users_count = sampled_users.estimated_document_count()
  adjusted_movie_threshold = MOVIE_MIN * (active_users_count / sampled_users_count) * 1.05

  popular_movies = get_or_build_collection(
      db, POPULAR_MOVIES_COLL,
      build_fn=lambda: filter_popular_movies(db, ratings, adjusted_movie_threshold),
      use_cache=use_cached_aggregations
  )

  print(f"Active users: {active_users.estimated_document_count():,}, popular movies: {popular_movies.estimated_document_count():,}, user sample: {sampled_users.estimated_document_count():}")
  final_sample = get_final_sample(db, sampled_users, deterministic_user_cap=False)
  print("Final training data sample size:", final_sample.estimated_document_count())


  # dense_ratings_pool = create_dense_ratings_pool(db, ratings)
  # dense_count = dense_ratings_pool.estimated_document_count()
  # print(f"dense pool ratings: {dense_count:,}")


if __name__ == "__main__":
  main(use_cached_aggregations=True)