import os
from pprint import pprint
from statistics import mean

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


def attach_synthetic_ratings(
    all_ratings: list, global_mean: float, global_weight: int = 5
) -> None:
    """
    Mutates `all_ratings` in place, adding `synthetic_rating_val`:
        - If rating_val >= 0: synthetic = rating_val
        - Else if liked == True: synthetic = weighted mean of (user liked+rated mean, global liked+rated mean)
        - Else: leave unset
    """

    # 1) Per-user mean over liked & rated
    liked_rated_by_user = {}
    for r in all_ratings:
        uid = r.get("user_id")
        if not uid:
            continue
        if (
            r.get("liked") is True
            and isinstance(r.get("rating_val"), (int, float))
            and r["rating_val"] >= 0
        ):
            liked_rated_by_user.setdefault(uid, []).append(r["rating_val"])

    user_mean_cache = {
        uid: (mean(vals), len(vals)) for uid, vals in liked_rated_by_user.items()
    }

    # 2) Attach synthetic per row
    for r in all_ratings:
        rv = r.get("rating_val")
        if isinstance(rv, (int, float)) and rv >= 0:
            # Explicit rating: prefer it
            r["synthetic_rating_val"] = rv
            continue

        # Only compute synthetic for unrated-but-liked
        if r.get("liked") is not True:
            continue

        uid = r.get("user_id")
        u_mean, u_n = user_mean_cache.get(uid, (None, 0))

        # Decide synthetic value
        synthetic = None
        if u_mean is not None and global_mean is not None:
            synthetic = (u_n * u_mean + global_weight * global_mean) / (
                u_n + global_weight
            )
        elif u_mean is not None:
            synthetic = u_mean
        elif global_mean is not None:
            synthetic = global_mean
        else:
            synthetic = None

        if synthetic is not None:
            # keep as float; if you prefer ints, you could round here
            r["synthetic_rating_val"] = float(synthetic)


def is_keepable(r: dict) -> bool:
    """Keep if explicit rating exists, or if liked."""
    rv = r.get("rating_val")
    return (isinstance(rv, (int, float)) and rv >= 0) or (r.get("liked") is True)


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
    except BulkWriteError as bwe:
        pprint(bwe.details)


if __name__ == "__main__":
    main()
