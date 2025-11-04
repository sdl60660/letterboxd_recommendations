#!/usr/local/bin/python3.12

import asyncio
import datetime
import os
from statistics import mean

from pymongo import ReplaceOne, UpdateOne

# from pymongo.operations import ReplaceOne


if os.getcwd().endswith("/data_processing"):
    from get_ratings import get_user_ratings
    from utils.db_connect import connect_to_db
    from utils.mongo_utils import safe_commit_ops
    from utils.utils import get_page_count

else:
    from data_processing.get_ratings import get_user_ratings
    from data_processing.utils.db_connect import connect_to_db
    from data_processing.utils.mongo_utils import safe_commit_ops
    from data_processing.utils.utils import get_page_count


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


def get_user_data(username, data_opt_in=False, include_liked_items=True):
    num_pages, display_name = get_page_count(
        username, url="https://letterboxd.com/{}/films/by/date"
    )
    if num_pages == -1:
        return [], "user_not_found"

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    future = asyncio.ensure_future(
        get_user_ratings(
            username,
            db_cursor=None,
            store_in_db=False,
            num_pages=num_pages,
            return_unrated=True,
            attach_liked_flag=include_liked_items,
        )
    )
    loop.run_until_complete(future)

    user_ratings = [x for x in future.result()]
    explicit_user_ratings = [x for x in user_ratings if x["rating_val"] >= 0]
    if data_opt_in:
        # Remove "liked" flag, if it exists, before sending to DB
        sanitized_ratings = [
            {k: v for k, v in r.items() if k != "liked"} for r in explicit_user_ratings
        ]
        send_to_db(username, display_name, user_ratings=sanitized_ratings)

    if include_liked_items:
        # user_ratings_with_likes = [
        #     x for x in user_ratings if x["rating_val"] >= 0 or x["liked"]
        # ]
        attach_synthetic_ratings(user_ratings, global_mean=8)

    return user_ratings, "success"


def send_to_db(username, display_name, user_ratings):
    # Connect to MongoDB client
    db_name, client = connect_to_db()

    # Find letterboxd database and user collection
    db = client[db_name]
    users = db.users
    ratings = db.ratings
    movies = db.movies

    user = {
        "username": username.lower(),
        "display_name": display_name,
        "num_reviews": len(user_ratings),
        "last_attempted": datetime.datetime.now(datetime.timezone.utc),
        "last_updated": datetime.datetime.now(datetime.timezone.utc),
    }

    users.update_one({"username": user["username"]}, {"$set": user}, upsert=True)

    upsert_ratings_operations = []
    upsert_movies_operations = []

    for rating in user_ratings:
        upsert_ratings_operations.append(
            ReplaceOne(
                {"user_id": username, "movie_id": rating["movie_id"]},
                rating,
                upsert=True,
            )
        )

        upsert_movies_operations.append(
            UpdateOne(
                {"movie_id": rating["movie_id"]},
                {"$set": {"movie_id": rating["movie_id"]}},
                upsert=True,
            )
        )

    safe_commit_ops(ratings, upsert_movies_operations)
    safe_commit_ops(movies, upsert_movies_operations)

    return


if __name__ == "__main__":
    username = "samlearner"
    get_user_data(username)
