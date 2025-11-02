#!/usr/local/bin/python3.12

import json
import os
import pickle
import random

import pandas as pd

if os.getcwd().endswith("data_processing"):
    from get_user_ratings import get_user_data
    from model import Model
    from utils.utils import explicit_exclude_list, get_rich_movie_data

else:
    from data_processing.get_user_ratings import get_user_data
    from data_processing.model import Model
    from data_processing.utils.utils import explicit_exclude_list, get_rich_movie_data


def split_user_events(user_data, rating_min=1.0, rating_max=10.0):
    rated = []
    seen = set()

    for x in user_data:
        mid = x["movie_id"]
        val = float(x["rating_val"])
        seen.add(mid)

        if rating_min <= val <= rating_max:
            rated.append({"movie_id": mid, "rating_val": val})

    return rated, seen


def get_top_n(predictions, n=20):
    top_n = [(iid, est) for uid, iid, true_r, est, _ in predictions]
    top_n.sort(key=lambda x: (x[1], random.random()), reverse=True)

    return top_n[:n]


def get_prediction_set(username, user_watched_list, sample_movie_list):
    exclude_list = set(user_watched_list).union(set(explicit_exclude_list))
    # unwatched_movies = [x for x in sample_movie_list if x not in user_watched_list]
    valid_unwatched_movies = [x for x in sample_movie_list if x not in exclude_list]
    prediction_set = [(username, x, 0) for x in valid_unwatched_movies]

    return prediction_set


def get_movie_data(sample_movie_list, sample_size):
    if os.getcwd().endswith("data_processing"):
        datafile_path = f"data/rich_movie_data/sample_movie_data_{sample_size}.parquet"
    else:
        datafile_path = f"data_processing/data/rich_movie_data/sample_movie_data_{sample_size}.parquet"

    if os.path.exists(datafile_path):
        try:
            movie_data = pd.read_parquet(datafile_path)
        except Exception as e:
            print(f"Error loading rich movie data Parquet file '{datafile_path}': {e}")
    else:
        movie_data = get_rich_movie_data(
            movie_ids=sample_movie_list, output_path=datafile_path
        )

    # movie_data = movie_data.set_index('movie_id', drop=False).to_dict('index')
    movie_data = json.loads(
        movie_data.set_index("movie_id", drop=False).to_json(
            orient="index", date_format="iso", default_handler=str
        )
    )

    return movie_data


def load_compressed_model(path):
    m = Model.from_npz(path)
    assert m.qi.shape[1] == m.n_factors
    return m


def run_model(
    username,
    algo,
    user_data,
    sample_movie_list,
    movie_data=None,
    num_recommendations=20,
    fold_in=True,
    verbose=False,
):
    rating_min, rating_max = (
        getattr(algo, "rating_min", 1.0),
        getattr(algo, "rating_max", 10.0),
    )
    rated_events, seen_ids = split_user_events(user_data, rating_min, rating_max)

    if fold_in:
        algo = algo.update_algo(username, rated_events)

    prediction_set = get_prediction_set(username, seen_ids, sample_movie_list)
    predictions = algo.test(prediction_set, clip_ratings=False)

    top_n_pairs = get_top_n(predictions, num_recommendations)

    results = []
    for movie_id, est_unclipped in top_n_pairs:
        est_clipped = min(rating_max, max(rating_min, est_unclipped))

        output_entry = {
            "movie_id": movie_id,
            "predicted_rating": round(est_clipped, 3),
            "unclipped_rating": round(est_unclipped, 3),
        }

        if movie_data:
            output_entry["movie_data"] = movie_data[movie_id]

        results.append(output_entry)

    # filter out any tv shows (based on TMDB data)
    # this conditional is a little funky for now because the "content_type" field hasn't finished backfilling in the movies collection
    if movie_data:
        results = [
            x
            for x in results
            if "content_type" not in x["movie_data"].keys()
            or x["movie_data"]["content_type"] != "tv"
        ]

    results.sort(key=lambda x: (x["unclipped_rating"]), reverse=True)

    if verbose:
        print(f"Top estimated results for user: {username}")
        print("=====================================")
        for item in results[:30]:
            print(f"{item['movie_id']}: {item['predicted_rating']}")

    return results


def main(
    username,
    sample_size=1000000,
    fold_in=True,
    num_recommendations=25,
    use_cached_user_data=False,
    verbose=True,
):
    algo = load_compressed_model(f"models/model_{sample_size}.npz")

    with open(f"data/movie_lists/sample_movie_list_{sample_size}.txt", "rb") as fp:
        sample_movie_list = pickle.load(fp)

    if use_cached_user_data == True and os.path.exists("testing/user_data.txt"):
        with open("testing/user_data.txt", "rb") as fp:
            user_data = pickle.load(fp)
    else:
        user_data = get_user_data(username)[0]

    print(user_data)

    # will use cached file if it exsits, or grab/cache if it doesn't
    movie_data = get_movie_data(sample_movie_list, sample_size)

    recs = run_model(
        username,
        algo,
        user_data,
        sample_movie_list,
        movie_data,
        num_recommendations,
        fold_in,
        verbose=verbose,
    )
    return recs


if __name__ == "__main__":
    main(
        "samlearner",
        sample_size=2000000,
        fold_in=True,
        num_recommendations=25,
        use_cached_user_data=True,
        verbose=True,
    )
