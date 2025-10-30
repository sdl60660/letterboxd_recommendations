#!/usr/local/bin/python3.12

import os
import pickle
import numpy as np
import random

from model import Model


if os.getcwd().endswith("data_processing"):
    from db_connect import connect_to_db
    from utils.utils import explicit_exclude_list
    from get_user_ratings import get_user_data

else:
    from data_processing.db_connect import connect_to_db
    from data_processing.utils.utils import explicit_exclude_list
    from data_processing.get_user_ratings import get_user_data


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


def get_movie_data(top_n):
    movie_fields = [
        "image_url",
        "movie_id",
        "movie_title",
        "year_released",
        "genres",
        "original_language",
        "popularity",
        "runtime",
        "release_date",
        "content_type"
    ]

    db_name, client = connect_to_db()
    db = client[db_name]
    movie_data = {
        x["movie_id"]: {k: v for k, v in x.items() if k in movie_fields}
        for x in db.movies.find({"movie_id": {"$in": [x[0] for x in top_n]}})
    }

    return movie_data


def load_compressed_model(path):
    m = Model.from_npz(path)
    assert m.qi.shape[1] == m.n_factors and m.pu.shape[1] == m.n_factors
    return m


def run_model(
    username, algo, user_data, sample_movie_list, num_recommendations=20, fold_in=True
):
    try:
        rating_thresholds = [algo.rating_min, algo.rating_max]
    except AttributeError:
        rating_thresholds = [1.0, 10.0]

    rated_events, seen_ids = split_user_events(user_data, rating_thresholds[0], rating_thresholds[1])

    if fold_in:
        algo = algo.update_algo(username, rated_events)

    prediction_set = get_prediction_set(username, seen_ids, sample_movie_list)
    predictions = algo.test(prediction_set)
    top_n = get_top_n(predictions, num_recommendations)
    
    movie_data = get_movie_data(top_n)

    return_object = [
        {
            "movie_id": x[0],
            "predicted_rating": round(x[1], 3),
            "unclipped_rating": round(x[1], 3),
            "movie_data": movie_data[x[0]],
        }
        for x in top_n
    ]

    for i, prediction in enumerate(return_object):
        if prediction["predicted_rating"] == 10:
            return_object[i]["unclipped_rating"] = float(
                algo.predict(username, prediction["movie_id"], clip=False).est
            )

    # filter out any tv shows (based on TMDB data)
    # this conditional is a little funky for now because the "content_type" field hasn't finished backfilling in the movies collection
    return_object = [x for x in return_object if 'content_type' not in x['movie_data'].keys() or x['movie_data']['content_type'] != 'tv']
    return_object.sort(key=lambda x: (x["unclipped_rating"]), reverse=True)

    return return_object


def main(username, sample_size = 1000000, fold_in=False, num_recommendations=25):
    # algo = load(f"models/model_{sample_size}.npz")[1]
    algo = load_compressed_model(f"models/model_{sample_size}.npz")

    with open(f"data/movie_lists/sample_movie_list_{sample_size}.txt", "rb") as fp:
        sample_movie_list = pickle.load(fp)

    if fold_in == True:
        user_data = get_user_data(username)[0]

    else:
        with open("models/user_data.txt", "rb") as fp:
            user_data = pickle.load(fp)
        
    recs = run_model(username, algo, user_data, sample_movie_list, num_recommendations, fold_in)
    print([{'movie': x['movie_id'], 'rating': x['predicted_rating']} for x in recs[:10]])
    return recs


if __name__ == "__main__":
    main("samlearner", sample_size=1000000, fold_in=True, num_recommendations=25)
