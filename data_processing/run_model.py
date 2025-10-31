#!/usr/local/bin/python3.12

import os
import pymongo
import pickle
import pandas as pd
import random

from collections import defaultdict

from surprise import Dataset
from surprise import SVD
from surprise import Reader
from surprise.model_selection import GridSearchCV
from surprise import SVD
from surprise.dump import load

if os.getcwd().endswith("data_processing"):
    from db_connect import connect_to_db
    from utils.utils import explicit_exclude_list

else:
    from data_processing.db_connect import connect_to_db
    from data_processing.utils.utils import explicit_exclude_list


# There are some pseudo-TV shows (docs about shows, etc) that people use to log a rating for a TV show
# which I think we want to exclude. For now, this is just a little manual/explicit with the most prominent
# ones, but can maybe find a better way to target later
# (Think I'm going to work on turning this into a default-on frontend filter,
# modeled after the exclusions here: https://letterboxd.com/dave/list/official-top-250-narrative-feature-films/)

def get_top_n(predictions, n=20):
    top_n = [(iid, est) for uid, iid, true_r, est, _ in predictions]
    top_n.sort(key=lambda x: (x[1], random.random()), reverse=True)

    return top_n[:n]


def run_model(
    username, algo, user_watched_list, sample_movie_list, num_recommendations=20
):
    db_name, client = connect_to_db()
    db = client[db_name]

    exclude_list = user_watched_list + explicit_exclude_list
    # unwatched_movies = [x for x in sample_movie_list if x not in user_watched_list]
    valid_unwatched_movies = [x for x in sample_movie_list if x not in exclude_list]
    prediction_set = [(username, x, 0) for x in valid_unwatched_movies]

    predictions = algo.test(prediction_set)

    top_n = get_top_n(predictions, num_recommendations)
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
    movie_data = {
        x["movie_id"]: {k: v for k, v in x.items() if k in movie_fields}
        for x in db.movies.find({"movie_id": {"$in": [x[0] for x in top_n]}})
    }

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


if __name__ == "__main__":
    with open("models/user_watched.txt", "rb") as fp:
        user_watched_list = pickle.load(fp)

    with open(f"data/movie_lists/sample_movie_list_1000000.txt", "rb") as fp:
        sample_movie_list = pickle.load(fp)

    algo = load("models/mini_model.pkl")[1]

    recs = run_model("samlearner", algo, user_watched_list, sample_movie_list, 25)
    print([x['movie_id'] for x in recs])
