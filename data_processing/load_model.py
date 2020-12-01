#!/usr/local/bin/python3.9

from collections import defaultdict

from surprise import Dataset
from surprise import SVD
from surprise import Reader
from surprise.model_selection import GridSearchCV
from surprise import SVD
from surprise.dump import load

import pickle

import pandas as pd


def get_top_n(predictions, n=20):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n



with open("models/user_watched.txt", "rb") as fp:
    user_watched_list = pickle.load(fp)

with open("models/threshold_movie_list.txt", "rb") as fp:
    threshold_movie_list = pickle.load(fp)

algo = load("models/mini_model.pkl")[1]

username = "samlearner"

unwatched_movies = [x for x in threshold_movie_list if x not in user_watched_list]
prediction_set = [(username, x, 0) for x in unwatched_movies]

# prediction = algo.predict(username, "despicable-me-3")
# print(prediction.est)

predictions = algo.test(prediction_set)
top_n = get_top_n(predictions, n=20)

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    if uid == "samlearner":
        print(uid, [(iid, _) for (iid, _) in user_ratings])