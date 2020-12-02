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


def run_model(username, algo, user_watched_list, threshold_movie_list, num_recommendations=20):
    
    def get_top_n(predictions, n=20):
        top_n = [(iid, est) for uid, iid, true_r, est, _ in predictions]
        top_n.sort(key=lambda x: x[1], reverse=True)

        return top_n[:n]

    unwatched_movies = [x for x in threshold_movie_list if x not in user_watched_list]
    prediction_set = [(username, x, 0) for x in unwatched_movies]

    # prediction = algo.predict(username, "despicable-me-3")
    # print(prediction.est)

    predictions = algo.test(prediction_set)
    top_n = get_top_n(predictions, num_recommendations)

    # Print the recommended items for user
    # for prediction in top_n:
    #     print(f"{prediction[0]}: {round(prediction[1], 2)}")
    return_object = [{"movie_id": x[0], "predicted_rating": x[1]} for x in top_n]
    return return_object

if __name__ == "__main__":
    with open("models/user_watched.txt", "rb") as fp:
        user_watched_list = pickle.load(fp)

    with open("models/threshold_movie_list.txt", "rb") as fp:
        threshold_movie_list = pickle.load(fp)

    algo = load("models/mini_model.pkl")[1]

    run_model("samlearner", algo, user_watched_list, threshold_movie_list, 25)