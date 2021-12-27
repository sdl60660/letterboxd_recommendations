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
import random

import os

import pymongo

try:
    from .db_config import config
except ImportError:
    config = {
        'MONGO_DB': os.getenv('MONGO_DB', ''),
        'CONNECTION_URL': os.getenv('CONNECTION_URL', '')
    }


def get_top_n(predictions, n=20):
    top_n = [(iid, est) for uid, iid, true_r, est, _ in predictions]
    top_n.sort(key=lambda x: (x[1], random.random()), reverse=True)

    return top_n[:n]


def run_model(username, algo, user_watched_list, threshold_movie_list, num_recommendations=20):
     # Connect to MongoDB Client
    db_name = config["MONGO_DB"]

    if "CONNECTION_URL" in config.keys():
        client = pymongo.MongoClient(config["CONNECTION_URL"], server_api=pymongo.server_api.ServerApi('1'))
    else:
        client = pymongo.MongoClient(f'mongodb+srv://{config["MONGO_USERNAME"]}:{config["MONGO_PASSWORD"]}@cluster0.{config["MONGO_CLUSTER_ID"]}.mongodb.net/{db_name}?retryWrites=true&w=majority')

    db = client[db_name]

    unwatched_movies = [x for x in threshold_movie_list if x not in user_watched_list]
    prediction_set = [(username, x, 0) for x in unwatched_movies]

    predictions = algo.test(prediction_set)
    top_n = get_top_n(predictions, num_recommendations)
    movie_data = {x["movie_id"]: {k:v for k,v in x.items() if k != "_id"} for x in db.movies.find({"movie_id": {"$in": [x[0] for x in top_n]}})}

    # Print the recommended items for user
    # for prediction in top_n:
    #     print(f"{prediction[0]}: {round(prediction[1], 2)}")

    return_object = [{"movie_id": x[0], "predicted_rating": x[1], "unclipped_rating": x[1], "movie_data": movie_data[x[0]] } for x in top_n]

    for i, prediction in enumerate(return_object):
        if prediction['predicted_rating'] == 10:
            return_object[i]['unclipped_rating'] = float(algo.predict(username, prediction["movie_id"], clip=False).est)

    return_object.sort(key=lambda x: (x["unclipped_rating"]), reverse=True)
    return return_object

if __name__ == "__main__":
    with open("models/user_watched.txt", "rb") as fp:
        user_watched_list = pickle.load(fp)

    with open("models/threshold_movie_list.txt", "rb") as fp:
        threshold_movie_list = pickle.load(fp)

    algo = load("models/mini_model.pkl")[1]

    recs = run_model("samlearner", algo, user_watched_list, threshold_movie_list, 25)
    print(recs)