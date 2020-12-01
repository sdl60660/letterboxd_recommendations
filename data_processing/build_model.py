#!/usr/local/bin/python3.9

import pandas as pd
import pickle

from surprise import Dataset
from surprise import Reader
from surprise import SVD
# from surprise.model_selection import GridSearchCV
from surprise.model_selection import cross_validate
from surprise.dump import dump

from config import config
from get_user_ratings import get_user_data


def build_model(username):
    # Load ratings data
    df = pd.read_csv('data/training_data.csv')
    # print(df.head())
 
    user_movies = get_user_data(username)
    user_rated = [x for x in user_movies if x['rating_val'] > 0]

    user_df = pd.DataFrame(user_rated)
    df = pd.concat([df, user_df]).reset_index(drop=True)
    df.drop_duplicates(inplace=True)

    # Surprise dataset loading
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(df[["user_id", "movie_id", "rating_val"]], reader)

    # Configure algorithm
    algo = SVD()
    # cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    trainingSet = data.build_full_trainset()
    algo.fit(trainingSet)

    user_watched_list = [x['movie_id'] for x in user_movies]

    return algo, user_watched_list

if __name__ == "__main__":
    algo, user_watched_list = build_model("samlearner")

    dump("models/mini_model.pkl", predictions=None, algo=algo, verbose=1)
    with open("models/user_watched.txt", "wb") as fp:
        pickle.dump(user_watched_list, fp)
