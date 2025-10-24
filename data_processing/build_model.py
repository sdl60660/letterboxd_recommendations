#!/usr/local/bin/python3.12

import pandas as pd
import pickle

from surprise import SVD, Reader, Dataset, BaselineOnly

from surprise.model_selection import cross_validate
from surprise.dump import dump

import random
import numpy as np


def build_model(df, sample_movie_list, user_data):
    # print(df.head())

    # Set random seed so that returned recs are always the same for same user with same ratings
    # This might make sense so that results are consistent, or you might want to refresh with different results
    my_seed = 12
    random.seed(my_seed)
    np.random.seed(my_seed)

    user_rated = [x for x in user_data if x["rating_val"] > 0 and x['movie_id'] in sample_movie_list]

    user_df = pd.DataFrame(user_rated)
    df = pd.concat([df, user_df]).reset_index(drop=True)
    df.drop_duplicates(inplace=True)
    del user_df

    # Surprise dataset loading
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(df[["user_id", "movie_id", "rating_val"]], reader)
    del df

    params = {
        # "n_factors": 150,
        # "n_epochs": 20,
        # "lr_all": 0.005,
        # "reg_all": 0.02
    }
    # Configure algorithm
    algo = SVD(**params)

    # bsl_options = {"method": "als", "n_epochs": 10, "reg_u": 1, "reg_i": 1}
    # algo = BaselineOnly(bsl_options=bsl_options)
    # cross_validate(algo, data, measures=['RMSE', 'MAE', 'FCP'], cv=3, verbose=True)

    trainingSet = data.build_full_trainset()
    algo.fit(trainingSet)

    user_watched_list = [x["movie_id"] for x in user_data]

    return algo, user_watched_list


if __name__ == "__main__":
    import os

    if os.getcwd().endswith("data_processing"):
        from get_user_ratings import get_user_data
    else:
        from data_processing.get_user_ratings import get_user_data

    # Load ratings data
    df = pd.read_parquet("data/training_data_samples/training_data_1000000.parquet")
    with open(f"data/movie_lists/sample_movie_list_1000000.txt", "rb") as fp:
        sample_movie_list = pickle.load(fp)

    user_data = get_user_data("samlearner")[0]
    algo, user_watched_list = build_model(df, sample_movie_list, user_data)

    dump("models/mini_model.pkl", predictions=None, algo=algo, verbose=1)
    with open("models/user_watched.txt", "wb") as fp:
        pickle.dump(user_watched_list, fp)
