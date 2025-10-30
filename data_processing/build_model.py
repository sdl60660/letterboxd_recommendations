#!/usr/local/bin/python3.12

import json
import random
import pickle
import numpy as np
import pandas as pd

from surprise import SVD, Reader, Dataset, BaselineOnly

from surprise.model_selection import cross_validate
from surprise.dump import dump

from model import Model

# a global/fallback to use as a default val, based on a traiing run/eval, but this shouldn't ever be used,
# either when this is called from the web server or from commandline
SVD_PARAMS = {"lr_all": 0.0028736, "n_epochs": 63, "n_factors": 114, "reg_all": 0.135171, "reg_bi": 0.281263224}

def get_dataset(df, rating_scale=(1,10), cols=['user_id', 'movie_id', 'rating_val']):
    # Surprise dataset loading
    reader = Reader(rating_scale=rating_scale)
    data = Dataset.load_from_df(df[cols], reader)
    return data


def prep_concat_dataframe(df, sample_movie_list, user_data):
    user_rated = [x for x in user_data if x["rating_val"] > 0 and x['movie_id'] in sample_movie_list]

    user_df = pd.DataFrame(user_rated)
    df = pd.concat([df, user_df]).reset_index(drop=True)
    df.drop_duplicates(inplace=True)
    del user_df

    data = get_dataset(df)
    del df

    return data


def train_model(data, model=SVD, params=SVD_PARAMS, run_cv=False):
    # Set random seed so that returned recs are always the same for same user with same ratings
    # This might make sense so that results are consistent, or you might want to refresh with different results
    my_seed = 12
    random.seed(my_seed)
    np.random.seed(my_seed)

    # Configure algorithm
    algo = model(**params, random_state=my_seed)

    if run_cv:
        cross_validate(algo, data, measures=['RMSE', 'MAE', 'FCP'], cv=3, verbose=True)

    training_set = data.build_full_trainset()
    algo.fit(training_set)

    return algo


def build_model(df, sample_movie_list, user_data, model=SVD, params=SVD_PARAMS, run_cv=False, concat_user_data=True):
    if concat_user_data:
        model_data = prep_concat_dataframe(df, sample_movie_list, user_data)
    else:
        model_data = get_dataset(df)

    algo = train_model(model_data, model, params, run_cv)
    user_watched_list = [x["movie_id"] for x in user_data]

    return algo, user_watched_list


def export_model(algo, sample_size, compressed=True, subdirectory_path="models"):
    if compressed:
        model = Model.from_surprise(algo)
        out_path = f"{subdirectory_path}/model_{sample_size}.npz"
        model.to_npz(out_path, items_only=True)
    else:
        dump(f"{subdirectory_path}/model_{sample_size}.pkl", predictions=None, algo=algo, verbose=1)


if __name__ == "__main__":
    import os
    if os.getcwd().endswith("data_processing"):
        from get_user_ratings import get_user_data
    else:
        from data_processing.get_user_ratings import get_user_data
    
    default_sample_size = 1000000

    # Load ratings data
    df = pd.read_parquet(f"data/training_data_samples/training_data_{default_sample_size}.parquet")
    with open(f"data/movie_lists/sample_movie_list_{default_sample_size}.txt", "rb") as fp:
        sample_movie_list = pickle.load(fp)
    
    with open("models/best_svd_params.json", 'r') as f:
        svd_params = json.load(f)

    user_data = get_user_data("samlearner")[0]
    algo, user_watched_list = build_model(df, sample_movie_list, user_data, SVD, params=svd_params, run_cv=False, concat_user_data=False)

    export_model(algo, default_sample_size, compressed=True)

    with open("models/user_watched.txt", "wb") as fp:
        pickle.dump(user_watched_list, fp)
    
    with open("models/user_data.txt", "wb") as fp:
        pickle.dump(user_data, fp)
