#!/usr/local/bin/python3.12

import json
import os
import pickle
import random

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.dump import dump
from surprise.model_selection import cross_validate

if os.getcwd().endswith("data_processing"):
    from get_user_ratings import get_user_data
    from model import Model
    from utils.config import random_seed, sample_sizes

else:
    from data_processing.get_user_ratings import get_user_data
    from data_processing.model import Model
    from data_processing.utils.config import random_seed, sample_sizes


# a global/fallback to use as a default val, based on a traiing run/eval, but this shouldn't ever be used,
# either when this is called from the web server or from commandline
SVD_PARAMS = {
    "lr_all": 0.0062939,
    "n_epochs": 69,
    "n_factors": 215,
    "reg_bi": 0.31902932,
    "reg_bu": 0.03736959,
    "reg_pu": 0.0458803,
    "reg_qi": 0.0457921065,
}


def get_dataset(df, rating_scale=(1, 10), cols=["user_id", "movie_id", "rating_val"]):
    # Surprise dataset loading
    reader = Reader(rating_scale=rating_scale)
    data = Dataset.load_from_df(df[cols], reader)
    return data


def prep_concat_dataframe(df, sample_movie_list, user_data):
    user_rated = [
        x
        for x in user_data
        if x["rating_val"] > 0 and x["movie_id"] in sample_movie_list
    ]

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
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Configure algorithm
    algo = model(**params, random_state=random_seed)

    if run_cv:
        cross_validate(algo, data, measures=["RMSE", "MAE", "FCP"], cv=3, verbose=True)

    training_set = data.build_full_trainset()
    algo.fit(training_set)

    return algo


def build_model(
    df,
    sample_movie_list,
    user_data,
    model=SVD,
    params=SVD_PARAMS,
    run_cv=False,
    concat_user_data=True,
):
    if concat_user_data:
        model_data = prep_concat_dataframe(df, sample_movie_list, user_data)
    else:
        model_data = get_dataset(df)

    algo = train_model(model_data, model, params, run_cv)
    # user_watched_list = [x["movie_id"] for x in user_data]

    return algo


def export_model(algo, sample_size, compressed=True, subdirectory_path="models"):
    if compressed:
        model = Model.from_surprise(algo)
        out_path = f"{subdirectory_path}/model_{sample_size}.npz"
        model.to_npz(out_path, items_only=True)
    else:
        dump(
            f"{subdirectory_path}/model_{sample_size}.pkl",
            predictions=None,
            algo=algo,
            verbose=1,
        )


def main(sample_size, params, user_data=[], run_cv=False, concat_user_data=False):
    # Load ratings data
    df = pd.read_parquet(
        f"data/training_data_samples/training_data_{sample_size}.parquet"
    )
    with open(f"data/movie_lists/sample_movie_list_{sample_size}.txt", "rb") as fp:
        sample_movie_list = pickle.load(fp)

    algo = build_model(
        df,
        sample_movie_list,
        user_data,
        SVD,
        params=params,
        run_cv=run_cv,
        concat_user_data=concat_user_data,
    )

    export_model(algo, sample_size, compressed=True)


if __name__ == "__main__":
    with open("models/eval_results/best_svd_params.json", "r") as f:
        svd_params = json.load(f)

    concat_user_data = False
    if concat_user_data:
        user_data = get_user_data("samlearner")[0]
    else:
        user_data = []

    for sample_size in sample_sizes:
        print(f"Building model for: {sample_size:,} rating sample...")
        main(
            sample_size=sample_size,
            params=svd_params,
            user_data=user_data,
            run_cv=False,
            concat_user_data=False,
        )
