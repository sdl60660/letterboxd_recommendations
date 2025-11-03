#!/usr/local/bin/python3.12

import json
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.stats.distributions as dists
from build_model import get_dataset, train_model
from model import Model
from run_model import run_model
from surprise import SVD, Prediction, accuracy
from surprise.model_selection import KFold, RandomizedSearchCV, cross_validate
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from utils.config import random_seed, sample_sizes


def precision_recall_at_k(predictions, k=20, threshold=6):
    """Return precision and recall at k metrics for each user"""

    # Map predcitions to users
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()

    for uid, user_ratings in user_est_true.items():
        # Sort users by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined/set to 0 here
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined/set to 0 here
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


def get_datasets(sample_sizes):
    datasets = []
    for sample_size in sample_sizes:
        df = pd.read_parquet(
            f"data/training_data_samples/training_data_{sample_size}.parquet"
        )
        dataset = get_dataset(df)
        datasets.append({"sample_size": sample_size, "dataset": dataset})

    return datasets


def evaluate_config(dataset, model, params={}, cv_folds=3):
    data = dataset["dataset"]
    sample_size = dataset["sample_size"]

    print(f"Testing model: {model['name']} at sample size {sample_size}...")
    algo, training_set = train_model(
        data=data, model=model["model"], params=params, run_cv=False
    )

    kf = KFold(cv_folds)
    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(testset)
        precisions, recalls = precision_recall_at_k(predictions, k=50, threshold=7)

    mean_precision = sum(prec for prec in precisions.values()) / len(precisions)
    mean_recall = sum(rec for rec in recalls.values()) / len(recalls)

    cv = cross_validate(
        algo, data, measures=["RMSE", "MAE", "FCP"], cv=cv_folds, verbose=False
    )

    eval_row = {
        "model": model["name"],
        "sample_size": sample_size,
        "params": params,
        "RMSE": cv["test_rmse"].mean(),
        "Precision@K": mean_precision,
        "Recall@K": mean_recall,
    }

    return eval_row


def create_user_test_train_sets(
    user_data_df: pd.DataFrame,
    test_users,
    user_train_test_split: float = 0.7,
    random_seed: int = random_seed,
    min_test_items: int = 10,
    min_explicit_train_items: int = 20,
):
    output_set = []
    # has_explicit_rating = user_data_df["rating_val"].ge(0)

    for user_id in test_users:
        user_data = user_data_df[user_data_df["user_id"] == user_id]

        # Split to train/test (row-level)
        user_train = user_data.sample(
            frac=user_train_test_split, random_state=random_seed
        )
        user_test = user_data.loc[~user_data.index.isin(user_train.index)]

        # --- test set: explicit-only ---
        user_test_explicit = user_test[user_test["rating_val"] >= 0].copy()
        if len(user_test_explicit) < min_test_items:
            # Not enough ground-truth to evaluate—skip this user
            continue

        # --- train (explicit-only) ---
        user_train_explicit = user_train[user_train["rating_val"] >= 0].copy()
        if len(user_train_explicit) < min_explicit_train_items:
            # Ensure the explicit-only baseline has enough signal
            continue

        # --- train_with_likes: project synthetic into rating_val where needed ---
        user_train_with_synth = user_train.copy()
        # Keep original rating for potential debugging down the line
        user_train_with_synth["explicit_rating_val"] = user_train_with_synth[
            "rating_val"
        ]

        # Fill rating_val from synthetic where no rating_val (< 0), liked is True, has synthetic_rating_val
        mask_unrated = user_train_with_synth["rating_val"].lt(0)
        mask_liked = user_train_with_synth.get("liked", False).astype(bool)
        mask_has_synth = user_train_with_synth.get("synthetic_rating_val", -1).ge(0)

        fill_mask = mask_unrated & mask_liked & mask_has_synth
        user_train_with_synth.loc[fill_mask, "rating_val"] = user_train_with_synth.loc[
            fill_mask, "synthetic_rating_val"
        ]
        user_train_with_synth = user_train_with_synth[
            user_train_with_synth["rating_val"] >= 0
        ].copy()

        # let's also validate here that the user's training set without the synthetic vals has at least ~20 items
        # to avoid cases where a set of almost entirely liked/synthetic items leaves us with almost no "true" ratings to
        # use in training for the user
        if len(user_test_explicit) < min_test_items:
            continue

        all_movies = user_data["movie_id"].unique()

        output_set.append(
            {
                "username": user_id,
                "train": user_train_explicit.to_dict(orient="records"),
                "train_with_likes": user_train_with_synth.to_dict(orient="records"),
                "test": user_test_explicit.to_dict(orient="records"),
                "test_key": user_test_explicit.set_index("movie_id").to_dict(
                    orient="index"
                ),
                "movie_set": set(all_movies),
            }
        )

    return output_set


def eval_fold_in_user(user_data_set, model):
    username = user_data_set["username"]
    user_data = user_data_set["train"]
    sample_movie_list = user_data_set["movie_set"]
    test_key = user_data_set["test_key"]

    results = run_model(
        username=username,
        algo=model,
        user_data=user_data,
        sample_movie_list=sample_movie_list,
        movie_data=None,
        num_recommendations=len(sample_movie_list),
        fold_in=True,
    )

    predictions = [
        Prediction(
            uid=username,
            iid=x["movie_id"],
            r_ui=test_key[x["movie_id"]]["rating_val"],
            est=x["predicted_rating"],
            details=None,
        )
        for x in results
        if x["movie_id"] in test_key
    ]

    rmse_value = accuracy.rmse(predictions, verbose=False)
    precisions, recalls = precision_recall_at_k(predictions, k=50, threshold=7)

    precision = sum(prec for prec in precisions.values()) / len(precisions)
    recall = sum(rec for rec in recalls.values()) / len(recalls)

    user_metrics = {
        "rmse": rmse_value,
        "precision": precision,
        "recall": recall,
        "total_test_predictions": len(predictions),
    }

    return user_metrics


def eval_param_set_fold_in(base_model, params, training_dataset, user_data_sets):
    # train_model returns a Surprise algo in your codebase; if it returns (algo, trainset), unpack accordingly
    algo = train_model(training_dataset, base_model, params=params, run_cv=False)
    model = Model.from_surprise(algo)

    sum_rmse = 0.0
    sum_prec = 0.0
    sum_rec = 0.0
    sum_preds = 0

    for user_data_set in tqdm(user_data_sets, desc="Fold-in users", leave=False):
        user_metrics = eval_fold_in_user(user_data_set, model)
        n = user_metrics["total_test_predictions"]
        sum_rmse += user_metrics["rmse"] * n
        sum_prec += user_metrics["precision"] * n
        sum_rec += user_metrics["recall"] * n
        sum_preds += n

    if sum_preds == 0:
        # Guard against divide-by-zero (e.g., empty test sets)
        return {
            "fold_in_rmse": float("nan"),
            "fold_in_precision@k": float("nan"),
            "fold_in_recall@k": float("nan"),
        }

    param_set_metrics = {
        "fold_in_rmse": sum_rmse / sum_preds,
        "fold_in_precision@k": sum_prec / sum_preds,
        "fold_in_recall@k": sum_rec / sum_preds,
    }
    return param_set_metrics


def add_foldin_ranks(fold_in_cols_df: pd.DataFrame) -> pd.DataFrame:
    df = fold_in_cols_df.copy()

    # Handle NaNs so they rank to the bottom appropriately
    # (RMSE NaN -> worst; Precision/Recall NaN -> worst)
    rmse_series = df["fold_in_rmse"].fillna(np.inf)
    prec_series = df["fold_in_precision@k"].fillna(-np.inf)
    rec_series = df["fold_in_recall@k"].fillna(-np.inf)

    # Individual ranks (1 = best). Use method='min' to match scikit-learn’s style.
    df["rank_foldin_rmse"] = rmse_series.rank(method="min", ascending=True).astype(int)
    df["rank_foldin_precision@k"] = prec_series.rank(
        method="min", ascending=False
    ).astype(int)
    df["rank_foldin_recall@k"] = rec_series.rank(method="min", ascending=False).astype(
        int
    )

    # A composite rank turned with these weights
    w_rmse, w_prec, w_rec = 0.6, 0.3, 0.1
    df["rank_foldin_composite"] = (
        (
            w_rmse * df["rank_foldin_rmse"]
            + w_prec * df["rank_foldin_precision@k"]
            + w_rec * df["rank_foldin_recall@k"]
        )
        .rank(method="min", ascending=True)
        .astype(int)
    )

    return df


def eval_fold_in(
    training_df, testing_df, base_model, param_set_df, num_test_users=1500
):
    # Parse JSON-ish params column safely
    params_set = [
        json.loads(x.replace("'", '"')) if isinstance(x, str) else x
        for x in param_set_df["params"].tolist()
    ]

    # user split
    # all_model_users = training_df["user_id"].unique()
    all_model_movies = training_df["movie_id"].unique()
    test_users = testing_df["user_id"].unique()
    # training_users = all_users[-num_test_users:]
    training_data = training_df[~training_df["user_id"].isin(test_users)]
    test_data = testing_df[testing_df["movie_id"].isin(all_model_movies)]

    # build training dataset for Surprise
    training_dataset = get_dataset(training_data)

    # per-user train/test splits to fold-in
    user_data_sets = create_user_test_train_sets(
        user_data_df=test_data, test_users=test_users
    )

    fold_in_eval_rows = []
    for i, params in enumerate(tqdm(params_set, desc="Param sets")):
        param_metrics = eval_param_set_fold_in(
            base_model, params, training_dataset, user_data_sets
        )
        fold_in_eval_rows.append(param_metrics)

    fold_in_cols_df = pd.DataFrame(fold_in_eval_rows)
    fold_in_cols_df = add_foldin_ranks(fold_in_cols_df)

    return fold_in_cols_df


def run_grid_search(model, dataset, num_candidates, cv_folds=3):
    param_dists = {
        "n_factors": dists.randint(100, 250),
        "n_epochs": dists.randint(40, 80),
        "lr_all": dists.uniform(0.003, 0.008),
        "reg_qi": dists.uniform(0.01, 0.04),
        "reg_pu": dists.uniform(0.01, 0.04),
        "reg_bu": dists.uniform(0.03, 0.08),
        "reg_bi": dists.uniform(0.08, 0.25),
        "init_std_dev": dists.uniform(0.05, 0.25),
    }

    rand_search = RandomizedSearchCV(
        model,
        param_dists,
        n_iter=num_candidates,
        measures=["rmse", "mae"],
        cv=cv_folds,
        n_jobs=cv_folds,
        joblib_verbose=0,
    )

    with tqdm_joblib(tqdm(total=num_candidates * cv_folds, desc="RandomizedSearchCV")):
        rand_search.fit(dataset)

    results_df = pd.DataFrame.from_dict(rand_search.cv_results)
    results_df_cols = [
        "mean_test_rmse",
        "std_test_rmse",
        "rank_test_rmse",
        "mean_test_mae",
        "std_test_mae",
        "rank_test_mae",
        "mean_fit_time",
        "params",
    ]
    results_df = results_df[results_df_cols]
    results_df.to_csv("./models/eval_results/model_param_test_results.csv", index=False)

    best_params = rand_search.best_params["rmse"]
    return best_params


def export_fold_in_eval_data(fold_in_cols_df, param_eval_df):
    best_foldin_row = fold_in_cols_df.loc[fold_in_cols_df["rank_foldin_rmse"].idxmin()]
    best_params = json.loads(
        param_eval_df.loc[best_foldin_row.name, "params"].replace("'", '"')
    )
    with open("./models/eval_results/best_svd_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Best params (by fold-in RMSE): {best_params}")

    # merge and write combined results
    rich_param_eval_df = pd.concat(
        [param_eval_df.reset_index(drop=True), fold_in_cols_df.reset_index(drop=True)],
        axis=1,
    )
    rich_param_eval_df.to_csv(
        "./models/eval_results/model_param_test_results_with_foldin.csv", index=False
    )


def main():
    np.random.seed(random_seed)
    random.seed(random_seed)

    sample_size_index = 0
    num_candidates = 80

    models = [{"name": "SVD", "model": SVD}]
    datasets = get_datasets(sample_sizes)
    training_sample_df = pd.read_parquet(
        f"data/training_data_samples/training_data_{sample_sizes[sample_size_index]}.parquet"
    )

    # best_params = run_grid_search(
    #     models[0]["model"], datasets[sample_size_index]["dataset"], num_candidates
    # )
    # with open("./models/eval_results/best_svd_params.json", "w") as f:
    #     json.dump(best_params, f)

    param_eval_df = pd.read_csv("./models/eval_results/model_param_test_results.csv")
    test_user_data = pd.read_parquet("./testing/test_user_data.parquet")
    fold_in_cols_df = eval_fold_in(
        training_df=training_sample_df,
        testing_df=test_user_data,
        base_model=models[0]["model"],
        param_set_df=param_eval_df,
    )
    export_fold_in_eval_data(fold_in_cols_df, param_eval_df)


if __name__ == "__main__":
    main()
