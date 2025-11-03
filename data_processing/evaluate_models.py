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
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Prediction, accuracy
from surprise.model_selection import KFold, RandomizedSearchCV, cross_validate
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from utils.config import random_seed, sample_sizes

BEST_PARAMS_FILEPATH = "./models/eval_results/best_svd_params.json"
EVAL_RESULTS_FILEPATH = "./models/eval_results/model_param_test_results.csv"
RICH_EVAL_RESULTS_FILEPATH = "./models/eval_results/model_param_rich_test_results.csv"
TEST_USER_DATASET_FILEPATH = "./testing/test_user_data.parquet"


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

    print(f"Testing model: {model['name']} at sample size {sample_size:,}...")
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

    cv = cross_validate(algo, data, measures=["RMSE"], cv=cv_folds, verbose=False)

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

        user_train_with_synth["rating_val"] = user_train_with_synth[
            "rating_val"
        ].astype(float)
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


def eval_fold_in_user(user_data_set, model, top_k=50, explicit_ratings_only=True):
    username = user_data_set["username"]

    if explicit_ratings_only:
        user_data = user_data_set["train"]
    else:
        user_data = user_data_set["train_with_likes"]

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
    precisions, recalls = precision_recall_at_k(predictions, k=top_k, threshold=7)

    precision = sum(prec for prec in precisions.values()) / len(precisions)
    recall = sum(rec for rec in recalls.values()) / len(recalls)

    top_k_recs = [x.iid for x in predictions[:top_k]]

    user_metrics = {
        "rmse": rmse_value,
        "precision": precision,
        "recall": recall,
        "total_test_predictions": len(predictions),
        "top_k_recs": top_k_recs,
        "user_id": username,
    }

    return user_metrics


def _safe_weighted_means(sums):
    # sums: dict with keys rmse, prec, rec, preds
    if sums["preds"] == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        sums["rmse"] / sums["preds"],
        sums["prec"] / sums["preds"],
        sums["rec"] / sums["preds"],
    )


def eval_param_set_fold_in(
    base_model, params, training_dataset, user_data_sets, all_model_movies
):
    # Train the Surprise model once
    algo = train_model(training_dataset, base_model, params=params, run_cv=False)
    model = Model.from_surprise(algo)

    # Accumulators for both variants of user fold-in testing
    # (explicit rating_vals only/including synthetic ratings based on likes)
    sums_explicit = {"rmse": 0.0, "prec": 0.0, "rec": 0.0, "preds": 0}
    sums_withlikes = {"rmse": 0.0, "prec": 0.0, "rec": 0.0, "preds": 0}

    personalization_testing_data_explicit = {}
    personalization_testing_data_with_likes = {}

    for user_data_set in tqdm(user_data_sets, desc="Fold-in users", leave=False):
        # ---- explicit-only ----
        m_exp = eval_fold_in_user(user_data_set, model, explicit_ratings_only=True)
        n_exp = m_exp["total_test_predictions"]
        sums_explicit["rmse"] += m_exp["rmse"] * n_exp
        sums_explicit["prec"] += m_exp["precision"] * n_exp
        sums_explicit["rec"] += m_exp["recall"] * n_exp
        sums_explicit["preds"] += n_exp

        personalization_testing_data_explicit[m_exp["user_id"]] = m_exp["top_k_recs"]

        # ---- explicit + likes/synthetic ----
        m_lik = eval_fold_in_user(user_data_set, model, explicit_ratings_only=False)
        n_lik = m_lik["total_test_predictions"]
        sums_withlikes["rmse"] += m_lik["rmse"] * n_lik
        sums_withlikes["prec"] += m_lik["precision"] * n_lik
        sums_withlikes["rec"] += m_lik["recall"] * n_lik
        sums_withlikes["preds"] += n_lik

        personalization_testing_data_with_likes[m_lik["user_id"]] = m_lik["top_k_recs"]

    # Weighted means (by # predictions)
    rmse_e, prec_e, rec_e = _safe_weighted_means(sums_explicit)
    rmse_l, prec_l, rec_l = _safe_weighted_means(sums_withlikes)

    personalization_e = personalization_score(
        personalization_testing_data_explicit, all_model_movies
    )
    personalization_l = personalization_score(
        personalization_testing_data_with_likes, all_model_movies
    )

    return {
        # explicit-only
        "fold_in_rmse_explicit": rmse_e,
        "fold_in_precision@k_explicit": prec_e,
        # "fold_in_recall@k_explicit": rec_e,
        "personalization_score_explicit": personalization_e,
        # explicit + likes/synthetic
        "fold_in_rmse_with_likes": rmse_l,
        "fold_in_precision@k_with_likes": prec_l,
        # "fold_in_recall@k_with_likes": rec_l,
        "personalization_score_with_likes": personalization_l,
    }


def _rank(series: pd.Series, ascending: bool) -> pd.Series:
    # NaNs -> worst
    filler = np.inf if ascending else -np.inf
    return series.fillna(filler).rank(method="min", ascending=ascending).astype(int)


def add_foldin_ranks(fold_in_cols_df: pd.DataFrame) -> pd.DataFrame:
    df = fold_in_cols_df.copy()

    # --- explicit-only ranks ---
    df["rank_foldin_rmse_explicit"] = _rank(df["fold_in_rmse_explicit"], True)
    df["rank_foldin_precision@k_explicit"] = _rank(
        df["fold_in_precision@k_explicit"], False
    )
    df["rank_personalization_explicit"] = _rank(
        df["personalization_score_explicit"], False
    )

    # --- with-likes ranks ---
    df["rank_foldin_rmse_with_likes"] = _rank(df["fold_in_rmse_with_likes"], True)
    df["rank_foldin_precision@k_with_likes"] = _rank(
        df["fold_in_precision@k_with_likes"], False
    )
    df["rank_personalization_with_likes"] = _rank(
        df["personalization_score_with_likes"], False
    )

    # A composite rank will be tuned with these weights
    # (though this isn't actually used for selection at the moment)
    # w_rmse, w_prec, w_rec = 0.6, 0.3, 0.1
    w_rmse, w_prec = 0.7, 0.3

    df["rank_foldin_composite_explicit"] = (
        (
            w_rmse * df["rank_foldin_rmse_explicit"]
            + w_prec * df["rank_foldin_precision@k_explicit"]
            # + w_rec * df["rank_foldin_recall@k_explicit"]
        )
        .rank(method="min", ascending=True)
        .astype(int)
    )

    df["rank_foldin_composite_with_likes"] = (
        (
            w_rmse * df["rank_foldin_rmse_with_likes"]
            + w_prec * df["rank_foldin_precision@k_with_likes"]
            # + w_rec * df["rank_foldin_recall@k_with_likes"]
        )
        .rank(method="min", ascending=True)
        .astype(int)
    )

    # Assigning a primary column for model selection downstream of here
    df["rank_foldin_primary"] = df["rank_foldin_rmse_with_likes"]

    return df[
        [
            "rank_foldin_primary",
            "rank_foldin_rmse_explicit",
            "rank_foldin_rmse_with_likes",
            "rank_personalization_explicit",
            "rank_personalization_with_likes",
        ]
    ]


def personalization_score(recommendations, all_items):
    """
    recommendations: dict {user_id: [list of recommended movie_ids]}
    """
    all_items = list(all_items)
    item_to_idx = {item: i for i, item in enumerate(all_items)}

    # Create a binary recommendation matrix (users x items)
    user_vectors = []
    for user, recs in recommendations.items():
        vec = np.zeros(len(all_items))
        for item in recs:
            if item in item_to_idx:
                vec[item_to_idx[item]] = 1
        user_vectors.append(vec)

    user_vectors = np.array(user_vectors)
    sim_matrix = cosine_similarity(user_vectors)
    # Average similarity between different users
    upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    return 1 - np.mean(upper_tri)


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
            base_model, params, training_dataset, user_data_sets, all_model_movies
        )
        fold_in_eval_rows.append(param_metrics)

    fold_in_cols_df = pd.DataFrame(fold_in_eval_rows)
    foldin_ranks = add_foldin_ranks(fold_in_cols_df)

    fold_in_cols_df = fold_in_cols_df.join(foldin_ranks)

    return fold_in_cols_df


def _dump_best_params(best_params, filepath=BEST_PARAMS_FILEPATH):
    with open(filepath, "w") as f:
        json.dump(best_params, f, indent=2)


def get_current_champion_results(model, dataset, params, cv_folds=3):
    best_model = model(**params)
    cv = cross_validate(
        best_model, dataset, measures=["rmse"], cv=cv_folds, verbose=False
    )
    mean_rmse = np.mean(cv["test_rmse"])
    std_rmse = np.std(cv["test_rmse"])
    mean_fit = np.mean(cv["fit_time"])

    extra_row = {
        "mean_test_rmse": mean_rmse,
        "std_test_rmse": std_rmse,
        "rank_test_rmse": np.nan,  # will re-rank after
        "mean_fit_time": mean_fit,
        "params": params,
    }

    return extra_row


def run_grid_search(model, dataset, num_candidates, current_best_params, cv_folds=3):
    param_dists = {
        "n_factors": dists.randint(100, 250),
        "n_epochs": dists.randint(40, 80),
        "lr_all": dists.uniform(0.003, 0.008),
        "reg_qi": dists.uniform(0.01, 0.04),
        "reg_pu": dists.uniform(0.01, 0.04),
        "reg_bu": dists.uniform(0.03, 0.08),
        "reg_bi": dists.uniform(0.08, 0.2),
        "init_std_dev": dists.uniform(0.05, 0.25),
    }

    n_iter = num_candidates
    if current_best_params:
        n_iter = num_candidates - 1

    rand_search = RandomizedSearchCV(
        model,
        param_dists,
        n_iter=n_iter,
        measures=["rmse"],
        cv=cv_folds,
        n_jobs=cv_folds,
        joblib_verbose=0,
        # we want deterministic random seed for things like the dataset split, but...
        # ideally we actually get different random distributions of params on each run here
        random_state=None,
    )

    with tqdm_joblib(tqdm(total=n_iter * cv_folds, desc="RandomizedSearchCV")):
        rand_search.fit(dataset)

    results_df = pd.DataFrame.from_dict(rand_search.cv_results)
    results_df_cols = [
        "mean_test_rmse",
        "std_test_rmse",
        "rank_test_rmse",
        "mean_fit_time",
        "params",
    ]
    results_df = results_df[results_df_cols]

    # if we've passed in the "current champion", calculate its results with this new dataset and add it to the set/re-rank
    # we can't just re-use results because the sample data may have changed, but this allows us to not select a worse param set
    # as our new "best" set just due to the randomized search and this will be passed into the fold-in eval after
    if current_best_params:
        extra_row = get_current_champion_results(
            model, dataset, params=current_best_params, cv_folds=cv_folds
        )
        results_df = pd.concat(
            [results_df, pd.DataFrame([extra_row])], ignore_index=True
        )
        results_df["rank_test_rmse"] = results_df["mean_test_rmse"].rank(method="min")

    results_df.to_csv(EVAL_RESULTS_FILEPATH, index=False)

    best_params = results_df.loc[results_df["mean_test_rmse"].idxmin(), "params"]
    _dump_best_params(best_params)

    return best_params


def export_fold_in_eval_data(fold_in_cols_df, param_eval_df):
    best_foldin_row = fold_in_cols_df.loc[
        fold_in_cols_df["rank_foldin_primary"].idxmin()
    ]
    best_params = json.loads(
        param_eval_df.loc[best_foldin_row.name, "params"].replace("'", '"')
    )
    _dump_best_params(best_params)

    # merge and write combined results
    rich_param_eval_df = pd.concat(
        [param_eval_df.reset_index(drop=True), fold_in_cols_df.reset_index(drop=True)],
        axis=1,
    )
    rich_param_eval_df.to_csv(RICH_EVAL_RESULTS_FILEPATH, index=False)


def main():
    # Set consistent random seed
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Config vars
    sample_size_index = 0
    num_candidates = 3

    # Load in data
    models = [{"name": "SVD", "model": SVD}]
    datasets = get_datasets(sample_sizes)
    training_sample_df = pd.read_parquet(
        f"data/training_data_samples/training_data_{sample_sizes[sample_size_index]}.parquet"
    )
    test_user_data = pd.read_parquet(TEST_USER_DATASET_FILEPATH)
    with open(BEST_PARAMS_FILEPATH, "r") as f:
        current_best_params = json.load(f)

    # 1) Run initial param grid search and evaluate against current best params
    #   this will give us a set of randomized params and results to then test with fold-in
    run_grid_search(
        models[0]["model"],
        datasets[sample_size_index]["dataset"],
        num_candidates,
        current_best_params,
    )

    # 2) Test candidate param sets over fold-in methodology and attach additional metrics
    param_eval_df = pd.read_csv(EVAL_RESULTS_FILEPATH)
    fold_in_cols_df = eval_fold_in(
        training_df=training_sample_df,
        testing_df=test_user_data,
        base_model=models[0]["model"],
        param_set_df=param_eval_df,
    )
    export_fold_in_eval_data(fold_in_cols_df, param_eval_df)


if __name__ == "__main__":
    main()
