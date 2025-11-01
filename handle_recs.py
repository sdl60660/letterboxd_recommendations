import pandas as pd
import pickle
import json

from rq import Queue, get_current_job
from rq.job import Job
from rq.registry import FinishedJobRegistry

from data_processing.get_user_ratings import get_user_data
from data_processing.get_user_watchlist import get_watchlist_data
from data_processing.build_model import build_model
from data_processing.run_model import run_model, load_compressed_model, get_movie_data

from surprise.dump import load

from worker import conn


def get_previous_job_from_registry(index=-1):
    q = Queue("high", connection=conn)
    registry = FinishedJobRegistry(queue=q)

    job_id = registry.get_job_ids()[index]
    job = q.fetch_job(job_id)

    return job


def filter_threshold_list(threshold_movie_list, review_count_threshold=2000):
    review_counts = pd.read_csv("data_processing/data/review_counts.csv")
    review_counts = review_counts.loc[review_counts["count"] < review_count_threshold]

    included_movies = review_counts["movie_id"].to_list()
    threshold_movie_list = [x for x in threshold_movie_list if x in included_movies]

    return threshold_movie_list


def get_client_user_data(username, data_opt_in):
    user_data = get_user_data(username, data_opt_in)
    user_watchlist = get_watchlist_data(username)

    current_job = get_current_job(conn)
    current_job.meta["user_status"] = user_data[1]
    current_job.meta["num_user_ratings"] = len(user_data[0])
    current_job.meta["user_watchlist"] = user_watchlist[0]
    current_job.save()

    return user_data[0]


def build_client_model(
    username, training_data_rows=1000000, num_items=30
):
    # Load user data from previous Redis job
    current_job = get_current_job(conn)
    user_data_job = current_job.dependency
    user_data = user_data_job.result
    # user_watched_list = [x["movie_id"] for x in user_data]

    current_job.meta["stage"] = "creating_sample_data"
    current_job.save()
    # Load in training full training dataset and filter it to the selected sample size
    # df = pd.read_csv("data_processing/data/training_data.csv")
    # model_df = df.head(training_data_rows)
    # model_df = pd.read_parquet(f"data_processing/data/training_data_samples/training_data_{training_data_rows}.parquet")

    # Load in the list of all availble movie ids (passed the threshold of at least five samples in dataset)
    with open(f"data_processing/data/movie_lists/sample_movie_list_{training_data_rows}.txt", "rb") as fp:
        sample_movie_list = pickle.load(fp)
    
    with open("data_processing/models/best_svd_params.json", 'r') as f:
        svd_params = json.load(f)

    current_job.meta["stage"] = "building_model"
    current_job.save()
    # Build model with appended user data
    # algo, user_watched_list = build_model(model_df, sample_movie_list, user_data, params=svd_params)
    # del model_df

    # algo = load(f"data_processing/models/model_{training_data_rows}.pkl")[1]
    algo = load_compressed_model(f"data_processing/models/model_{training_data_rows}.npz")
    movie_data = get_movie_data(sample_movie_list, sample_size=training_data_rows)

    current_job.meta["stage"] = "running_model"
    current_job.save()
    # Get recommendations from the model, excluding movies a user has watched and return top recommendations (of length num_items)
    recs = run_model(username, algo, user_data, sample_movie_list, movie_data, num_items, fold_in=True)
    return recs
