import pandas as pd
import pickle

from rq import Queue, get_current_job
from rq.job import Job
from rq.registry import FinishedJobRegistry

from data_processing.get_user_ratings import get_user_data
from data_processing.get_user_watchlist import get_watchlist_data
from data_processing.build_model import build_model
from data_processing.run_model import run_model

from worker import conn


def get_previous_job_from_registry(index=-1):
    q = Queue('high', connection=conn)
    registry = FinishedJobRegistry(queue=q)
    
    job_id = registry.get_job_ids()[index]
    job = q.fetch_job(job_id)

    return job


def filter_threshold_list(threshold_movie_list, review_count_threshold=2000):
    review_counts = pd.read_csv('data_processing/data/review_counts.csv')
    review_counts = review_counts.loc[review_counts['count'] < review_count_threshold]
    
    included_movies = review_counts['movie_id'].to_list()
    threshold_movie_list = [x for x in threshold_movie_list if x in included_movies]

    return threshold_movie_list


def get_client_user_data(username, data_opt_in):
    user_data = get_user_data(username, data_opt_in)
    user_watchlist = get_watchlist_data(username)
    
    current_job = get_current_job(conn)
    current_job.meta['user_status'] = user_data[1]
    current_job.meta['num_user_ratings'] = len(user_data[0])
    current_job.meta['user_watchlist'] = user_watchlist
    current_job.save()

    return user_data[0]


def build_client_model(username, training_data_rows=200000, popularity_threshold=None, num_items=30):
    # Load user data from previous Redis job
    current_job = get_current_job(conn)
    user_data_job = current_job.dependency
    user_data = user_data_job.result

    current_job.meta['stage'] = 'creating_sample_data'
    current_job.save()
    # Load in training full training dataset and filter it to the selected sample size
    df = pd.read_csv('data_processing/data/training_data.csv')
    model_df = df.head(training_data_rows)

    # Load in the list of all availble movie ids (passed the threshold of at least five samples in dataset)
    with open("data_processing/models/threshold_movie_list.txt", "rb") as fp:
        threshold_movie_list = pickle.load(fp)

    # If user has requested only less often reviewed movies, apply review count threshold to the movie id list
    if popularity_threshold:
        threshold_movie_list = filter_threshold_list(threshold_movie_list, popularity_threshold)
    
    current_job.meta['stage'] = 'building_model'
    current_job.save()
    # Build model with appended user data
    algo, user_watched_list = build_model(model_df, user_data)
    del model_df

    current_job.meta['stage'] = 'running_model'
    current_job.save()
    # Get recommendations from the model, excluding movies a user has watched and return top recommendations (of length num_items)
    recs = run_model(username, algo, user_watched_list, threshold_movie_list, num_items)
    return recs    
