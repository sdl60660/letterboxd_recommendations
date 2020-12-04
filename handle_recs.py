import pandas as pd
import pickle

from rq import Queue, get_current_job
from rq.job import Job
from rq.registry import FinishedJobRegistry

from data_processing.get_user_ratings import get_user_data
from data_processing.build_model import build_model
from data_processing.run_model import run_model

from worker import conn


def get_previous_job_from_registry(index=-1):
    q = Queue('default', connection=conn)
    registry = FinishedJobRegistry(queue=q)
    
    job_id = registry.get_job_ids()[index]
    job = q.fetch_job(job_id)

    return job


def create_training_data(training_data_rows=200000, popularity_filter=False):
    df = pd.read_csv('data_processing/data/training_data.csv')
    with open("data_processing/models/threshold_movie_list.txt", "rb") as fp:
        threshold_movie_list = pickle.load(fp)

    model_df = df.head(training_data_rows)
    # print(model_df.head())

    if popularity_filter:
        review_count_threshold = 2000

        review_counts = pd.read_csv('data_processing/data/review_counts.csv')
        review_counts = review_counts.loc[review_counts['rating_val'] < review_count_threshold]
        
        included_movies = review_counts['movie_id'].to_list()
        threshold_movie_list = [x for x in threshold_movie_list if x in included_movies]
    
    return model_df, threshold_movie_list


def build_client_model(username):
    current_job = get_current_job(conn)
    training_data_job = current_job.dependency

    model_df = training_data_job.result[0]
    threshold_movie_list = training_data_job.result[1]
    
    algo, user_watched_list = build_model(model_df, username)
    return algo, user_watched_list, threshold_movie_list


def run_client_model(username, num_items=30):
    current_job = get_current_job(conn)
    build_model_job = current_job.dependency

    algo = build_model_job.result[0]
    user_watched_list = build_model_job.result[1]
    threshold_movie_list = build_model_job.result[2]

    recs = run_model(username, algo, user_watched_list, threshold_movie_list, num_items)

    return recs


def get_recommendations(username, training_data_rows=200000, popularity_filter=False, num_items=30):    
    df = pd.read_csv('data_processing/data/training_data.csv')
    with open("data_processing/models/threshold_movie_list.txt", "rb") as fp:
        threshold_movie_list = pickle.load(fp)

    model_df = df.head(training_data_rows)
    print(model_df.head())    

    if popularity_filter:
        review_count_threshold = 2000

        review_counts = pd.read_csv('data_processing/data/review_counts.csv')
        review_counts = review_counts.loc[review_counts['rating_val'] < review_count_threshold]
        
        included_movies = review_counts['movie_id'].to_list()
        threshold_movie_list = [x for x in threshold_movie_list if x in included_movies]

    algo, user_watched_list = build_model(model_df, username)
    print("model built")
    recs = run_model(username, algo, user_watched_list, threshold_movie_list, num_items)
    print("recs received")

    return recs
