from data_processing.build_model import build_model
from data_processing.run_model import run_model

import pandas as pd
import pickle

def get_recommendations(username, num_items=30, training_data_rows=200000):    
    df = pd.read_csv('data_processing/data/training_data.csv')
    with open("data_processing/models/threshold_movie_list.txt", "rb") as fp:
        threshold_movie_list = pickle.load(fp)
    
    model_df = df.head(training_data_rows)
    print(model_df.head())

    algo, user_watched_list = build_model(model_df, username)
    print("model built")
    recs = run_model(username, algo, user_watched_list, threshold_movie_list, num_items)
    print("recs received")

    return recs