#!/usr/local/bin/python3.9

import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD
# from surprise.model_selection import GridSearchCV
from surprise.model_selection import cross_validate
from surprise.dump import dump

from config import config

import pickle


# Load ratings data
df = pd.read_csv('data/sample_rating_data.csv')

# Surprise dataset loading
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(df[["user_id", "movie_id", "rating_val"]], reader)

# Configure algorithm
algo = SVD()
# cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

trainingSet = data.build_full_trainset()
algo.fit(trainingSet)

dump("models/mini_model.pkl", predictions=None, algo=algo, verbose=1)
pickle.dump(data, open("models/mini_model_data.pkl", "wb"))
