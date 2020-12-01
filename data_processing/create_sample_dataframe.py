#!/usr/local/bin/python3.9

import pandas as pd

from numpy import asarray
from numpy import savetxt

import pickle

import pymongo
from config import config

# Connect to MongoDB Client
db_name = config["MONGO_DB"]
client = pymongo.MongoClient(f'mongodb+srv://{config["MONGO_USERNAME"]}:{config["MONGO_PASSWORD"]}@cluster0.{config["MONGO_CLUSTER_ID"]}.mongodb.net/{db_name}?retryWrites=true&w=majority')

db = client[db_name]
ratings = db.ratings
users = db.users

# all_ratings = ratings.find({}, { "user_id": 1, "movie_id": 1, "rating_val": 1, "_id": 0 })
# all_ratings = ratings.find({}, { "user_id": 1, "movie_id": 1, "rating_val": 1, "_id": 0 }, limit=250000)
all_ratings = []

target_sample_size = 200000
sample_size = int(target_sample_size*1.15)
max_chunk_size = 30000
num_iterations = 1 + (sample_size // max_chunk_size)

for iteration in range(num_iterations):
    iteration_size = min(max_chunk_size, sample_size - (max_chunk_size*iteration))

    rating_sample = ratings.aggregate([
        {"$sample": {"size": iteration_size}}
    ])

    all_ratings += list(rating_sample)

df = pd.DataFrame(all_ratings)
df = df[["user_id", "movie_id", "rating_val"]]
print(df.shape)
df.drop_duplicates(inplace=True)
print(df.shape)

df.to_csv('data/sample_rating_data.csv', index=False)
# df = pd.read_csv('data/sample_rating_data.csv')

min_review_threshold = 5

grouped_df = df.groupby(by=["movie_id"]).sum().reset_index()
grouped_df = grouped_df.loc[grouped_df['rating_val'] > min_review_threshold]

with open('models/threshold_movie_list.txt', 'wb') as fp:
    pickle.dump(grouped_df["movie_id"].to_list(), fp)
