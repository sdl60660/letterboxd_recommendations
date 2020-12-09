#!/usr/local/bin/python3.9

import pandas as pd

from numpy import asarray
from numpy import savetxt

import pickle

import pymongo
from db_config import config


def get_sample(cursor, iteration_size):
    while True:
        try:
            rating_sample = cursor.aggregate([
                {"$sample": {"size": iteration_size}}
            ])
            return list(rating_sample)
        except pymongo.errors.OperationFailure:
            print("Encountered $sample operation error. Retrying...")
    
    
def create_training_data(db_client, sample_size=200000):
    ratings = db_client.ratings

    all_ratings = []
    unique_records = 0
    while unique_records < sample_size:
        rating_sample = get_sample(ratings, 100000)
        all_ratings += rating_sample
        unique_records = len(set([(x['movie_id'] + x['user_id']) for x in all_ratings]))
        print(unique_records)

    df = pd.DataFrame(all_ratings)
    df = df[["user_id", "movie_id", "rating_val"]]
    df.drop_duplicates(inplace=True)
    df = df.head(sample_size)
    print(df.head())

    min_review_threshold = 5

    grouped_df = df.groupby(by=["movie_id"]).count().reset_index()
    print(grouped_df.head())
    grouped_df = grouped_df.loc[grouped_df['rating_val'] > min_review_threshold]
    full_movie_list = grouped_df["movie_id"].to_list()

    return df, full_movie_list


def create_movie_data_sample(db_client, movie_list):
    movies = db_client.movies
    included_movies = movies.find( { "movie_id": { "$in": movie_list } } )
    
    movie_df = pd.DataFrame(list(included_movies))
    movie_df = movie_df[['movie_id', 'image_url', 'movie_title', 'year_released']]
    movie_df['image_url'] = movie_df['image_url'].str.replace('https://a.ltrbxd.com/resized/', '', regex=False)
    movie_df['image_url'] = movie_df['image_url'].str.replace('https://s.ltrbxd.com/static/img/empty-poster-230.c6baa486.png', '', regex=False)
    
    return movie_df

if __name__ == "__main__":
    # Connect to MongoDB Client
    db_name = config["MONGO_DB"]

    if "CONNECTION_URL" in config.keys():
        client = pymongo.MongoClient(config["CONNECTION_URL"])
    else:
        client = pymongo.MongoClient(f'mongodb+srv://{config["MONGO_USERNAME"]}:{config["MONGO_PASSWORD"]}@cluster0.{config["MONGO_CLUSTER_ID"]}.mongodb.net/{db_name}?retryWrites=true&w=majority')

    db = client[db_name]

    # Generate training data sample
    training_df, threshold_movie_list = create_training_data(db, 1000000)

    # Create review counts dataframe
    review_counts_df = pd.DataFrame(list(db.ratings.find({}))).groupby(by=["movie_id"]).count().reset_index()
    # We'll pull review counts from the full DB dataset, but then only include those in the threshold list in the final dataframe
    # This is because only those on the threshold list will make it into the model anyway, so filtering the dataframe now avoids processing later
    # But we start with the full dataset so as to get more accurate review counts, rather than using review counts from a smaller sample
    review_counts_df = review_counts_df[review_counts_df['movie_id'].isin(threshold_movie_list)]
    review_counts_df["count"] = review_counts_df["_id"]
    review_counts_df = review_counts_df[["movie_id", "count"]]
    
    # Generate movie data CSV
    movie_df = create_movie_data_sample(db, threshold_movie_list)
    print(movie_df.head())
    print(movie_df.shape)
    

    # Store Data
    with open('models/threshold_movie_list.txt', 'wb') as fp:
        pickle.dump(threshold_movie_list, fp)
    
    training_df.to_csv('data/training_data.csv', index=False)
    review_counts_df.to_csv('data/review_counts.csv', index=False)
    movie_df.to_csv('../static/data/movie_data.csv', index=False)