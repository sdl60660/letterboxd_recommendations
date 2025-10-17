#!/usr/local/bin/python3.12

import pandas as pd

import numpy as np
from numpy import asarray
from numpy import savetxt

import pickle

import pymongo

from db_connect import connect_to_db


def get_sample(cursor, iteration_size):
    while True:
        try:
            rating_sample = cursor.aggregate([{"$sample": {"size": iteration_size}}])
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
        unique_records = len(set([(x["movie_id"] + x["user_id"]) for x in all_ratings]))
        print(unique_records)

    df = pd.DataFrame(all_ratings)
    df = df[["user_id", "movie_id", "rating_val"]]
    df.drop_duplicates(inplace=True)
    df = df.head(sample_size)

    print(df.head())

    return df


def create_movie_data_sample(db_client, movie_list):
    movies = db_client.movies
    included_movies = movies.find({"movie_id": {"$in": movie_list}})

    movie_df = pd.DataFrame(list(included_movies))
    movie_df = movie_df[["movie_id", "image_url", "movie_title", "year_released"]]
    movie_df["image_url"] = (
        movie_df["image_url"]
        .fillna("")
        .str.replace("https://a.ltrbxd.com/resized/", "", regex=False)
    )
    movie_df["image_url"] = (
        movie_df["image_url"]
        .fillna("")
        .str.replace(
            "https://s.ltrbxd.com/static/img/empty-poster-230.c6baa486.png",
            "",
            regex=False,
        )
    )

    return movie_df


if __name__ == "__main__":
    # Connect to MongoDB client
    db_name, client, tmdb_key = connect_to_db()
    db = client[db_name]

    min_review_threshold = 20

    # Generate training data sample
    training_df = create_training_data(db, 1200000)

    # Create review counts dataframe
    review_count = db.ratings.aggregate(
        [
            {"$group": {"_id": "$movie_id", "review_count": {"$sum": 1}}},
            {"$match": {"review_count": {"$gte": min_review_threshold}}},
        ]
    )
    review_counts_df = pd.DataFrame(list(review_count))
    review_counts_df.rename(
        columns={"_id": "movie_id", "review_count": "count"}, inplace=True
    )

    threshold_movie_list = review_counts_df["movie_id"].to_list()

    # Generate movie data CSV
    movie_df = create_movie_data_sample(db, threshold_movie_list)
    print(movie_df.head())
    print(movie_df.shape)

    # Use movie_df to remove any items from threshold_list that do not have a "year_released"
    # This virtually always means it's a collection of more popular movies (such as the LOTR trilogy) and we don't want it included in recs
    retain_list = movie_df.loc[
        (movie_df["year_released"].notna() & movie_df["year_released"] != 0.0)
    ]["movie_id"].to_list()

    threshold_movie_list = [x for x in threshold_movie_list if x in retain_list]

    # Store Data
    with open("models/threshold_movie_list.txt", "wb") as fp:
        pickle.dump(threshold_movie_list, fp)

    training_df.to_csv("data/training_data.csv", index=False)
    review_counts_df.to_csv("data/review_counts.csv", index=False)
    movie_df.to_csv("../static/data/movie_data.csv", index=False)
