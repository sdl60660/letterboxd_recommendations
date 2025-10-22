#!/usr/local/bin/python3.12

import datetime
import json
from bs4 import BeautifulSoup

import asyncio
from aiohttp import ClientSession, TCPConnector
import requests
from pprint import pprint

import pymongo
import pandas as pd

import time
from tqdm import tqdm

from pymongo import UpdateOne, DESCENDING, ASCENDING
from pymongo.errors import BulkWriteError

from db_connect import connect_to_db

import os

def get_update_list(user_collection, db):
  non_matching_test = list(user_collection.aggregate([
    {
      "$addFields": {
          "usernameStr": { "$toString": "$username" }
        }
    },
    {
        "$addFields": {
            "usernameLower": { "$toLower": "$usernameStr" },
            "hasUppercase": {
                "$regexMatch": {
                    "input": "$usernameStr",
                    "regex": "[A-Z]"
                }
            }
        }
    },
    {
        "$group": {
            "_id": "$usernameLower",
            "users": { "$push": "$$ROOT" },   # full documents
            "count": { "$sum": 1 },
            "anyUppercase": { "$max": "$hasUppercase" }
        }
    },
    {
      "$match": { "anyUppercase": True }
    }
  ]))
  
  # non_matching_test = list(user_collection.find(
  #       {"$expr": {"$ne": [{"$toLower": "$username"}, "$username"]}}))
  print('HERE', len(non_matching_test), [x['count'] for x in non_matching_test[:10]])


def get_colliding_ratings(ratings, db):
  pipeline = [
    {
        "$addFields": {
            "user_lower": { "$toLower": { "$toString": "$user_id" } }
        }
    },
    {
        "$group": {
            "_id": { "user_lower": "$user_lower", "movie_id": "$movie_id" },
            "docs": { "$push": "$$ROOT" },
            "count": { "$sum": 1 }
        }
    },
    { "$match": { "count": { "$gt": 1 } } },  # only groups that will collide
  ]
  collisions = list(ratings.aggregate(pipeline))
  print(len(collisions))

def create_temp_ratings_coll(db):
  pipeline = [
    {
        "$addFields": {
            "user_lower": { "$toLower": { "$toString": "$user_id" } }
        }
    },
    # Choose canonical doc per (user_lower, movie_id)
    {
        "$sort": { "updated_at": DESCENDING, "_id": DESCENDING }  # tie-breaker
    },
    {
        "$group": {
            "_id": { "user_lower": "$user_lower", "movie_id": "$movie_id" },
            "doc": { "$first": "$$ROOT" }  # keep the most recent
        }
    },
    # Rewrite the kept doc so that user_id is lowercased
    {
        "$replaceRoot": {
            "newRoot": {
                "$mergeObjects": [
                    "$doc",
                    { "user_id": "$_id.user_lower" }
                ]
            }
        }
    },
    # (Optional) clean helper field if it exists
    { "$unset": ["user_lower"] },

    # Write into a temp collection
    {
        "$merge": {
            "into": "ratings_normalized_tmp",
            "whenMatched": "replace",
            "whenNotMatched": "insert"
        }
    }
  ]

  db.ratings.aggregate(pipeline, allowDiskUse=True)


def main():
  # Connect to MongoDB client
  db_name, client, tmdb_key = connect_to_db()

  db = client[db_name]
  users = db.users
  ratings = db.ratings

  # colliding_ratings = get_colliding_ratings(ratings, db)
  create_temp_ratings_coll(db)

  # all_usernames = get_update_list(users, db)

  # large_chunk_size = 100
  # num_chunks = math.ceil(len(all_usernames) / large_chunk_size)

  # pbar = tqdm(range(num_chunks))
  # for chunk in pbar:
  #     pbar.set_description(
  #         f"Scraping ratings data for user group {chunk+1} of {num_chunks}"
  #     )
  #     username_set = all_usernames[
  #         chunk * large_chunk_size : (chunk + 1) * large_chunk_size
  #     ]

  #     loop = asyncio.new_event_loop()
  #     asyncio.set_event_loop(loop)
  #     try:
  #         pages_by_user = loop.run_until_complete(get_page_counts(username_set, users))
  #         loop.run_until_complete(get_ratings(username_set, pages_by_user, mongo_db=db, store_in_db=True))
  #     finally:
  #         loop.close()


if __name__ == "__main__":
  main()