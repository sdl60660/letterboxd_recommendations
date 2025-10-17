#!/usr/local/bin/python3.12

from re import U
from bs4 import BeautifulSoup
from pymongo.operations import ReplaceOne
import requests

import asyncio
from aiohttp import ClientSession

import pymongo
from pymongo import UpdateOne, ReplaceOne
from pymongo.errors import BulkWriteError

import datetime

from pprint import pprint

import os

if os.getcwd().endswith("/data_processing"):
    from get_ratings import get_user_ratings
    from http_utils import BROWSER_HEADERS

else:
    from data_processing.get_ratings import get_user_ratings
    from data_processing.http_utils import BROWSER_HEADERS


def get_page_count(username):
    url = "https://letterboxd.com/{}/films/by/date"
    r = requests.get(url.format(username), headers=BROWSER_HEADERS)

    soup = BeautifulSoup(r.text, "lxml")
    body = soup.find("body")

    try:
        if "error" in body["class"]:
            return -1, None
    except KeyError:
        print(body)
        return -1, None

    try:
        links = soup.select("li.paginate-page a")
        if links:
            num_pages = int(links[-1].get_text(strip=True).replace(",", ""))
        else:
            num_pages = 1

        header = soup.select_one("section.profile-header h1.title-3")
        display_name = header.get_text(strip=True) if header else None
    except Exception:
        num_pages = 1
        display_name = None

    return num_pages, display_name


def get_user_data(username, data_opt_in=False):
    num_pages, display_name = get_page_count(username)
    if num_pages == -1:
        return [], "user_not_found"

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    future = asyncio.ensure_future(
        get_user_ratings(
            username,
            db_cursor=None,
            store_in_db=False,
            num_pages=num_pages,
            return_unrated=True,
        )
    )
    loop.run_until_complete(future)

    user_ratings = [x for x in future.result() if x["rating_val"] >= 0]
    if data_opt_in:
        send_to_db(username, display_name, user_ratings=user_ratings)

    return future.result(), "success"


def send_to_db(username, display_name, user_ratings):
    database_url = os.getenv("DATABASE_URL", None)

    if database_url:
        client = pymongo.MongoClient(
            database_url, server_api=pymongo.server_api.ServerApi("1")
        )

        db = client["letterboxd"]
        users = db.users
        ratings = db.ratings
        movies = db.movies

        user = {
            "username": username.lower(),
            "display_name": display_name,
            "num_reviews": len(user_ratings),
            "last_attempted": datetime.datetime.now(datetime.timezone.utc),
            "last_updated": datetime.datetime.now(datetime.timezone.utc),
        }

        users.update_one({"username": user["username"]}, {"$set": user}, upsert=True)

        upsert_ratings_operations = []
        upsert_movies_operations = []
        # print(len(user_ratings))
        for rating in user_ratings:
            upsert_ratings_operations.append(
                ReplaceOne(
                    {"user_id": username, "movie_id": rating["movie_id"]},
                    rating,
                    upsert=True,
                )
            )

            upsert_movies_operations.append(
                UpdateOne(
                    {"movie_id": rating["movie_id"]},
                    {"$set": {"movie_id": rating["movie_id"]}},
                    upsert=True,
                )
            )

        try:
            if len(upsert_ratings_operations) > 0:
                ratings.bulk_write(upsert_ratings_operations, ordered=False)
            if len(upsert_movies_operations) > 0:
                movies.bulk_write(upsert_movies_operations, ordered=False)
        except BulkWriteError as bwe:
            pprint(bwe.details)

        return


if __name__ == "__main__":
    username = "samlearner"
    get_user_data(username)
