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

from pymongo import UpdateOne
from pymongo.errors import BulkWriteError

from db_connect import connect_to_db

import os

if os.getcwd().endswith("/data_processing"):
    from http_utils import BROWSER_HEADERS
    from utils.utils import get_backoff_days

else:
    from data_processing.http_utils import BROWSER_HEADERS
    from data_processing.utils.utils import get_backoff_days


def format_img_link_stub(raw_link):
    image_url = raw_link.replace("https://a.ltrbxd.com/resized/", "").split(
                ".jpg"
            )[0]
    
    if "https://s.ltrbxd.com/static/img/empty-poster" in raw_link:
        image_url = ""
    
    return image_url

def get_meta_data_from_script_tag(soup):
    # find the <script type="application/ld+json"> tag
    data_script = soup.find("script", attrs={"type": "application/ld+json"})
    if data_script and data_script.string:
        # clean out the /* <![CDATA[ */ and /* ]]> */ wrappers if present
        raw_json = data_script.string.strip()
        raw_json = raw_json.replace("/* <![CDATA[ */", "").replace("/* ]]> */", "").strip()

        data = json.loads(raw_json)
        image_url = data.get("image", "")
        genres = data.get("genre")

        rating_data = data.get("aggregateRating", {})
        rating_count = rating_data.get("ratingCount")
        avg_rating = rating_data.get("ratingValue")
        
        return {"image_url": image_url, "letterboxd_rating_count": rating_count, "letterboxd_avg_rating": avg_rating, "letterboxd_genres": genres }


def parse_letterboxd_page_data(response, movie_id):
    # Parse ratings page response for each rating/review, use lxml parser for speed
    soup = BeautifulSoup(response, "lxml")

    movie_header = soup.find("section", class_="production-masthead")

    try:
        movie_title = movie_header.find("h1").text
    except AttributeError:
        movie_title = ""

    try:
        year = int(
            movie_header.find("span", class_="releasedate").find("a").text
        )
    except AttributeError:
        year = None
    
    try:
        imdb_link = soup.find("a", attrs={"data-track-action": "IMDb"})["href"]
        imdb_id = imdb_link.split("/title")[1].strip("/").split("/")[0]
    except:
        imdb_link = ""
        imdb_id = ""

    try:
        tmdb_link = soup.find("a", attrs={"data-track-action": "TMDB"})["href"]
        content_type = "movie" if "/movie/" in tmdb_link else "tv"
        tmdb_id = tmdb_link.split(f"/{content_type}")[1].strip("/").split("/")[0]
    except:
        tmdb_link = ""
        tmdb_id = ""
        content_type = None
    
    movie_update_object = {
        "movie_id": movie_id,
        "movie_title": movie_title,
        "year_released": year,
        "imdb_link": imdb_link,
        "tmdb_link": tmdb_link,
        "imdb_id": imdb_id,
        "tmdb_id": tmdb_id,
        "content_type": content_type,
        "scrape_status": "ok",
        "fail_count": 0,
        "next_retry_at": None
    }

    try:
        script_tag_data = get_meta_data_from_script_tag(soup)

        for k, v in script_tag_data.items():
            if v is None:
                continue
            elif k == 'image_url':
                movie_update_object[k] = format_img_link_stub(v)
            else:
                movie_update_object[k] = v
                    
    except:
        # it is particularly important to have a value (or empty value) for the poster so that the frontend knows what to do
        # our update crawl will treat items differently if they have a null/empty-string value for the poster vs. no value at all
        # so even if the script data isn't present for some reason, let's ensure that we mark this as an empty string
        movie_update_object['image_url'] = ""

    return movie_update_object


def format_failed_update(movie_id, fail_count):
    # backoff_days = get_backoff_days(fail_count)
    backoff_days = 7
    now = datetime.datetime.now(datetime.timezone.utc)
    next_retry = now + datetime.timedelta(days=backoff_days)

    movie_update_object = {
        "movie_id": movie_id,
        "scrape_status": "failed",
        "next_retry_at": next_retry,
    }

    return movie_update_object

async def fetch_letterboxd(url, session, input_data={}):
    async with session.get(url) as r:
        response = await r.read()

        movie_id = input_data['movie_id']
        if r.status == 404:
            fail_count = input_data.get("fail_count", 0) + 1
            movie_update_object = format_failed_update(movie_id, fail_count)
            update_operation = UpdateOne(
                { "movie_id": input_data["movie_id"]}, {"$set": movie_update_object,  "$inc": {"fail_count": 1}}, upsert=True
            )
        else:
            movie_update_object = parse_letterboxd_page_data(response, movie_id)
            update_operation = UpdateOne(
                {"movie_id": input_data["movie_id"]}, {"$set": movie_update_object}, upsert=True
            )

        return update_operation


async def fetch_poster(url, session, input_data={}):
    async with session.get(url) as r:
        response = await r.read()

        # Parse poster standalone page
        soup = BeautifulSoup(response, "lxml")

        try:
            image_url = (
                soup.find("div", class_="film-poster")
                .find("img")["src"]
                .split("?")[0]
            )
            image_url = image_url.replace("https://a.ltrbxd.com/resized/", "").split(
                ".jpg"
            )[0]
            if "https://s.ltrbxd.com/static/img/empty-poster" in image_url:
                image_url = ""
        except AttributeError:
            image_url = ""

        movie_object = {
            "movie_id": input_data["movie_id"],
        }

        if image_url != "":
            movie_object["image_url"] = image_url

        movie_object["last_updated"] = datetime.datetime.now(datetime.timezone.utc)

        update_operation = UpdateOne(
            {"movie_id": input_data["movie_id"]}, {"$set": movie_object}, upsert=True
        )

        return update_operation


async def fetch_tmdb_data(url, session, movie_data, input_data={}):
    async with session.get(url) as r:
        response = await r.json()

        movie_object = movie_data

        object_fields = ["genres", "production_countries", "spoken_languages"]
        for field_name in object_fields:
            try:
                movie_object[field_name] = [x["name"] for x in response[field_name]]
            except:
                movie_object[field_name] = None

        simple_fields = [
            "popularity",
            "overview",
            "runtime",
            "vote_average",
            "vote_count",
            "release_date",
            "original_language",
        ]
        for field_name in simple_fields:
            try:
                movie_object[field_name] = response[field_name]
            except:
                movie_object[field_name] = None

        movie_object["last_updated"] = datetime.datetime.now(datetime.timezone.utc)

        update_operation = UpdateOne(
            {"movie_id": input_data["movie_id"]}, {"$set": movie_object}, upsert=True
        )

        return update_operation


async def get_movies(movie_list, db_cursor, mongo_db):
    url = "https://letterboxd.com/film/{}/"

    async with ClientSession() as session:
        tasks = []
        # Make a request for each ratings page and add to task queue
        for movie in movie_list:
            task = asyncio.ensure_future(
                fetch_letterboxd(url.format(movie), session, {"movie_id": movie})
            )
            tasks.append(task)

        # Gather all ratings page responses
        upsert_operations = await asyncio.gather(*tasks)

    try:
        if len(upsert_operations) > 0:
            # Create/reference "ratings" collection in db
            movies = mongo_db.movies
            movies.bulk_write(upsert_operations, ordered=False)
    except BulkWriteError as bwe:
        pprint(bwe.details)


async def get_rich_data(movie_list, db_cursor, mongo_db, tmdb_key):
    base_url = "https://api.themoviedb.org/3/{}/{}?api_key={}"

    async with ClientSession(headers=BROWSER_HEADERS, connector=TCPConnector(limit=6)) as session:
        tasks = []
        movie_list = [x for x in movie_list if x["tmdb_id"]]
        # Make a request for each ratings page and add to task queue
        for movie in movie_list:
            content_type = movie["content_type"] or "movie"
            task = asyncio.ensure_future(
                fetch_tmdb_data(
                    base_url.format(content_type, movie["tmdb_id"], tmdb_key),
                    session,
                    movie,
                    {"movie_id": movie["movie_id"]},
                )
            )
            tasks.append(task)

        # Gather all ratings page responses
        upsert_operations = await asyncio.gather(*tasks)

    try:
        if len(upsert_operations) > 0:
            # Create/reference "ratings" collection in db
            movies = mongo_db.movies
            movies.bulk_write(upsert_operations, ordered=False)
    except BulkWriteError as bwe:
        pprint(bwe.details)


def get_ids_for_update(movies_collection, data_type):
    now = datetime.datetime.now(datetime.timezone.utc)
    one_month_ago = now - datetime.timedelta(days=30)

    # Find all movies with missing metadata, which implies that they were added during get_ratings and have not been scraped yet
    # All other movies have already had their data scraped and since this is almost always unchanging data, we won't rescrape 200,000+ records
    if data_type == "letterboxd":
        update_ids = set()

        # 1000 least recently updated items, excluding anything updated in the last month
        update_ids |= {
            x["movie_id"]
            for x in movies_collection.find(
                {"last_updated": {"$lte": one_month_ago}},
                {"movie_id": 1}
            )
            .sort("last_updated", 1)
            .limit(1000)
        }

        # grab a sample of those which had a failed crawl and are now due for a retry
        update_ids |= {
            x["movie_id"]
            for x in movies_collection.find(
                # {"scrape_status": "failed"},
                {"next_retry_at": {"$lte": now}, "scrape_status": "failed"},
                {"movie_id": 1}
            ).sort("next_retry_at", 1)
        }

        # backfill a chunk of the records that are missing 'content_type' (newly-added)
        update_ids |= {
            x["movie_id"]
            for x in movies_collection.find(
                {        
                    "content_type": {"$exists": False}
                },
                {"movie_id": 1},
            )
            .limit(1000)
        }

        # anything newly added or missing key data (including missing poster image)
        update_ids |= {
            x["movie_id"]
            for x in movies_collection.find(
                {        
                    "$or": [
                        {"movie_title": {"$exists": False}},
                        {"tmdb_id": {"$exists": False}},
                        {"image_url": {"$exists": False}},
                    ]
                },
                {"movie_id": 1},
            )
        }

        # missing key data (but has been attempted before), limited to a batch of 500 per update
        update_ids |= {
            x["movie_id"]
            for x in movies_collection.find(
                {
                    "$and": [
                        {
                            "$or": [
                                {"movie_title": {"$in": ["", None]}},
                                {"tmdb_id": {"$in": ["", None]}},
                                {"image_url": {"$in": ["", None]}},
                                {"content_type": {"in": ["", None]}},
                            ]
                        },
                        {
                            "$or": [
                                {"last_updated": {"$exists": False}},
                                {"last_updated": {"$lte": one_month_ago}},
                            ]
                        },
                    ]
                },
                {"movie_id": 1},
            )
            .sort("last_updated", 1)
            .limit(500) 
        }

        all_movies = list(update_ids)

    else:
        all_movies = [
            x
            for x in list(
                movies_collection.find(
                    {
                        "genres": {"$exists": False},
                        "content_type": {"$exists": True}
                    }
                )
            )
        ]

    return all_movies

def main(data_type="letterboxd"):
    # Connect to MongoDB client
    db_name, client = connect_to_db()
    tmdb_key = os.environ["TMDB_KEY"]

    db = client[db_name]
    movies = db.movies

    movies_for_update = get_ids_for_update(movies, data_type)

    loop = asyncio.get_event_loop()
    chunk_size = 20
    num_chunks = len(movies_for_update) // chunk_size + 1

    print("Total Movies to Scrape:", len(movies_for_update))
    print("Total Chunks:", num_chunks)
    print("=======================\n")

    pbar = tqdm(range(num_chunks))
    for chunk_i in pbar:
        pbar.set_description(f"Scraping chunk {chunk_i + 1} of {num_chunks}")

        if chunk_i == num_chunks - 1:
            chunk = movies_for_update[chunk_i * chunk_size :]
        else:
            chunk = movies_for_update[chunk_i * chunk_size : (chunk_i + 1) * chunk_size]

        for attempt in range(5):
            try:
                if data_type == "letterboxd":
                    future = asyncio.ensure_future(get_movies(chunk, movies, db))
                else:
                    future = asyncio.ensure_future(
                        get_rich_data(chunk, movies, db, tmdb_key)
                    )
                loop.run_until_complete(future)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Error on attempt {attempt+1}, retrying...")
            else:
                break
        else:
            print(f"Count not complete requests for chunk {chunk_i+1}")

    return


if __name__ == "__main__":
    main("letterboxd")
    main("tmdb")
