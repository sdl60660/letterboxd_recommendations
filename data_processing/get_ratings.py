#!/usr/local/bin/python3.12

import os
import time
from tqdm import tqdm
import math
import datetime
from itertools import chain

import asyncio
from aiohttp import ClientSession, TCPConnector, ClientTimeout
from bs4 import BeautifulSoup

from pprint import pprint

from pymongo import UpdateOne
from pymongo.errors import BulkWriteError

if os.getcwd().endswith("data_processing"):
    from db_connect import connect_to_db
    from utils import utils
    from http_utils import BROWSER_HEADERS

else:
    from data_processing.db_connect import connect_to_db
    from data_processing.utils import utils
    from data_processing.http_utils import BROWSER_HEADERS

async def fetch(url, session, input_data={}, *, retries=3):
    for attempt in range(retries):
        try:
            async with session.get(
                url,
                timeout=ClientTimeout(total=20)
            ) as resp:
                if resp.status == 200:
                    return await resp.read(), input_data
                # backoff on transient blocks
                if resp.status in (429, 503, 520, 521, 522):
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                return None, None
        except Exception:
            await asyncio.sleep(1.0 * (attempt + 1))
    return None, None


async def get_page_counts(usernames, users_cursor):
    url = "https://letterboxd.com/{}/films/"
    tasks = []

    async with ClientSession(headers=BROWSER_HEADERS, connector=TCPConnector(limit=6)) as session:
        for username in usernames:
            task = asyncio.ensure_future(
                fetch(url.format(username), session, {"username": username})
            )
            tasks.append(task)

        responses = await asyncio.gather(*tasks)
        responses = [x for x in responses if x and x[0]]

        update_operations = []
        for i, response in enumerate(responses):
            soup = BeautifulSoup(response[0], "lxml")
            links = soup.select("li.paginate-page a")
            num_pages = int(links[-1].get_text(strip=True).replace(",", "")) if links else 1

            user = users_cursor.find_one({"username": response[1]["username"]})

            try:
                previous_num_pages = user["num_ratings_pages"]
                if not previous_num_pages:
                    previous_num_pages = 0
            except KeyError:
                previous_num_pages = 0

            # To avoid re-scraping a bunch of reviews already hit, we'll only scrape new pages
            # To be safe, and because pagination is funky, we'll do one more than the difference
            # ...between previously scraped page count and current page count, but then cap it at total
            # ...pages in case there were zero previously scraped pages or only a single page, etc
            new_pages = min(num_pages, num_pages - previous_num_pages + 1)

            # Also, pages cap at 128, so if they've got 128 pages and we've scraped most of them before, we'll
            # ...just do the most recent 10 pages
            if num_pages >= 128 and new_pages < 10:
                new_pages = 10

            update_operations.append(
                UpdateOne(
                    {"username": response[1]["username"]},
                    {
                        "$set": {
                            "num_ratings_pages": num_pages,
                            "recent_page_count": new_pages,
                            "last_updated": datetime.datetime.now(),
                        }
                    },
                    upsert=True,
                )
            )

        try:
            if len(update_operations) > 0:
                users_cursor.bulk_write(update_operations, ordered=False)
        except BulkWriteError as bwe:
            pprint(bwe.details)


async def generate_ratings_operations(response, send_to_db=True, return_unrated=False):
    if not response or not response[0]:
        return [], []

    # Parse ratings page response for each rating/review, use lxml parser for speed
    try:
        soup = BeautifulSoup(response[0], "lxml")
    except Exception:
        return [], []

    reviews = soup.find_all("li", class_="griditem")

    # Create empty array to store list of bulk operations or rating objects
    ratings_operations = []
    movie_operations = []

    # For each review, parse data from scraped page and append an UpdateOne operation for bulk execution or a rating object
    for review in reviews:
        rc = review.select_one("div.react-component")
        movie_id = rc.get("data-item-slug") if rc else None

        rating_el = review.select_one("span.rating")
        if not rating_el:
            if return_unrated == False:
                continue
            else:
                rating_val = -1
        
        else:
            classes = rating_el.get("class", []) if rating_el else []
            rated = next((c for c in classes if c.startswith("rated-")), None)  # e.g. "rated-8"
            rating_val = int(rated.split("-")[-1]) if rated else -1

        rating_object = {
            "movie_id": movie_id,
            "rating_val": rating_val,
            "user_id": response[1]["username"],
        }

        # We're going to eventually send a bunch of upsert operations for movies with just IDs
        # For movies already in the database, this won't impact anything
        # But this will allow us to easily figure out which movies we need to scraped data on later,
        # Rather than scraping data for hundreds of thousands of movies everytime there's a broader data update
        skeleton_movie_object = {"movie_id": movie_id}

        # If returning objects, just append the object to return list
        if not send_to_db:
            ratings_operations.append(rating_object)

        # Otherwise return an UpdateOne operation to bulk execute
        else:
            ratings_operations.append(
                UpdateOne(
                    {"user_id": response[1]["username"], "movie_id": movie_id},
                    {"$set": rating_object},
                    upsert=True,
                )
            )

            movie_operations.append(
                UpdateOne(
                    {"movie_id": movie_id}, {"$set": skeleton_movie_object}, upsert=True
                )
            )

    return ratings_operations, movie_operations


async def get_user_ratings(
    username,
    db_cursor=None,
    mongo_db=None,
    store_in_db=True,
    num_pages=None,
    return_unrated=False,
):
    url = "https://letterboxd.com/{}/films/by/date/page/{}/"

    if not num_pages:
        # Find them in the MongoDB database and grab the number of ratings pages
        user = db_cursor.find_one({"username": username})

        # We're trying to limit the number of pages we crawl instead of wasting tons of time on
        # gathering ratings we've already hit (see comment in get_page_counts)
        num_pages = user["recent_page_count"]

    # Fetch all responses within one Client session,
    # keep connection alive for all requests.
    async with ClientSession() as session:
        tasks = []
        # Make a request for each ratings page and add to task queue
        for i in range(num_pages):
            task = asyncio.ensure_future(
                fetch(url.format(username, i + 1), session, {"username": username})
            )
            tasks.append(task)

        # Gather all ratings page responses
        scrape_responses = await asyncio.gather(*tasks)
        scrape_responses = [x for x in scrape_responses if x]

    # Process each ratings page response, converting it into bulk upsert operations or output dicts
    tasks = []
    for response in scrape_responses:
        task = asyncio.ensure_future(
            generate_ratings_operations(
                response, send_to_db=store_in_db, return_unrated=return_unrated
            )
        )
        tasks.append(task)

    parse_responses = await asyncio.gather(*tasks)

    if store_in_db == False:
        parse_responses = list(
            chain.from_iterable(list(chain.from_iterable(parse_responses)))
        )
        return parse_responses

    # Concatenate each response's upsert operations/output dicts
    upsert_ratings_operations = []
    upsert_movies_operations = []
    for response in parse_responses:
        upsert_ratings_operations += response[0]
        upsert_movies_operations += response[1]

    return upsert_ratings_operations, upsert_movies_operations


async def get_ratings(usernames, db_cursor=None, mongo_db=None, store_in_db=True):
    ratings_collection = mongo_db.ratings
    movies_collection = mongo_db.movies

    chunk_size = 10
    total_chunks = math.ceil(len(usernames) / chunk_size)

    for chunk_index in range(total_chunks):
        tasks = []
        db_ratings_operations = []
        db_movies_operations = []

        start_index = chunk_size * chunk_index
        end_index = chunk_size * chunk_index + chunk_size
        username_chunk = usernames[start_index:end_index]

        # pbar.set_description(f"Scraping ratings data for user group {chunk_index+1} of {total_chunks}")

        # For a given chunk, scrape each user's ratings and form an array of database upsert operations
        for i, username in enumerate(username_chunk):
            # print((chunk_size*chunk_index)+i, username)
            task = asyncio.ensure_future(
                get_user_ratings(
                    username,
                    db_cursor=db_cursor,
                    mongo_db=mongo_db,
                    store_in_db=store_in_db,
                )
            )
            tasks.append(task)
        
        # Gather all ratings page responses, concatenate all db upsert operatons for use in a bulk write
        user_responses = await asyncio.gather(*tasks)
        for response in user_responses:
            db_ratings_operations += response[0]
            db_movies_operations += response[1]

        if store_in_db:
            # Execute bulk upsert operations
            try:
                if len(db_ratings_operations) > 0:
                    # Bulk write all upsert operations into ratings collection in db
                    ratings_collection.bulk_write(db_ratings_operations, ordered=False)

                if len(db_movies_operations) > 0:
                    movies_collection.bulk_write(db_movies_operations, ordered=False)

            except BulkWriteError as bwe:
                pprint(bwe.details)


def print_status(start, chunk_size, chunk_index, total_operations, total_records):
    total_time = round((time.time() - start), 2)
    completed_records = chunk_size * chunk_index
    time_per_user = round(total_time / completed_records, 2)
    remaining_estimate = round(time_per_user * (total_records - completed_records), 2)

    print("\n================")
    print(f"Users Complete: {completed_records}")
    print(f"Users Remaining: {(total_records - completed_records)}")
    print("Chunk Database Operations:", total_operations)
    print()
    print("Current Time/User:", f"{time_per_user} seconds")
    print("Elapsed Time:", utils.format_seconds(total_time))
    print("Est. Time Remaining:", utils.format_seconds(remaining_estimate))
    print("================\n")


def main():
    # Connect to MongoDB client
    db_name, client, tmdb_key = connect_to_db()

    # Find letterboxd database and user collection
    db = client[db_name]
    users = db.users

    # Starting to attach last_updated times, so we can cycle though updates instead of updating every user's...
    # ...ratings every time. We'll just grab the 2000 records which are least recently updated + those without a last_updated value
    # all_users = list(users.find({}).sort("last_updated", 1).limit(2000))
    all_users = list(
        users.find(
                {"$or": [
                    {"last_updated": {"$exists": False}},
                    {"last_updated": None},
                ]}
            )
    ) + list(
        users.find({"last_updated": {"$exists": True}})
            .sort("last_updated", 1)
            .limit(2000)
    )
    
    all_usernames = [x["username"] for x in all_users]

    large_chunk_size = 100
    num_chunks = math.ceil(len(all_usernames) / large_chunk_size)

    pbar = tqdm(range(num_chunks))
    for chunk in pbar:
        pbar.set_description(
            f"Scraping ratings data for user group {chunk+1} of {num_chunks}"
        )
        username_set = all_usernames[
            chunk * large_chunk_size : (chunk + 1) * large_chunk_size
        ]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Find number of ratings pages for each user
            future = asyncio.ensure_future(get_page_counts(username_set, users))
            loop.run_until_complete(future)

            # Find and store ratings for each user
            future = asyncio.ensure_future(get_ratings(username_set, users, db))
            loop.run_until_complete(future)
        finally:
            loop.close()


if __name__ == "__main__":
    main()
