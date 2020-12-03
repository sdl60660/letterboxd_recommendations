#!/usr/local/bin/python3.9

import asyncio
from aiohttp import ClientSession
import requests
from pprint import pprint

from bs4 import BeautifulSoup

import pymongo
from pymongo import UpdateOne
from pymongo.errors import BulkWriteError

import time


async def fetch(url, session, input_data={}):
    async with session.get(url) as response:
        return await response.read(), input_data
            

async def get_page_counts(usernames, users_cursor):
    url = "https://letterboxd.com/{}/films/"
    tasks = []

    async with ClientSession() as session:
        for username in usernames:
            task = asyncio.ensure_future(fetch(url.format(username), session, {"username": username}))
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)

        for response in responses:
            soup = BeautifulSoup(response[0], "lxml")
            try:
                page_link = soup.findAll("li", attrs={"class", "paginate-page"})[-1]
                num_pages = int(page_link.find("a").text.replace(',', ''))
            except IndexError:
                num_pages = 1

            users_cursor.update_one({"username": response[1]['username']}, {"$set": {"num_ratings_pages": num_pages}})


async def generate_ratings_operations(response, send_to_db=True, return_unrated=False):
    
    # Parse ratings page response for each rating/review, use lxml parser for speed
    soup = BeautifulSoup(response[0], "lxml")
    reviews = soup.findAll("li", attrs={"class": "poster-container"})

    # Create empty array to store list of bulk operations or rating objects
    operations = []

    # For each review, parse data from scraped page and append an UpdateOne operation for bulk execution or a rating object
    for review in reviews:
        movie_id = review.find('div', attrs={"class", "film-poster"})['data-target-link'].split('/')[-2]

        rating = review.find("span", attrs={"class": "rating"})
        if not rating:
            if return_unrated == False:
                continue
            else:
                rating_val = -1
        else:
            rating_class = rating['class'][-1]
            rating_val = int(rating_class.split('-')[-1])

        rating_object = {
                    "movie_id": movie_id,
                    "rating_val": rating_val,
                    "user_id": response[1]["username"]
                }

        # If returning objects, just append the object to return list
        if not send_to_db:
            operations.append(rating_object)
        # Otherwise return an UpdateOne operation to bulk execute
        else:
            operations.append(UpdateOne({
                    "user_id": response[1]["username"],
                    "movie_id": movie_id
                },
                {
                    "$set": rating_object
                }, upsert=True))
    
    return operations
    

async def get_user_ratings(username, db_cursor=None, mongo_db=None, store_in_db=True, num_pages=None, return_unrated=False):
    # url = "https://letterboxd.com/{}/films/ratings/page/{}/"
    url = "https://letterboxd.com/{}/films/by/date/page/{}/"
    
    if not num_pages:
        # Find them in the MongoDB database and grab the number of ratings pages
        user = db_cursor.find_one({"username": username})
        num_pages = user["num_ratings_pages"]

    # Fetch all responses within one Client session,
    # keep connection alive for all requests.
    async with ClientSession() as session:
        # print("Starting Scrape", time.time() - start)

        tasks = []
        # Make a request for each ratings page and add to task queue
        for i in range(num_pages):
            task = asyncio.ensure_future(fetch(url.format(username, i+1), session, {"username": username}))
            tasks.append(task)

        # Gather all ratings page responses
        scrape_responses = await asyncio.gather(*tasks)

        # print("Finishing Scrape", time.time() - start)
        
    # Process each ratings page response, converting it into bulk upsert operations or output dicts
    tasks = []
    for response in scrape_responses:
        task = asyncio.ensure_future(generate_ratings_operations(response, send_to_db=store_in_db, return_unrated=return_unrated))
        tasks.append(task)
    
    parse_responses = await asyncio.gather(*tasks)

    # Concatenate each response's upsert operations/output dicts
    upsert_operations = []
    for response in parse_responses:
        upsert_operations += response
    
    if not store_in_db:
        return upsert_operations

    # print("Starting Upsert", time.time() - start)

    # Execute bulk upsert operations
    try:
        if len(upsert_operations) > 0:
            # Create/reference "ratings" collection in db
            ratings = mongo_db.ratings
            ratings.bulk_write(upsert_operations, ordered=False)
    except BulkWriteError as bwe:
        pprint(bwe.details)

    # print("Finishing Upsert", time.time() - start)


async def get_ratings(usernames, db_cursor=None, mongo_db=None, store_in_db=True):
    start = time.time()
    # print("Function Start")

    # Loop through each user
    for i, username in enumerate(usernames):
        print(i, username, round((time.time() - start), 2))
        await get_user_ratings(username, db_cursor=db_cursor, mongo_db=mongo_db, store_in_db=store_in_db)
                

def main():
    from db_config import config

    # Connect to MongoDB Client
    db_name = config["MONGO_DB"]
    # client = motor.motor_asyncio.AsyncIOMotorClient(f'mongodb+srv://{config["MONGO_USERNAME"]}:{config["MONGO_PASSWORD"]}@cluster0.{config["MONGO_CLUSTER_ID"]}.mongodb.net/?retryWrites=true&w=majority')
    client = pymongo.MongoClient(f'mongodb+srv://{config["MONGO_USERNAME"]}:{config["MONGO_PASSWORD"]}@cluster0.{config["MONGO_CLUSTER_ID"]}.mongodb.net/{db_name}?retryWrites=true&w=majority')

    # Find letterboxd database and user collection
    db = client[db_name]
    users = db.users
    all_users = users.find({})
    all_usernames = [x['username'] for x in all_users]

    loop = asyncio.get_event_loop()

    # Find number of ratings pages for each user and add to their Mongo document (note: max of 128 scrapable pages)
    # future = asyncio.ensure_future(get_page_counts(all_usernames, users))
    future = asyncio.ensure_future(get_page_counts([], users))
    loop.run_until_complete(future)

    # Find and store ratings for each user
    future = asyncio.ensure_future(get_ratings(all_usernames, users, db))
    # future = asyncio.ensure_future(get_ratings(["samlearner", "colonelmortimer"], users, db))
    loop.run_until_complete(future)


if __name__ == "__main__":
    main()