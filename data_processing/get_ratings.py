#!/usr/local/bin/python3.9

import asyncio
from aiohttp import ClientSession
import requests
from pprint import pprint

from bs4 import BeautifulSoup

import pymongo
from pymongo import UpdateOne
from pymongo.errors import BulkWriteError

import motor.motor_asyncio
from config import config

import time


async def fetch(url, session, input_data={}):
    async with session.get(url) as response:
        if input_data != {}:
            return await response.read(), input_data
        else:
            return await response.read()
            

async def get_page_counts(usernames, db_cursor):
    url = "https://letterboxd.com/{}/films/ratings/"
    tasks = []

    async with ClientSession() as session:
        for username in usernames:
            task = asyncio.ensure_future(fetch(url.format(username), session, {"username": username}))
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)

        for response in responses:
            soup = BeautifulSoup(response[0], "html.parser")
            try:
                page_link = soup.findAll("li", attrs={"class", "paginate-page"})[-1]
                num_pages = int(page_link.find("a").text.replace(',', ''))
            except IndexError:
                num_pages = 1

            # print(num_pages, response[1])
            users.update_one({"username": response[1]['username']}, {"$set": {"num_ratings_pages": num_pages}})


async def generate_rating_upsert_operations(response):
    
    # Parse ratings page response for each rating/review, use lxml parser for speed
    soup = BeautifulSoup(response[0], "lxml")
    reviews = soup.findAll("li", attrs={"class": "poster-container"})

    # Create empty array to store list of bulk operations
    operations = []

    # For each review, parse data from scraped page and append an UpdateOne operation for bulk execution
    for review in reviews:
        movie_id = review.find('div', attrs={"class", "film-poster"})['data-target-link'].split('/')[-2]

        rating_class = review.find("span", attrs={"class": "rating"})['class'][1]
        rating_val = int(rating_class.split('-')[-1])

        operations.append(UpdateOne({
                "user_id": response[1]["username"],
                "movie_id": movie_id
            },
            {
                "$set": {
                    "movie_id": movie_id,
                    "rating_val": rating_val,
                    "user_id": response[1]["username"]
                }
            }, upsert=True))
    
    return operations
    


async def get_ratings(usernames, db_cursor, mongo_db):
    start = time.time()
    print("Function Start")

    url = "https://letterboxd.com/{}/films/ratings/page/{}/"

    # Create/reference "ratings" collection in db
    ratings = db.ratings

    # Loop through each user
    for i, username in enumerate(usernames):
        print(i, username)
        
        # Find them in the MongoDB database and grab the number of ratings pages
        user = db_cursor.find_one({"username": username})
        num_pages = user["num_ratings_pages"]

        # Fetch all responses within one Client session,
        # keep connection alive for all requests.
        async with ClientSession() as session:
            print("Starting Scrape", time.time() - start)

            tasks = []
            # Make a request for each ratings page and add to task queue
            for i in range(num_pages):
                task = asyncio.ensure_future(fetch(url.format(username, i), session, user))
                tasks.append(task)

            # Gather all ratings page responses
            scrape_responses = await asyncio.gather(*tasks)

            print("Finishing Scrape", time.time() - start)
            
            # Process each ratings page response, converting it into bulk upsert operations
            tasks = []
            for response in scrape_responses:
                task = asyncio.ensure_future(generate_rating_upsert_operations(response))
                tasks.append(task)
            
            parse_responses = await asyncio.gather(*tasks)

            # Concatenate each response's upsert operations
            upsert_operations = []
            for response in parse_responses:
                upsert_operations += response

            print("Starting Upsert", time.time() - start)

            # Execute bulk upsert operations
            try:
                if len(upsert_operations) > 0:
                    ratings.bulk_write(upsert_operations, ordered=False)
            except BulkWriteError as bwe:
                pprint(bwe.details)

            print("Finishing Upsert", time.time() - start)
                

# Connect to MongoDB Client
db_name = config["MONGO_DB"]
# client = motor.motor_asyncio.AsyncIOMotorClient(f'mongodb+srv://{config["MONGO_USERNAME"]}:{config["MONGO_PASSWORD"]}@cluster0.{config["MONGO_CLUSTER_ID"]}.mongodb.net/?retryWrites=true&w=majority')
client = pymongo.MongoClient(f'mongodb+srv://{config["MONGO_USERNAME"]}:{config["MONGO_PASSWORD"]}@cluster0.{config["MONGO_CLUSTER_ID"]}.mongodb.net/{db_name}?retryWrites=true&w=majority')

# Find letterboxd database and user collection
db = client[db_name]
users = db.users
all_users = users.find({})
all_usernames = [x['username'] for x in all_users] + ["samlearner"]

loop = asyncio.get_event_loop()

# Find number of ratings pages for each user and add to their Mongo document (note: max of 128 scrapable pages)
# future = asyncio.ensure_future(get_page_counts(all_usernames, users))
future = asyncio.ensure_future(get_page_counts([], users))
loop.run_until_complete(future)

# Find and store ratings for each user
future = asyncio.ensure_future(get_ratings(all_usernames[1800:], users, db))
# future = asyncio.ensure_future(get_ratings(["samlearner"], users, db))
loop.run_until_complete(future)