#!/usr/local/bin/python3.9

import requests
from bs4 import BeautifulSoup

import asyncio
from aiohttp import ClientSession
import requests
from pprint import pprint

import pymongo
import pandas as pd

import time
from tqdm import tqdm

import pymongo
from pymongo import UpdateOne
from pymongo.errors import BulkWriteError

from db_config import config


async def fetch(url, session, input_data={}):
    async with session.get(url) as r:
        response = await r.read()

        # Parse ratings page response for each rating/review, use lxml parser for speed
        soup = BeautifulSoup(response, "lxml")
        # rating = review.find("span", attrs={"class": "rating"})
        
        movie_header = soup.find('section', attrs={'id': 'featured-film-header'})

        try:
            movie_title = movie_header.find('h1').text
        except AttributeError:
            movie_title = ''

        try:
            year = int(movie_header.find('small', attrs={'class': 'number'}).find('a').text)
        except AttributeError:
            year = None

        try:
            image_url = soup.find('div', attrs={'class': 'film-poster'}).find('img')['src'].split('?')[0]
            image_url = image_url.replace('https://a.ltrbxd.com/resized/', '').split('.jpg')[0]
            if 'https://s.ltrbxd.com/static/img/empty-poster' in image_url:
                image_url = ''
        except AttributeError:
            image_url = ''
        
        # JS Component
        # try:
        #     watches = soup.find('ul', attrs={'class': 'film-stats'}).find('li', attrs={'class', 'filmstat-watches'}).find('a')['data-original-title']
        #     watches = int(watches.strip('Watched by ').strip(' members').replace(',', ''))
        # except:
        #     watches = None

        soup.find("span", attrs={"class": "rating"})
        
        movie_object = {
                    "movie_id": input_data["movie_id"],
                    "movie_title": movie_title,
                    "image_url": image_url,
                    "year_released": year
                }

        update_operation = UpdateOne({
                "movie_id": input_data["movie_id"]
            },
            {
                "$set": movie_object
            }, upsert=True)


        return update_operation


async def get_movies(movie_list, db_cursor, mongo_db):
    url = "https://letterboxd.com/film/{}/"
    
    async with ClientSession() as session:
        # print("Starting Scrape", time.time() - start)

        tasks = []
        # Make a request for each ratings page and add to task queue
        for movie in movie_list:
            task = asyncio.ensure_future(fetch(url.format(movie), session, {"movie_id": movie}))
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


def main():
    db_name = config["MONGO_DB"]

    if "CONNECTION_URL" in config.keys():
        client = pymongo.MongoClient(config["CONNECTION_URL"], server_api=pymongo.server_api.ServerApi('1'))
    else:
        client = pymongo.MongoClient(f'mongodb+srv://{config["MONGO_USERNAME"]}:{config["MONGO_PASSWORD"]}@cluster0.{config["MONGO_CLUSTER_ID"]}.mongodb.net/{db_name}?retryWrites=true&w=majority')

    db = client[db_name]
    movies = db.movies

    # Find all movies with missing metadata, which implies that they were added during get_ratings and have not been scraped yet
    # All other movies have already had their data scraped and since this is almost always unchanging data, we won't rescrape 200,000+ records
    all_movies = [x['movie_id'] for x in list(movies.find({ "year_released": { "$exists": False }, "movie_title": { "$exists": False}}))]
    
    loop = asyncio.get_event_loop()
    chunk_size = 500
    num_chunks = len(all_movies) // chunk_size + 1

    print("Total Movies to Scrape:", len(all_movies))
    print('Total Chunks:', num_chunks)
    print("==========================\n")

    pbar = tqdm(range(num_chunks))
    for chunk in pbar:
        pbar.set_description(f"Scraping chunk {chunk+1} of {num_chunks}")

        if chunk == num_chunks - 1:
            chunk = all_movies[chunk*chunk_size:]
        else:
            chunk = all_movies[chunk*chunk_size:(chunk+1)*chunk_size]

        for attempt in range(5):
            try:
                future = asyncio.ensure_future(get_movies(chunk, movies, db))
                loop.run_until_complete(future)
            except:
                print(f"Error on attempt {attempt+1}, retrying...")
            else:
                break
        else:
            print(f"Count not complete requests for chunk {attempt+1}")
        


if __name__ == "__main__":
    main()