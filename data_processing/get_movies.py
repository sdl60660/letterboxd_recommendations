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

# This is a MongoDB connection URL and a TMDB API key imported from a file in the .gitignore
from db_config import config, tmdb_key


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

        try:
            imdb_link = soup.find("a", attrs={"data-track-action": "IMDb"})['href']
            imdb_id = imdb_link.split('/title')[1].strip('/').split('/')[0]
        except:
            imdb_link = ''
            imdb_id = ''

        try:
            tmdb_link = soup.find("a", attrs={"data-track-action": "TMDb"})['href']
            tmdb_id = tmdb_link.split('/movie')[1].strip('/').split('/')[0]
        except:
            tmdb_link = ''
            tmdb_id = ''
        
        movie_object = {
                    "movie_id": input_data["movie_id"],
                    "movie_title": movie_title,
                    "image_url": image_url,
                    "year_released": year,
                    "imdb_link": imdb_link,
                    "tmdb_link": tmdb_link,
                    "imdb_id": imdb_id,
                    "tmdb_id": tmdb_id
                }

        update_operation = UpdateOne({
                "movie_id": input_data["movie_id"]
            },
            {
                "$set": movie_object
            }, upsert=True)


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
        
        simple_fields = ["popularity", "overview", "runtime", "vote_average", "vote_count", "release_date", "original_language"]
        for field_name in simple_fields:
            try:
                movie_object[field_name] = response[field_name]
            except:
                movie_object[field_name] = None

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

async def get_rich_data(movie_list, db_cursor, mongo_db):
    base_url = "https://api.themoviedb.org/3/movie/{}?api_key={}"

    async with ClientSession() as session:
        # print("Starting Scrape", time.time() - start)

        tasks = []
        movie_list = [x for x in movie_list if x['tmdb_id']]
        # Make a request for each ratings page and add to task queue
        for movie in movie_list:
            # print(base_url.format(movie["tmdb_id"], tmdb_key))
            task = asyncio.ensure_future(fetch_tmdb_data(base_url.format(movie["tmdb_id"], tmdb_key), session, movie, {"movie_id": movie["movie_id"]}))
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


def main(data_type="letterboxd"):
    # Connect to MongoDB client
    try:
        from db_config import config

        db_name = config["MONGO_DB"]
        if "CONNECTION_URL" in config.keys():
            client = pymongo.MongoClient(config["CONNECTION_URL"], server_api=pymongo.server_api.ServerApi('1'))
        else:
            client = pymongo.MongoClient(f'mongodb+srv://{config["MONGO_USERNAME"]}:{config["MONGO_PASSWORD"]}@cluster0.{config["MONGO_CLUSTER_ID"]}.mongodb.net/{db_name}?retryWrites=true&w=majority')

    except ModuleNotFoundError:
        # If not running locally, since db_config data is not committed to git
        import os
        db_name = os.environ['MONGO_DB']
        client = pymongo.MongoClient(os.environ["CONNECTION_URL"], server_api=pymongo.server_api.ServerApi('1'))

    db = client[db_name]
    movies = db.movies

    # Find all movies with missing metadata, which implies that they were added during get_ratings and have not been scraped yet
    # All other movies have already had their data scraped and since this is almost always unchanging data, we won't rescrape 200,000+ records
    # all_movies = [x['movie_id'] for x in list(movies.find({ "year_released": { "$exists": False }, "movie_title": { "$exists": False}}))]
    if data_type == "letterboxd":
        all_movies = [x['movie_id'] for x in list(movies.find({ "tmdb_id": { "$exists": False }}))]
    else:
        all_movies = [x for x in list(movies.find({ "genres": { "$eq": None }, "tmdb_id": {"$ne": ""}, "tmdb_id": { "$exists": True }}))]
    
    loop = asyncio.get_event_loop()
    chunk_size = 10
    num_chunks = len(all_movies) // chunk_size + 1

    print("Total Movies to Scrape:", len(all_movies))
    print('Total Chunks:', num_chunks)
    print("==========================\n")

    pbar = tqdm(range(num_chunks))
    for chunk_i in pbar:
        pbar.set_description(f"Scraping chunk {chunk_i+1} of {num_chunks}")

        if chunk_i == num_chunks - 1:
            chunk = all_movies[chunk_i*chunk_size:]
        else:
            chunk = all_movies[chunk_i*chunk_size:(chunk_i+1)*chunk_size]

        for attempt in range(5):
            try:
                if data_type == "letterboxd":
                    future = asyncio.ensure_future(get_movies(chunk, movies, db))
                else:
                    future = asyncio.ensure_future(get_rich_data(chunk, movies, db))
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