#!/usr/local/bin/python3.13

from re import U
from bs4 import BeautifulSoup
from pymongo.operations import ReplaceOne
import requests
from itertools import chain

import asyncio
from aiohttp import ClientSession, TCPConnector

import pymongo
from pymongo import UpdateOne, ReplaceOne
from pymongo.errors import BulkWriteError


from pprint import pprint


BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


async def fetch(url, session, input_data={}):
    async with session.get(url) as response:
        try:
            return await response.read(), input_data
        except:
            return None, None


async def parse_watchlist_page(response):
    # Parse ratings page response for each watchlist item, use lxml parser for speed
    soup = BeautifulSoup(response[0], "lxml")
    watchlist_movies = soup.findAll("li", attrs={"class": "poster-container"})

    # Create empty array to store list of watchlist movie IDs
    movie_ids = []

    # For each review, parse data from scraped page and append ID to output array
    for watchlist_movie in watchlist_movies:
        movie_id = watchlist_movie.find("div", attrs={"class", "film-poster"})[
            "data-target-link"
        ].split("/")[-2]
        movie_ids.append(movie_id)

    return movie_ids


async def get_user_watchlist(
    username,
    num_pages,
):
    url = "https://letterboxd.com/{}/watchlist/page/{}/"

    # Fetch all responses within one Client session,
    # keep connection alive for all requests.
    async with ClientSession(headers=BROWSER_HEADERS, connector=TCPConnector(limit=6)) as session:
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
        task = asyncio.ensure_future(parse_watchlist_page(response))
        tasks.append(task)

    parse_responses = await asyncio.gather(*tasks)
    parse_responses = list(chain.from_iterable(parse_responses))

    return parse_responses


def get_page_count(username):
    url = "https://letterboxd.com/{}/watchlist"
    r = requests.get(url.format(username))

    soup = BeautifulSoup(r.text, "lxml")

    body = soup.find("body")

    try:
        if "error" in body["class"]:
            return -1
    except KeyError:
        print(body)
        return -1

    try:
        page_link = soup.findAll("li", attrs={"class", "paginate-page"})[-1]
        num_pages = int(page_link.find("a").text.replace(",", ""))
    except IndexError:
        num_pages = 1

    return num_pages


def get_watchlist_data(username):
    num_pages = get_page_count(username)

    if num_pages == -1:
        return [], "user_not_found"

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    future = asyncio.ensure_future(
        get_user_watchlist(
            username,
            num_pages=num_pages,
        )
    )
    loop.run_until_complete(future)

    return future.result(), "success"


if __name__ == "__main__":
    username = "samlearner"
    get_watchlist_data(username)
