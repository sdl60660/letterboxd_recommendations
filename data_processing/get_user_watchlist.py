#!/usr/local/bin/python3.12

import asyncio
import os
from itertools import chain

from aiohttp import ClientSession, TCPConnector
from bs4 import BeautifulSoup

if os.getcwd().endswith("/data_processing"):
    from utils.http_utils import BROWSER_HEADERS
    from utils.selectors import LBX_REVIEW_TILE
    from utils.utils import get_page_count
else:
    from data_processing.utils.http_utils import BROWSER_HEADERS
    from data_processing.utils.selectors import (
        LBX_REVIEW_TILE,
    )
    from data_processing.utils.utils import get_page_count


async def fetch(url, session, input_data={}):
    async with session.get(url) as response:
        try:
            return await response.read(), input_data
        except:
            return None, None


async def parse_watchlist_page(response):
    # Parse ratings page response for each watchlist item, use lxml parser for speed
    soup = BeautifulSoup(response[0], "lxml")
    watchlist_movies = soup.find_all(*LBX_REVIEW_TILE)

    # Create empty array to store list of watchlist movie IDs
    movie_ids = []

    # For each review, parse data from scraped page and append ID to output array
    for watchlist_movie in watchlist_movies:
        rc = watchlist_movie.select_one("div.react-component")
        movie_id = rc.get("data-item-slug") if rc else None

        if not movie_id:
            continue

        movie_ids.append(movie_id)

    return movie_ids


async def get_user_watchlist(
    username,
    num_pages,
):
    url = "https://letterboxd.com/{}/watchlist/page/{}/"

    # Fetch all responses within one Client session,
    # keep connection alive for all requests.
    async with ClientSession(
        headers=BROWSER_HEADERS, connector=TCPConnector(limit=6)
    ) as session:
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


def get_watchlist_data(username):
    num_pages, _ = get_page_count(username, url="https://letterboxd.com/{}/watchlist")

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
