#!/usr/local/bin/python3.9

from bs4 import BeautifulSoup
import requests

import asyncio
from aiohttp import ClientSession

import os
if os.getcwd().endswith("/data_processing"):
    from get_ratings import get_user_ratings
else:
    from data_processing.get_ratings import get_user_ratings


def get_page_count(username):
    url = "https://letterboxd.com/{}/films/by/date"
    r = requests.get(url.format(username))

    soup = BeautifulSoup(r.text, "lxml")
    
    body = soup.find("body")
    if "error" in body["class"]:
        return -1

    try:
        page_link = soup.findAll("li", attrs={"class", "paginate-page"})[-1]
        num_pages = int(page_link.find("a").text.replace(',', ''))
    except IndexError:
        num_pages = 1

    return num_pages

def get_user_data(username):
    num_pages = get_page_count(username)
    
    if num_pages == -1:
        return [], "user_not_found"

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    future = asyncio.ensure_future(get_user_ratings(username, db_cursor=None, mongo_db=None, store_in_db=False, num_pages=num_pages, return_unrated=True))
    loop.run_until_complete(future)
    # print(future.result())

    return future.result(), "success"

if __name__ == "__main__":
    username = "samlearner"
    get_user_data(username)
