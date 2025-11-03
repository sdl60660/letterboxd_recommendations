#!/usr/local/bin/python3.12

import os
import re
from pprint import pprint

import requests
from bs4 import BeautifulSoup

# import datetime
from pymongo.errors import BulkWriteError
from pymongo.operations import UpdateOne
from tqdm import tqdm

if os.getcwd().endswith("/data_processing"):
    from utils.db_connect import connect_to_db
    from utils.http_utils import BROWSER_HEADERS

else:
    from data_processing.utils.db_connect import connect_to_db
    from data_processing.utils.http_utils import BROWSER_HEADERS


# Connect to MongoDB client
db_name, client = connect_to_db()

db = client[db_name]
users = db.users

# base_url = "https://letterboxd.com/members/popular/page/{}"
base_url = "https://letterboxd.com/members/popular/this/week/page/{}/"

total_pages = 128
pbar = tqdm(range(1, total_pages + 1))
for page in pbar:
    pbar.set_description(f"Scraping page {page} of {total_pages} of top users")

    r = requests.get(base_url.format(page), headers=BROWSER_HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", class_="member-table")
    rows = table.find_all("td", class_="table-person")

    update_operations = []
    for row in rows:
        link = row.find("a")["href"]
        username = link.strip("/").lower()
        display_name = row.find("a", class_="name").text.strip()
        reviews_link = row.select_one('small.metadata a[href$="/reviews/"]')

        txt = reviews_link.get_text(" ", strip=True) if reviews_link else ""
        m = re.search(r"([\d,]+)\s*reviews", txt, flags=re.I)
        num_reviews = int(m.group(1).replace(",", "")) if m else 0

        user = {
            "username": username,
            "display_name": display_name,
            "num_reviews": num_reviews,
            # "last_updated": datetime.datetime.now(datetime.timezone.utc),
        }

        update_operations.append(
            UpdateOne({"username": user["username"]}, {"$set": user}, upsert=True)
        )

        # users.update_one({"username": user["username"]}, {"$set": user}, upsert=True)

    try:
        if len(update_operations) > 0:
            users.bulk_write(update_operations, ordered=False)
    except BulkWriteError as bwe:
        pprint(bwe.details)
