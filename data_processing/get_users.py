#!/usr/local/bin/python3.12

# import datetime
import os
import re

import requests
from bs4 import BeautifulSoup

# import datetime
from pymongo.operations import UpdateOne
from tqdm import tqdm

if os.getcwd().endswith("/data_processing"):
    from utils.db_connect import connect_to_db
    from utils.http_utils import BROWSER_HEADERS
    from utils.mongo_utils import safe_commit_ops
    from utils.selectors import LBX_USER_ROW, LBX_USER_TABLE

else:
    from data_processing.utils.db_connect import connect_to_db
    from data_processing.utils.http_utils import BROWSER_HEADERS
    from data_processing.utils.mongo_utils import safe_commit_ops
    from data_processing.utils.selectors import LBX_USER_ROW, LBX_USER_TABLE


def parse_user_tile(user_item):
    link = user_item.find("a")["href"]
    username = link.strip("/").lower()
    display_name = user_item.find("a", class_="name").text.strip()
    reviews_link = user_item.select_one('small.metadata a[href$="/reviews/"]')

    txt = reviews_link.get_text(" ", strip=True) if reviews_link else ""
    m = re.search(r"([\d,]+)\s*reviews", txt, flags=re.I)
    num_reviews = int(m.group(1).replace(",", "")) if m else 0

    user = {
        "username": username,
        "display_name": display_name,
        "num_reviews": num_reviews,
        # "last_updated": datetime.datetime.now(datetime.timezone.utc),
    }

    return user


def parse_user_list_page(html, data_as_ops=True):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find(*LBX_USER_TABLE)
    table_items = table.find_all(*LBX_USER_ROW)

    user_data_list = []
    for user_tile in table_items:
        user = parse_user_tile(user_tile)
        user_data_list.append(user)

    return user_data_list


def form_user_upsert_op(record):
    return UpdateOne({"username": record["username"]}, {"$set": record}, upsert=True)


def process_user_page(base_url, page, users, send_to_db=True):
    r = requests.get(base_url.format(page), headers=BROWSER_HEADERS)
    all_user_data = parse_user_list_page(r.text)

    update_operations = [form_user_upsert_op(user) for user in all_user_data]

    if send_to_db:
        safe_commit_ops(users, update_operations)

    return {"data": all_user_data, "ops": update_operations}


def main():
    # Connect to MongoDB client
    db_name, client = connect_to_db()

    db = client[db_name]
    users = db.users

    base_url = "https://letterboxd.com/members/popular/this/week/page/{}/"

    total_pages = 128
    pbar = tqdm(range(1, total_pages + 1))
    for page in pbar:
        pbar.set_description(f"Scraping page {page} of {total_pages} of top users")
        process_user_page(base_url, page, users, send_to_db=True)


if __name__ == "__main__":
    main()
