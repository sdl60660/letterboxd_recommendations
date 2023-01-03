#!/usr/local/bin/python3.9

from pymongo.operations import ReplaceOne, UpdateOne
import requests
from bs4 import BeautifulSoup

import pymongo
from pymongo.errors import BulkWriteError
from db_config import config

from pprint import pprint
from tqdm import tqdm


db_name = config["MONGO_DB"]

if "CONNECTION_URL" in config.keys():
    client = pymongo.MongoClient(config["CONNECTION_URL"], server_api=pymongo.server_api.ServerApi('1'))
else:
    client = pymongo.MongoClient(f'mongodb+srv://{config["MONGO_USERNAME"]}:{config["MONGO_PASSWORD"]}@cluster0.{config["MONGO_CLUSTER_ID"]}.mongodb.net/{db_name}?retryWrites=true&w=majority')

db = client[db_name]
users = db.users

base_url = "https://letterboxd.com/members/popular/page/{}"

total_pages = 128
pbar = tqdm(range(1, total_pages+1))
for page in pbar:
    pbar.set_description(f"Scraping page {page} of {total_pages} of top users")

    r = requests.get(base_url.format(page))
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", attrs={"class": "person-table"})
    rows = table.findAll("td", attrs={"class": "table-person"})
    
    update_operations = []
    for row in rows:
        link = row.find("a")["href"]
        username = link.strip('/')
        display_name = row.find("a", attrs={"class": "name"}).text.strip()
        num_reviews = int(row.find("small").find("a").text.replace('\xa0', ' ').split()[0].replace(',', ''))

        user = {
            "username": username,
            "display_name": display_name,
            "num_reviews": num_reviews
        }

        update_operations.append(
            UpdateOne({
                "username": user["username"]
                },
                {"$set": user},
                upsert=True
            )
        )

        # users.update_one({"username": user["username"]}, {"$set": user}, upsert=True)
        
    try:
        if len(update_operations) > 0:
            users.bulk_write(update_operations, ordered=False)
    except BulkWriteError as bwe:
        pprint(bwe.details)