#!/usr/local/bin/python3.12
import os

import pymongo


def connect_to_db():
    db_name = os.environ["MONGO_DB"]
    client = pymongo.MongoClient(
        os.environ["CONNECTION_URL"], server_api=pymongo.server_api.ServerApi("1")
    )

    return db_name, client
