#!/usr/local/bin/python3.12
import os
import sys

from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from pymongo.server_api import ServerApi


def connect_to_db():
    uri = os.getenv("CONNECTION_URL")
    db_name = os.getenv("MONGO_DB")

    if not uri or not db_name:
        print(
            "❌ Missing CONNECTION_URL or MONGO_DB. Copy .env.example to .env and fill it in.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    try:
        client = MongoClient(
            uri,
            server_api=ServerApi("1"),
            serverSelectionTimeoutMS=4000,
        )
        client.admin.command("ping")
        return db_name, client
    except ServerSelectionTimeoutError:
        print(
            f"❌ Could not reach Mongo at {uri}. If using Docker, run: docker compose up -d",
            file=sys.stderr,
        )
        raise SystemExit(2)
