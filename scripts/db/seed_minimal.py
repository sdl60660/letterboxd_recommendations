import datetime
import os

from pymongo import MongoClient

client = MongoClient(os.environ.get("CONNECTION_URL", "mongodb://mongo:27017"))
db = client[os.environ.get("MONGO_DB", "letterboxd")]

db.users.update_one(
    {"username": "samtestacct"},
    {
        "$set": {
            "username": "samtestacct",
            "display_name": "Test Account",
            "num_reviews": 0,
            "last_updated": datetime.datetime.utcnow(),
        }
    },
    upsert=True,
)

db.movies.update_one(
    {"movie_id": "the-zone-of-interest"},
    {
        "$set": {
            "movie_id": "the-zone-of-interest",
            "movie_title": "The Zone of Interest",
        }
    },
    upsert=True,
)

print("Seeded minimal users/movies.")
