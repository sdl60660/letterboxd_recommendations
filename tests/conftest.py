import asyncio
import os

import mongomock
import pytest
from dotenv import load_dotenv

# Load .env.test first if it exists, else fall back to .env
if os.path.exists(".env.test"):
    load_dotenv(".env.test", override=True)
else:
    load_dotenv(override=True)

TEST_DB_NAME = os.getenv("MONGO_DB", "letterboxd_test")


@pytest.fixture(scope="session")
def event_loop():
    # pytest-asyncio: create one loop for the whole session
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# @pytest.fixture(scope="session")
# def mongo_client():
#     uri = os.getenv("CONNECTION_URL", "mongodb://localhost:27017")
#     client = MongoClient(uri, serverSelectionTimeoutMS=5000)
#     yield client
#     # don't close in case other services use same client; harmless if we do:
#     client.close()


@pytest.fixture(scope="session")
def mongo_client():
    client = mongomock.MongoClient()  # in-memory Mongo
    yield client


@pytest.fixture(scope="function")
def mongo_db(mongo_client):
    db = mongo_client[TEST_DB_NAME]
    # clean before each test
    for coll in db.list_collection_names():
        db.drop_collection(coll)
    yield db
    # clean after each test
    mongo_client.drop_database(TEST_DB_NAME)


@pytest.fixture()
def html_sample_path():
    # helper to read sample HTML from testdata
    import pathlib

    base = pathlib.Path(__file__).parent.parent / "testdata" / "html"
    return base


# For mocking aiohttp HTTP calls in async code
@pytest.fixture()
def http_mock():
    from aioresponses import aioresponses

    with aioresponses() as m:
        yield m
