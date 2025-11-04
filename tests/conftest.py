import asyncio
import os
import pathlib
import socket

import mongomock
import pytest
from dotenv import load_dotenv

# We're no longer using a test db, instead only using mongomock, so no need for separate .env
# if os.path.exists(".env.test"):
#     load_dotenv(".env.test", override=True)
# else:
#     load_dotenv(override=True)

load_dotenv(override=True)

TEST_DB_NAME = os.getenv("MONGO_DB", "letterboxd_test")


def _internet_available() -> bool:
    try:
        with socket.create_connection(("8.8.8.8", 53), timeout=2):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    online = _internet_available()
    if online:
        return

    skip_live = pytest.mark.skip(reason="No internet connection — skipping live tests.")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture(scope="session")
def event_loop():
    # pytest-asyncio: create one loop for the whole session
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


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
    return pathlib.Path(__file__).parent / "testdata" / "html"


# For mocking aiohttp HTTP calls in async code
@pytest.fixture()
def http_mock():
    from aioresponses import aioresponses

    with aioresponses() as m:
        yield m
