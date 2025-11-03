#!/usr/local/bin/python3.12

import asyncio
import datetime
import json
import os
import re
import traceback
from pprint import pprint
from urllib.parse import urlparse

from aiohttp import ClientSession, TCPConnector
from bs4 import BeautifulSoup
from db_connect import connect_to_db
from pymongo import UpdateOne
from pymongo.errors import BulkWriteError
from tqdm import tqdm

if os.getcwd().endswith("/data_processing"):
    from http_utils import BROWSER_HEADERS

else:
    from data_processing.http_utils import BROWSER_HEADERS


_IMDB_ID_RE = re.compile(r"/title/([A-Za-z0-9]+)/?")
_TMDB_MOVIE_RE = re.compile(r"/movie/(\d+)/?")
_TMDB_TV_RE = re.compile(r"/tv/(\d+)/?")


def format_img_link_stub(raw_link):
    image_url = raw_link.replace("https://a.ltrbxd.com/resized/", "").split(".jpg")[0]

    if "https://s.ltrbxd.com/static/img/empty-poster" in raw_link:
        image_url = ""

    return image_url


def extract_movie_id_from_url(url: str) -> str | None:
    m = re.search(r"/film/([^/]+)/", str(url))
    return m.group(1) if m else None


def get_meta_data_from_script_tag(soup):
    # find the <script type="application/ld+json"> tag
    data_script = soup.find("script", attrs={"type": "application/ld+json"})
    if data_script and data_script.string:
        # clean out the /* <![CDATA[ */ and /* ]]> */ wrappers if present
        raw_json = data_script.string.strip()
        raw_json = (
            raw_json.replace("/* <![CDATA[ */", "").replace("/* ]]> */", "").strip()
        )

        data = json.loads(raw_json)
        image_url = data.get("image", "")
        genres = data.get("genre")

        rating_data = data.get("aggregateRating", {})
        rating_count = rating_data.get("ratingCount")
        avg_rating = rating_data.get("ratingValue")

        return {
            "image_url": image_url,
            "letterboxd_rating_count": rating_count,
            "letterboxd_avg_rating": avg_rating,
            "letterboxd_genres": genres,
        }


def _safe_text(tag, default=""):
    return tag.get_text(strip=True) if tag else default


def _safe_int(s, default=None):
    try:
        return int(s)
    except (TypeError, ValueError):
        return default


def _attr(tag, name, default=None):
    # returns tag[name] if present; else default
    try:
        return tag.attrs.get(name, default)
    except AttributeError:
        return default


def _extract_imdb_id(url: str | None) -> str:
    if not url:
        return ""
    m = _IMDB_ID_RE.search(url)
    return m.group(1) if m else ""


def _extract_tmdb(url: str | None):
    if not url:
        return None, ""
    path = urlparse(url).path or ""
    m_movie = _TMDB_MOVIE_RE.search(path)
    if m_movie:
        return "movie", m_movie.group(1)
    m_tv = _TMDB_TV_RE.search(path)
    if m_tv:
        return "tv", m_tv.group(1)
    return None, ""


def parse_letterboxd_page_data(response: str, movie_id: str) -> dict:
    soup = BeautifulSoup(response, "lxml")

    header = soup.find("section", class_="production-masthead")

    # title
    title_el = header.find("h1") if header else None
    movie_title = title_el.get_text(strip=True) if title_el else ""

    # year
    rel_span = header.find("span", class_="releasedate") if header else None
    rel_a = rel_span.find("a") if rel_span else None
    year_text = rel_a.get_text(strip=True) if rel_a else None
    try:
        year = int(year_text) if year_text and year_text.isdigit() else None
    except ValueError:
        year = None

    # imdb
    imdb_link = _attr(soup.find("a", attrs={"data-track-action": "IMDb"}), "href", "")
    imdb_id = _extract_imdb_id(imdb_link)

    # tmdb
    tmdb_link = _attr(soup.find("a", attrs={"data-track-action": "TMDB"}), "href", "")
    content_type, tmdb_id = _extract_tmdb(tmdb_link)

    movie_update_object = {
        "movie_id": movie_id,
        "movie_title": movie_title,
        "year_released": year,
        "imdb_link": imdb_link or "",
        "tmdb_link": tmdb_link or "",
        "imdb_id": imdb_id,
        "tmdb_id": tmdb_id,
        "content_type": content_type,  # None if unknown
        "scrape_status": "ok",
        "fail_count": 0,
        "next_retry_at": None,
    }

    # script-tag metadata: be explicit about the expected failures
    try:
        script_tag_data = get_meta_data_from_script_tag(soup)  # your function
    except (KeyError, TypeError, AttributeError, ValueError):
        script_tag_data = None

    if script_tag_data:
        for k, v in script_tag_data.items():
            if v is None:
                continue
            elif k == "image_url":
                try:
                    movie_update_object[k] = format_img_link_stub(v)
                except Exception:
                    # limit scope; if format fails, still ensure string
                    movie_update_object[k] = v or ""
            else:
                movie_update_object[k] = v
    else:
        # important to set an explicit empty string for poster
        movie_update_object["image_url"] = ""

    return movie_update_object


def format_failed_update(movie_id, fail_count):
    # backoff_days = get_backoff_days(fail_count)
    backoff_days = 7
    now = datetime.datetime.now(datetime.timezone.utc)
    next_retry = now + datetime.timedelta(days=backoff_days)

    movie_update_object = {
        "movie_id": movie_id,
        "scrape_status": "failed",
        "next_retry_at": next_retry,
    }

    return movie_update_object


def handle_redirects(r, movie_update_object, movie_id, mongo_db):
    redirect_to = extract_movie_id_from_url(r.real_url)

    if redirect_to and redirect_to != movie_id:
        movie_update_object["redirect_to"] = redirect_to
        movie_update_object["redirect_seen_at"] = datetime.datetime.now(
            datetime.timezone.utc
        )

    # small helper collection
    redirects_coll = mongo_db.movie_redirects
    redirects_coll.update_one(
        {"old_id": movie_id},
        {
            "$set": {
                "old_id": movie_id,
                "new_id": redirect_to,
                "last_seen_at": datetime.datetime.now(datetime.timezone.utc),
                "status": "pending",  # will flip to 'merged' after consolidation
            }
        },
        upsert=True,
    )


async def fetch_letterboxd(url, session, mongo_db, input_data={}):
    async with session.get(url) as r:
        response = await r.read()

        movie_id = input_data["movie_id"]
        if r.status == 404:
            fail_count = input_data.get("fail_count", 0) + 1
            movie_update_object = format_failed_update(movie_id, fail_count)
            update_operation = UpdateOne(
                {"movie_id": input_data["movie_id"]},
                {"$set": movie_update_object, "$inc": {"fail_count": 1}},
                upsert=True,
            )
        else:
            movie_update_object = parse_letterboxd_page_data(response, movie_id)
            update_operation = UpdateOne(
                {"movie_id": input_data["movie_id"]},
                {"$set": movie_update_object},
                upsert=True,
            )

        # handle logging any redirects (to be updated in the database later, but flagged for now)
        if r.history:
            movie_update_object = handle_redirects(
                r, movie_update_object, movie_id, mongo_db
            )

        return update_operation


async def fetch_tmdb_data(url, session, movie_data, input_data={}):
    async with session.get(url) as r:
        response = await r.json()

        movie_object = movie_data

        object_fields = ["genres", "production_countries", "spoken_languages"]
        for field_name in object_fields:
            data = response.get(field_name)
            if isinstance(data, list):
                movie_object[field_name] = [x.get("name") for x in data if "name" in x]
            else:
                movie_object[field_name] = None

        simple_fields = [
            "popularity",
            "overview",
            "runtime",
            "vote_average",
            "vote_count",
            "release_date",
            "original_language",
        ]
        for field_name in simple_fields:
            movie_object[field_name] = response.get(field_name)

        movie_object["last_updated"] = datetime.datetime.now(datetime.timezone.utc)

        update_operation = UpdateOne(
            {"movie_id": input_data["movie_id"]}, {"$set": movie_object}, upsert=True
        )

        return update_operation


async def get_movies(movie_list, db_cursor, mongo_db):
    url = "https://letterboxd.com/film/{}/"

    async with ClientSession() as session:
        tasks = []
        # Make a request for each ratings page and add to task queue
        for movie in movie_list:
            task = asyncio.ensure_future(
                fetch_letterboxd(
                    url.format(movie), session, mongo_db, {"movie_id": movie}
                )
            )
            tasks.append(task)

        # Gather all ratings page responses
        upsert_operations = await asyncio.gather(*tasks)

    try:
        if len(upsert_operations) > 0:
            # Create/reference "ratings" collection in db
            movies = mongo_db.movies
            movies.bulk_write(upsert_operations, ordered=False)
    except BulkWriteError as bwe:
        pprint(bwe.details)


async def get_rich_data(movie_list, db_cursor, mongo_db, tmdb_key):
    base_url = "https://api.themoviedb.org/3/{}/{}?api_key={}"

    async with ClientSession(
        headers=BROWSER_HEADERS, connector=TCPConnector(limit=6)
    ) as session:
        tasks = []
        movie_list = [x for x in movie_list if x["tmdb_id"]]
        # Make a request for each ratings page and add to task queue
        for movie in movie_list:
            content_type = movie["content_type"] or "movie"
            task = asyncio.ensure_future(
                fetch_tmdb_data(
                    base_url.format(content_type, movie["tmdb_id"], tmdb_key),
                    session,
                    movie,
                    {"movie_id": movie["movie_id"]},
                )
            )
            tasks.append(task)

        # Gather all ratings page responses
        upsert_operations = await asyncio.gather(*tasks)

    try:
        if len(upsert_operations) > 0:
            # Create/reference "ratings" collection in db
            movies = mongo_db.movies
            movies.bulk_write(upsert_operations, ordered=False)
    except BulkWriteError as bwe:
        pprint(bwe.details)


def get_ids_for_update(movies_collection, data_type):
    now = datetime.datetime.now(datetime.timezone.utc)
    one_month_ago = now - datetime.timedelta(days=30)

    # Find all movies with missing metadata, which implies that they were added during get_ratings and have not been scraped yet
    # All other movies have already had their data scraped and since this is almost always unchanging data, we won't rescrape 200,000+ records
    if data_type == "letterboxd":
        update_ids = set()

        # 1000 least recently updated items, excluding anything updated in the last month
        update_ids |= {
            x["movie_id"]
            for x in movies_collection.find(
                {"last_updated": {"$lte": one_month_ago}}, {"movie_id": 1}
            )
            .sort("last_updated", 1)
            .limit(1000)
        }

        # grab a sample of those which had a failed crawl and are now due for a retry
        update_ids |= {
            x["movie_id"]
            for x in movies_collection.find(
                # {"scrape_status": "failed"},
                {"next_retry_at": {"$lte": now}, "scrape_status": "failed"},
                {"movie_id": 1},
            ).sort("next_retry_at", 1)
        }

        # backfill a chunk of the records that are missing 'content_type' (newly-added)
        update_ids |= {
            x["movie_id"]
            for x in movies_collection.find(
                {"content_type": {"$exists": False}},
                {"movie_id": 1},
            ).limit(10000)
        }

        # anything newly added or missing key data (including missing poster image)
        update_ids |= {
            x["movie_id"]
            for x in movies_collection.find(
                {
                    "$or": [
                        {"movie_title": {"$exists": False}},
                        {"tmdb_id": {"$exists": False}},
                        {"image_url": {"$exists": False}},
                        {"year_released": {"$exists": False}},
                        {"year_released": {"$in": ["", None]}},
                    ]
                },
                {"movie_id": 1},
            )
        }

        # missing key data (but has been attempted before), limited to a batch of 500 per update
        update_ids |= {
            x["movie_id"]
            for x in movies_collection.find(
                {
                    "$and": [
                        {
                            "$or": [
                                {"movie_title": {"$in": ["", None]}},
                                {"tmdb_id": {"$in": ["", None]}},
                                {"image_url": {"$in": ["", None]}},
                                {"content_type": {"$in": ["", None]}},
                            ]
                        },
                        {
                            "$or": [
                                {"last_updated": {"$exists": False}},
                                {"last_updated": {"$lte": one_month_ago}},
                            ]
                        },
                    ]
                },
                {"movie_id": 1},
            )
            .sort("last_updated", 1)
            .limit(500)
        }

        all_movies = list(update_ids)

    else:
        all_movies = [
            x
            for x in list(
                movies_collection.find(
                    {
                        "genres": {"$exists": False},
                        "content_type": {"$exists": True, "$ne": None},
                    }
                )
            )
        ]

    return all_movies


async def main(data_type: str = "letterboxd"):
    # Connect to MongoDB client
    db_name, client = connect_to_db()
    tmdb_key = os.environ["TMDB_KEY"]

    db = client[db_name]
    movies = db.movies

    movies_for_update = get_ids_for_update(movies, data_type)

    chunk_size = 20
    num_chunks = (len(movies_for_update) + chunk_size - 1) // chunk_size  # ceil div

    print("Total Movies to Scrape:", len(movies_for_update))
    print("Total Chunks:", num_chunks)
    print("=======================\n")

    if num_chunks == 0:
        return

    pbar = tqdm(range(num_chunks))
    for chunk_i in pbar:
        pbar.set_description(f"Scraping chunk {chunk_i + 1} of {num_chunks}")

        start = chunk_i * chunk_size
        end = start + chunk_size
        chunk = movies_for_update[start:end]

        # up to 5 attempts per chunk
        for attempt in range(5):
            try:
                if data_type == "letterboxd":
                    await get_movies(chunk, movies, db)
                else:
                    await get_rich_data(chunk, movies, db, tmdb_key)
            except Exception as e:
                print(f"Error: {e}")
                print(traceback.format_exc())
                print(f"Error on attempt {attempt + 1}, retrying...")
                # short backoff to be nice to the site / network:
                await asyncio.sleep(1.0 + attempt * 0.5)
            else:
                break
        else:
            print(f"Could not complete requests for chunk {chunk_i + 1}")


if __name__ == "__main__":
    asyncio.run(main("letterboxd"))
    asyncio.run(main("tmdb"))
