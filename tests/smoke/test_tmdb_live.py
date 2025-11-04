import os

import pytest
import requests
from aioresponses import aioresponses

import data_processing.get_movies as get_movies

pytestmark = [pytest.mark.smoke, pytest.mark.tmdb]

TMDB_KEY = os.getenv("TMDB_KEY")


@pytest.fixture(autouse=True, scope="module")
def _skip_if_no_key():
    if not TMDB_KEY:
        pytest.skip(
            "TMDB_KEY not set; skipping TMDB live smoke tests.", allow_module_level=True
        )


def test_tmdb_movie_detail_matrix():
    """
    Simple live hit to TMDB for a very stable title (The Matrix, id=603).
    Checks basic shape only.
    """
    url = f"https://api.themoviedb.org/3/movie/603?api_key={TMDB_KEY}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    j = r.json()

    assert j.get("id") == 603
    assert isinstance(j.get("title"), str)
    assert isinstance(j.get("genres", []), list)


def test_tmdb_tv_detail_got():
    """
    Another stable TMDB endpoint: Game of Thrones, id=1399.
    """
    url = f"https://api.themoviedb.org/3/tv/1399?api_key={TMDB_KEY}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    j = r.json()

    assert j.get("id") == 1399
    assert isinstance(j.get("name"), str)
    assert isinstance(j.get("genres", []), list)


@pytest.mark.asyncio
async def test_tmdb_rich_data_updates_movie_document(mongo_db):
    """
    Mock TMDB response and verify get_rich_data writes expected fields to movies.
    This does NOT require a real TMDB key; we stub the HTTP call.
    """
    # seed a movie doc with tmdb_id/content_type so get_rich_data will request it
    movie = {"movie_id": "the-matrix", "tmdb_id": "603", "content_type": "movie"}
    movies = mongo_db.movies
    movies.update_one({"movie_id": movie["movie_id"]}, {"$set": movie}, upsert=True)

    tmdb_key = "TEST_KEY"
    tmdb_url = f"https://api.themoviedb.org/3/movie/603?api_key={tmdb_key}"

    fake_json = {
        "genres": [{"id": 1, "name": "Action"}, {"id": 2, "name": "Sci-Fi"}],
        "production_countries": [{"iso_3166_1": "US", "name": "United States"}],
        "spoken_languages": [{"iso_639_1": "en", "name": "English"}],
        "popularity": 123.4,
        "overview": "A computer hacker learns the shocking truth...",
        "runtime": 136,
        "vote_average": 8.7,
        "vote_count": 25000,
        "release_date": "1999-03-31",
        "original_language": "en",
    }

    with aioresponses() as m:
        m.get(tmdb_url, payload=fake_json, status=200)

        # get_rich_data expects each item to have keys it will reuse/modify
        await get_movies.get_rich_data(
            movie_list=[movie], db_cursor=None, mongo_db=mongo_db, tmdb_key=tmdb_key
        )

    out = movies.find_one({"movie_id": "the-matrix"})
    assert out is not None

    # object fields are lists of names
    assert out["genres"] == ["Action", "Sci-Fi"]
    assert out["production_countries"] == ["United States"]
    assert out["spoken_languages"] == ["English"]

    # simple numeric/text fields
    assert out["popularity"] == fake_json["popularity"]
    assert out["overview"] == fake_json["overview"]
    assert out["runtime"] == fake_json["runtime"]
    assert out["vote_average"] == fake_json["vote_average"]
    assert out["vote_count"] == fake_json["vote_count"]
    assert out["release_date"] == fake_json["release_date"]
    assert out["original_language"] == fake_json["original_language"]

    # timestamp added
    assert "last_updated" in out
