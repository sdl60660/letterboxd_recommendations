import re

import pytest
import requests
from bs4 import BeautifulSoup

import data_processing.get_movies as get_movies
import data_processing.get_ratings as get_ratings
from data_processing.utils.http_utils import BROWSER_HEADERS, default_request_timeout
from data_processing.utils.selectors import (
    LBX_JSON_LD_SCRIPT,
    LBX_MOVIE_HEADER,
)

pytestmark = [pytest.mark.smoke, pytest.mark.live]

BASE = "https://letterboxd.com"
FILM = f"{BASE}/film"


def _fetch(url: str) -> str:
    r = requests.get(url, headers=BROWSER_HEADERS, timeout=default_request_timeout)
    r.raise_for_status()
    return r.text


@pytest.fixture(scope="module")
def letterboxd_available():
    """Check if Letterboxd is up; skip all dependent tests if not."""
    try:
        r = requests.get(
            "https://letterboxd.com/",
            headers=BROWSER_HEADERS,
            timeout=default_request_timeout,
            allow_redirects=True,
        )
        # 503 = site down, or network issue
        if r.status_code == 503:
            pytest.skip("Letterboxd appears to be down (503).")
    except requests.exceptions.RequestException:
        pytest.skip("Cannot reach Letterboxd.com (network or DNS error).")
    return True


def test_zone_of_interest_structure_and_script_tag_meta():
    """
    Sanity-check a stable film page:
        - masthead exists
        - JSON-LD script exists and parses
        - get_meta_data_from_script_tag returns expected keys
    """
    slug = "the-zone-of-interest"
    html = _fetch(f"{FILM}/{slug}/")
    soup = BeautifulSoup(html, "lxml")

    # structure present
    header = soup.find(*LBX_MOVIE_HEADER)
    assert header is not None

    # JSON-LD <script type="application/ld+json"> exists
    json_ld = soup.find(*LBX_JSON_LD_SCRIPT)
    assert json_ld and json_ld.string and json_ld.string.strip()

    # our parser uses that JSON-LD
    meta = get_movies.get_meta_data_from_script_tag(soup)
    assert isinstance(meta, dict)
    # required keys
    for k in (
        "image_url",
        "letterboxd_rating_count",
        "letterboxd_avg_rating",
        "letterboxd_genres",
    ):
        assert k in meta

    # image_url should be non-empty
    assert isinstance(meta["image_url"], str) and meta["image_url"]

    # rating_count/avg_rating should be number-like if present
    if meta["letterboxd_rating_count"] is not None:
        assert isinstance(meta["letterboxd_rating_count"], (int, float))
    if meta["letterboxd_avg_rating"] is not None:
        assert isinstance(meta["letterboxd_avg_rating"], (int, float))

    # full page parse still works
    out = get_movies.parse_letterboxd_page_data(html, movie_id=slug)
    assert out["movie_id"] == slug
    assert re.search(r"zone\s+of\s+interest", out["movie_title"], flags=re.I)

    # poster stub is always present (string, maybe "")
    assert "image_url" in out


def test_redirects_resolve_to_expected_slug():
    """Old LOTR slug should redirect to Fellowship canonical slug."""
    redir_slug = "the-lord-of-the-rings-2003"
    r = requests.get(
        f"{FILM}/{redir_slug}/",
        headers=BROWSER_HEADERS,
        timeout=default_request_timeout,
        allow_redirects=True,
    )
    final_url = r.url
    assert "the-lord-of-the-rings-the-fellowship-of-the-ring" in final_url

    final_slug = get_movies.extract_movie_id_from_url(final_url)
    assert final_slug == "the-lord-of-the-rings-the-fellowship-of-the-ring"


def test_dummy_slug_404s_cleanly():
    """A fabricated slug should 404 (change slug if this ever becomes real)."""
    slug = "dummy-dummy-dummy-0123456-test"
    r = requests.get(
        f"{FILM}/{slug}/", headers=BROWSER_HEADERS, timeout=default_request_timeout
    )
    assert r.status_code == 404


def test_testuser_ratings_page_is_stable_enough():
    """
    Parse the live first page of ratings for 'samtestacct' and assert a stable shape.

    We assert:
        - parser returns only rated items (return_unrated=False)
        - length equals known count for page 1 (expected: 63)
        - every record has movie_id and rating_val > 0
        - liked flag is present & boolean when requested
    If this fails, it likely indicates a structure change on ratings tiles.
    """
    username = "samtestacct"
    url = f"{BASE}/{username}/films/by/date/"
    html = _fetch(url)

    # generate_ratings_operations is synchronous now
    ratings_ops, _movie_ops = get_ratings.generate_ratings_operations(
        [html, {"username": username}],
        send_to_db=False,
        return_unrated=False,
        attach_liked_flag=True,
    )

    # This is the count from the static fixture and is the expected count on here moving forward (I won't update this user page)
    # If this fails, it's because something has changed about the structure of the page/default filtering/etc, which might not be
    # particularly significant, but is actually worth getting a failed test on
    expected_count_page1 = 62
    assert len(ratings_ops) == expected_count_page1, (
        f"Expected {expected_count_page1} rated tiles on page 1 for {username}, "
        f"got {len(ratings_ops)} — page structure may have changed."
    )

    for rec in ratings_ops:
        assert isinstance(rec.get("movie_id"), str) and rec["movie_id"]
        assert isinstance(rec.get("rating_val"), int) and rec["rating_val"] > 0
        assert "liked" in rec and isinstance(rec["liked"], bool)


@pytest.mark.asyncio
async def test_redirect_path_marks_movie_and_redirects_collection(mongo_db):
    """
    Hit a known redirecting slug via the real site and ensure:
        - movies doc (by old_id) gets redirect_to + redirect_seen_at
        - movie_redirects has (old_id -> new_id) with status 'pending'
    """
    old_slug = "the-lord-of-the-rings-2003"
    await get_movies.get_movies([old_slug], None, mongo_db)

    # movies collection updated with redirect metadata
    movies = mongo_db.movies
    doc = movies.find_one({"movie_id": old_slug})
    assert doc is not None
    assert (
        "redirect_to" in doc
        and isinstance(doc["redirect_to"], str)
        and doc["redirect_to"]
    )
    assert "redirect_seen_at" in doc

    # redirect record created
    redirects = mongo_db.movie_redirects
    rdoc = redirects.find_one({"old_id": old_slug})
    assert rdoc is not None
    assert rdoc["old_id"] == old_slug
    assert isinstance(rdoc.get("new_id"), str) and rdoc["new_id"]
    assert rdoc.get("status") == "pending"
    assert "last_seen_at" in rdoc


@pytest.mark.asyncio
async def test_404_path_sets_failed_and_increments_fail_count(mongo_db):
    """
    Use a dummy slug that should return 404; ensure:
        - movies doc has scrape_status='failed'
        - fail_count was incremented to 1
        - next_retry_at is set (datetime-like)
    """
    slug = "dummy-dummy-dummy-0123456-test"
    await get_movies.get_movies([slug], None, mongo_db)

    movies = mongo_db.movies
    doc = movies.find_one({"movie_id": slug})
    assert doc is not None
    assert doc.get("scrape_status") == "failed"
    assert doc.get("fail_count") == 1
    assert "next_retry_at" in doc
