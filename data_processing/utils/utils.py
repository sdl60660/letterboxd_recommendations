import os

import pandas as pd
import requests
from bs4 import BeautifulSoup

if os.getcwd().endswith("data_processing"):
    from utils.db_connect import connect_to_db
    from utils.http_utils import BROWSER_HEADERS
    from utils.selectors import LBX_USER_RATINGS_PAGE_LINKS
else:
    from data_processing.utils.db_connect import connect_to_db
    from data_processing.utils.http_utils import BROWSER_HEADERS
    from data_processing.utils.selectors import LBX_USER_RATINGS_PAGE_LINKS


def format_seconds(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)


def get_backoff_days(fail_count, max_days=180):
    return min(2 ** min(fail_count, 5), max_days)


def get_rich_movie_data(movie_ids, output_path=None):
    db_name, client = connect_to_db()
    db = client[db_name]

    movie_fields = {
        "movie_id",
        "image_url",
        "movie_title",
        "year_released",
        "genres",
        "original_language",
        "popularity",
        "runtime",
        "release_date",
        "content_type",
    }

    projection = {k: 1 for k in movie_fields} | {"_id": 0}
    movie_docs = db.movies.find(
        {"movie_id": {"$in": list(movie_ids)}}, projection=projection
    )
    movie_data = {d["movie_id"]: d for d in movie_docs}

    # if data file output path is provided, cache it as a parquet file at that path
    if output_path:
        pd.DataFrame(movie_data.values()).to_parquet(output_path, index=False)

    return movie_data


def get_page_count(username, url="https://letterboxd.com/{}/films/by/date"):
    r = requests.get(url.format(username), headers=BROWSER_HEADERS)

    soup = BeautifulSoup(r.text, "lxml")
    body = soup.find("body")

    try:
        if "error" in body["class"]:
            return -1, None
    except KeyError:
        print(body)
        return -1, None

    try:
        links = soup.select(LBX_USER_RATINGS_PAGE_LINKS)
        if links:
            num_pages = int(links[-1].get_text(strip=True).replace(",", ""))
        else:
            num_pages = 1

        header = soup.select_one("section.profile-header h1.title-3")
        display_name = header.get_text(strip=True) if header else None
    except Exception:
        num_pages = 1
        display_name = None

    return num_pages, display_name


# There are some pseudo-TV shows (docs about shows, etc) that people use to log a rating for a TV show
# which I think we want to exclude. For now, this is just a little manual/explicit with the most prominent
# ones, but can maybe find a better way to target later
# (Think I'm going to work on turning this into a default-on frontend filter,
# modeled after the exclusions here: https://letterboxd.com/dave/list/official-top-250-narrative-feature-films/)
explicit_exclude_list = [
    "no-half-measures-creating-the-final-season-of-breaking-bad",
    "twin-peaks",
    "fullmetal-alchemist-brotherhood",
    "monster-2004",
    "cowboy-bebop",
    "one-piece-fan-letter",
    "avatar-spirits",
    "twin-peaks-the-return",
    "attack-on-titan-the-last-attack",
    "attack-on-titan-the-final-chapters-special-2",
    "attack-on-titan-the-final-chapters-special-1-2023",
    "attack-on-titan-chronicle",
    "frieren-beyond-journeys-end",
    "tapping-the-wire",
    "andor-a-disney-day-special-look",
    "one-crazy-summer-a-look-back-at-gravity-falls",
    "adventure-time",
    "bojack-horseman-christmas-special-sabrinas",
    "euphoria-trouble-dont-last-always",
    "game-of-thrones-the-imax-experience",
    "making-true-detective",
    "hannibal-this-is-my-design",
    "paint-drying",
    "over-the-garden-wall-10th-anniversary-tribute",
    "the-owl-house-thanks-to-them",
    "its-always-sunny-in-philadelphia-sunny-side-up",
    "the-office-retrospective",
    "the-grand-tour-one-for-the-road",
    "sherlock-the-reichenbach-fall",
    "bluey-the-sign-2024",
    "grounded-making-the-last-of-us",
]
