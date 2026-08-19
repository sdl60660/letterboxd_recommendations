#!/usr/local/bin/python3.12

"""
Turn the CSVs from a Letterboxd data export into the same user_data structure
the scraper produces, so an upload can feed the recommendation model without
scraping. The export ZIP is unzipped in the browser; this module only ever sees
CSV text (a list of {name, text} files), never a ZIP.

The export only contains film Name/Year/short-link, not the slug (movie_id) the
model is keyed on, so Name + Year are resolved back to a slug using the rich
movie data shipped with the project. Unresolved films are dropped -- they
aren't in the model's universe, so they can't be folded in or recommended.
"""

import csv
import glob
import io
import os
import re
import unicodedata

import pandas as pd

if os.getcwd().endswith("data_processing"):
    from get_user_ratings import attach_synthetic_ratings
else:
    from data_processing.get_user_ratings import attach_synthetic_ratings


# Export file (by archive path) -> category. Used to label uploaded CSVs
_KNOWN_FILES = {
    "ratings.csv": "ratings",
    "watched.csv": "watched",
    "watchlist.csv": "watchlist",
    "diary.csv": "diary",
    "likes/films.csv": "likes",
    "profile.csv": "profile",
}

# Cached (exact_index, title_index) lookup, built lazily once per process
_MOVIE_INDEX_CACHE = None


def _rich_data_dir():
    if os.getcwd().endswith("data_processing"):
        return "data/rich_movie_data"
    return "data_processing/data/rich_movie_data"


def normalize_title(title):
    """Lowercase and strip accents/punctuation so titles compare cleanly."""
    if not title:  # guard None/"" so str(None) doesn't become "none"
        return ""
    s = unicodedata.normalize("NFKD", str(title))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    return s


def _coerce_year(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def build_movie_index(force=False):
    """Cache lookups from the rich movie data: (title, year) -> slug and
    title -> {slugs}, unioned across every shipped sample."""
    global _MOVIE_INDEX_CACHE
    if _MOVIE_INDEX_CACHE is not None and not force:
        return _MOVIE_INDEX_CACHE

    exact_index = {}
    title_index = {}
    for path in sorted(glob.glob(os.path.join(_rich_data_dir(), "*.parquet"))):
        try:
            df = pd.read_parquet(
                path, columns=["movie_id", "movie_title", "year_released"]
            )
        except Exception:
            continue
        for movie_id, title, year in zip(
            df["movie_id"], df["movie_title"], df["year_released"]
        ):
            if not isinstance(title, str) or not title or not isinstance(movie_id, str):
                continue
            norm = normalize_title(title)
            if not norm:
                continue
            year_int = _coerce_year(year)
            if year_int is not None:
                exact_index.setdefault((norm, year_int), movie_id)
            title_index.setdefault(norm, set()).add(movie_id)

    _MOVIE_INDEX_CACHE = (exact_index, title_index)
    return _MOVIE_INDEX_CACHE


def resolve_movie_id(name, year, index=None):
    """Resolve a film Name + Year to a Letterboxd slug, or None if not found."""
    exact_index, title_index = index or build_movie_index()
    norm = normalize_title(name)
    if not norm:
        return None

    year_int = _coerce_year(year)
    if year_int is not None:
        # Exact year first, then tolerate off-by-one (release vs listed year)
        for candidate in (year_int, year_int - 1, year_int + 1):
            slug = exact_index.get((norm, candidate))
            if slug:
                return slug

    candidates = title_index.get(norm)
    if candidates and len(candidates) == 1:
        return next(iter(candidates))
    return None


def _read_csv_rows(data):
    if isinstance(data, bytes):
        data = data.decode("utf-8-sig", errors="replace")
    elif data and data[0] == "\ufeff":  # strip a UTF-8 BOM if present
        data = data[1:]
    reader = csv.DictReader(io.StringIO(data))
    return [row for row in reader if any((v or "").strip() for v in row.values())]


def _has_columns(rows, *names):
    return bool(rows) and all(name in rows[0] for name in names)


def classify_csv(filename, rows):
    """Classify a CSV by filename, falling back to its columns."""
    name = (filename or "").lower()
    for key in ("watchlist", "diary", "ratings", "like", "watched"):
        if key in name:
            return "likes" if key == "like" else key
    if _has_columns(rows, "Rating"):
        return "diary" if _has_columns(rows, "Watched Date") else "ratings"
    return "watched"


def _stars_to_rating_val(raw):
    """Convert 0.5-5.0 export stars to the model's 1-10 scale (the scraper's rated-N)."""
    try:
        stars = float(raw)
    except (TypeError, ValueError):
        return None
    return int(round(stars * 2)) if stars > 0 else None


def build_categories_from_files(files):
    """Sort uploaded CSVs into category -> rows. `files` is a list of
    {name, text}; each file's category comes from its name (export path) or,
    failing that, its columns."""
    categories = {
        k: [] for k in ("ratings", "diary", "watchlist", "watched", "likes", "profile")
    }
    for f in files or []:
        if not isinstance(f, dict):
            continue
        text = f.get("text")
        if not isinstance(text, str) or not text:
            continue
        rows = _read_csv_rows(text)
        if not rows:
            continue
        key = (f.get("name") or "").lstrip("/")
        category = _KNOWN_FILES.get(key) or classify_csv(key, rows)
        if not categories[category]:  # first file of a category wins
            categories[category] = rows
    return categories


def _entry_key(row):
    uri = (row.get("Letterboxd URI") or "").strip()
    return uri or f"{normalize_title(row.get('Name'))}|{_coerce_year(row.get('Year'))}"


def build_user_data_from_files(files):
    """Turn uploaded export CSVs into {user_ratings, watchlist, username,
    display_name, num_explicit_ratings, status}, matching the scraper's
    user_data. `files` is a list of {name, text}."""
    categories = build_categories_from_files(files)
    index = build_movie_index()

    username = display_name = None
    if categories["profile"]:
        prof = categories["profile"][0]
        username = (prof.get("Username") or "").strip().lower() or None
        names = (
            (prof.get("Given Name") or "").strip(),
            (prof.get("Family Name") or "").strip(),
        )
        display_name = " ".join(p for p in names if p) or username

    # Merge watched/rated/liked into one record per film. Unrated films keep
    # rating_val -1 so they're still excluded from recs, like the scraper.
    merged = {}

    def _touch(row):
        return merged.setdefault(
            _entry_key(row),
            {
                "name": row.get("Name"),
                "year": row.get("Year"),
                "rating_val": -1,
                "liked": False,
            },
        )

    for row in categories["watched"]:
        _touch(row)
    for row in categories["ratings"] or categories["diary"]:  # ratings.csv preferred
        rec = _touch(row)
        rating_val = _stars_to_rating_val(row.get("Rating"))
        if rating_val is not None:
            rec["rating_val"] = rating_val
    for row in categories["likes"]:
        _touch(row)["liked"] = True

    user_id = username or "uploaded-user"

    # Resolve to slugs, dropping films not in the model; if two records resolve
    # to the same slug, keep any explicit rating and any like.
    user_ratings = {}
    for rec in merged.values():
        movie_id = resolve_movie_id(rec["name"], rec["year"], index)
        if not movie_id:
            continue
        existing = user_ratings.get(movie_id)
        if existing is None:
            user_ratings[movie_id] = {
                "movie_id": movie_id,
                "rating_val": rec["rating_val"],
                "user_id": user_id,
                "liked": rec["liked"],
            }
        else:
            if rec["rating_val"] >= 0:
                existing["rating_val"] = rec["rating_val"]
            existing["liked"] = existing["liked"] or rec["liked"]

    user_ratings_list = list(user_ratings.values())
    attach_synthetic_ratings(user_ratings_list, global_mean=8)

    watchlist = []
    seen = set()
    for row in categories["watchlist"]:
        movie_id = resolve_movie_id(row.get("Name"), row.get("Year"), index)
        if movie_id and movie_id not in seen:
            seen.add(movie_id)
            watchlist.append(movie_id)

    num_explicit = sum(1 for r in user_ratings_list if r["rating_val"] >= 0)
    return {
        "user_ratings": user_ratings_list,
        "watchlist": watchlist,
        "username": username,
        "display_name": display_name,
        "num_explicit_ratings": num_explicit,
        "status": "success" if user_ratings_list or watchlist else "no_data",
    }
