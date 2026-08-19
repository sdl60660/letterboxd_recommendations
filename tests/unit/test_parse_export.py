import pytest

from data_processing import parse_export

# These titles/years are chosen to exist in the rich movie data that ships with
# the repo, so they resolve to real slugs. The "boxd.it" URIs are synthetic --
# resolution is by Name + Year, not by the short link.
RATINGS_CSV = (
    "Date,Name,Year,Letterboxd URI,Rating\n"
    "2021-01-20,Parasite,2019,https://boxd.it/aaaa,5\n"
    "2021-01-26,Inception,2010,https://boxd.it/bbbb,4\n"
    "2021-01-26,12 Angry Men,1957,https://boxd.it/cccc,4.5\n"
    "2021-01-26,A Film That Is Not In The Dataset,2099,https://boxd.it/dddd,3\n"
)

WATCHED_CSV = (
    "Date,Name,Year,Letterboxd URI\n"
    "2021-01-20,Parasite,2019,https://boxd.it/aaaa\n"
    "2021-01-20,Inception,2010,https://boxd.it/bbbb\n"
    "2021-01-20,12 Angry Men,1957,https://boxd.it/cccc\n"
    "2021-01-20,The Grand Budapest Hotel,2014,https://boxd.it/eeee\n"
)

WATCHLIST_CSV = (
    "Date,Name,Year,Letterboxd URI\n"
    "2021-01-21,Vertigo,1958,https://boxd.it/ffff\n"
    "2021-01-21,Parasite,2019,https://boxd.it/aaaa\n"
)

LIKES_CSV = (
    "Date,Name,Year,Letterboxd URI\n"
    "2021-01-20,The Grand Budapest Hotel,2014,https://boxd.it/eeee\n"
)

PROFILE_CSV = (
    "Date Joined,Username,Given Name,Family Name,Email Address,Location,"
    "Website,Bio,Pronoun,Favorite Films\n"
    "2021-01-20,TestUser,Test,Person,,,,,,\n"
)


def _full_export_files():
    return [
        {"name": "ratings.csv", "text": RATINGS_CSV},
        {"name": "watched.csv", "text": WATCHED_CSV},
        {"name": "watchlist.csv", "text": WATCHLIST_CSV},
        {"name": "likes/films.csv", "text": LIKES_CSV},
        {"name": "profile.csv", "text": PROFILE_CSV},
    ]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_normalize_title():
    assert parse_export.normalize_title("The Grand Budapest Hotel") == (
        "the grand budapest hotel"
    )
    assert parse_export.normalize_title("WALL·E") == "wall e"
    assert parse_export.normalize_title("Amélie") == "amelie"
    assert parse_export.normalize_title("12 Angry Men!") == "12 angry men"
    assert parse_export.normalize_title(None) == ""  # no "none" false-match


@pytest.mark.parametrize(
    "stars,expected",
    [
        ("5", 10),
        ("4.5", 9),
        ("0.5", 1),
        (3, 6),
        ("0", None),
        ("", None),
        (None, None),
        ("not-a-number", None),
    ],
)
def test_stars_to_rating_val(stars, expected):
    assert parse_export._stars_to_rating_val(stars) == expected


def test_classify_csv_by_filename():
    assert parse_export.classify_csv("watchlist.csv", []) == "watchlist"
    assert parse_export.classify_csv("my-ratings.csv", []) == "ratings"
    assert parse_export.classify_csv("likes/films.csv", []) == "likes"


def test_classify_csv_by_columns():
    rating_rows = [{"Name": "X", "Year": "2000", "Rating": "4"}]
    assert parse_export.classify_csv("export.csv", rating_rows) == "ratings"

    diary_rows = [{"Name": "X", "Rating": "4", "Watched Date": "2020-01-01"}]
    assert parse_export.classify_csv("export.csv", diary_rows) == "diary"

    watched_rows = [{"Name": "X", "Year": "2000"}]
    assert parse_export.classify_csv("export.csv", watched_rows) == "watched"


# ---------------------------------------------------------------------------
# Resolution (depends on shipped rich movie data)
# ---------------------------------------------------------------------------


def test_resolve_movie_id_known_and_unknown():
    assert parse_export.resolve_movie_id("Parasite", "2019") == "parasite-2019"
    assert parse_export.resolve_movie_id("Inception", "2010") == "inception"
    assert parse_export.resolve_movie_id("12 Angry Men", "1957") == "12-angry-men"
    assert (
        parse_export.resolve_movie_id("A Film That Is Not In The Dataset", "2099")
        is None
    )


# ---------------------------------------------------------------------------
# Sorting uploaded files into categories
# ---------------------------------------------------------------------------


def test_build_categories_from_files_uses_name_and_columns():
    cats = parse_export.build_categories_from_files(
        [
            {"name": "likes/films.csv", "text": LIKES_CSV},
            {"name": "renamed.csv", "text": WATCHLIST_CSV},  # classified by columns
            {"name": "ignored", "text": ""},
            {"name": "bad.csv", "text": 123},  # non-string text is skipped, not crashed
            "not-a-dict",
        ]
    )
    assert len(cats["likes"]) == 1
    assert len(cats["watched"]) == 2  # WATCHLIST_CSV has no Rating column


# ---------------------------------------------------------------------------
# Building user data
# ---------------------------------------------------------------------------


def test_build_user_data_from_full_export():
    result = parse_export.build_user_data_from_files(_full_export_files())

    assert result["status"] == "success"
    assert result["username"] == "testuser"
    assert result["display_name"] == "Test Person"

    by_id = {r["movie_id"]: r for r in result["user_ratings"]}

    # The unknown film is dropped; the 4 resolvable watched/rated films remain.
    assert set(by_id) == {
        "parasite-2019",
        "inception",
        "12-angry-men",
        "the-grand-budapest-hotel",
    }

    # 5 stars -> 10, 4.5 stars -> 9 on the model's 1-10 scale.
    assert by_id["parasite-2019"]["rating_val"] == 10
    assert by_id["12-angry-men"]["rating_val"] == 9
    assert result["num_explicit_ratings"] == 3

    # Watched-but-unrated film is kept (so it's excluded from recs) at -1.
    assert by_id["the-grand-budapest-hotel"]["rating_val"] == -1

    # The liked-but-unrated film gets a synthetic rating.
    assert by_id["the-grand-budapest-hotel"]["liked"] is True
    assert "synthetic_rating_val" in by_id["the-grand-budapest-hotel"]

    # Watchlist resolves and de-dupes.
    assert result["watchlist"] == ["vertigo", "parasite-2019"]

    # All entries are attributed to the profile username.
    assert all(r["user_id"] == "testuser" for r in result["user_ratings"])


def test_build_user_data_single_ratings_csv():
    result = parse_export.build_user_data_from_files(
        [{"name": "ratings.csv", "text": RATINGS_CSV}]
    )

    assert result["status"] == "success"
    assert result["username"] is None
    assert result["num_explicit_ratings"] == 3
    # No watched.csv, so "seen" is just the rated films.
    assert len(result["user_ratings"]) == 3
    assert result["watchlist"] == []
    assert all(r["user_id"] == "uploaded-user" for r in result["user_ratings"])


def test_build_user_data_single_watchlist_csv():
    result = parse_export.build_user_data_from_files(
        [{"name": "watchlist.csv", "text": WATCHLIST_CSV}]
    )

    assert result["status"] == "success"
    assert result["user_ratings"] == []
    assert result["watchlist"] == ["vertigo", "parasite-2019"]


def test_build_user_data_empty_is_no_data():
    result = parse_export.build_user_data_from_files([])
    assert result["status"] == "no_data"
    assert result["user_ratings"] == []
    assert result["watchlist"] == []
