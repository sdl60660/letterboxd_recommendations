import data_processing.get_movies as get_movies


async def _run_once(mongo_db, http_mock):
    url = "https://letterboxd.com/film/the-matrix/"
    # return a small deterministic HTML page
    http_mock.get(
        url,
        status=200,
        body="""
        <section class="production-masthead">
            <h1>The Matrix</h1>
            <span class="releasedate"><a>1999</a></span>
        </section>
        <a data-track-action="IMDb" href="https://www.imdb.com/title/tt0133093/"></a>
        <a data-track-action="TMDB" href="https://www.themoviedb.org/movie/603"></a>
        <script type="application/ld+json">
            {"image":"https://a.ltrbxd.com/resized/path/to/poster.jpg"}
        </script>
        """,
    )

    # call your async function
    await get_movies.get_movies(["the-matrix"], None, mongo_db)


def test_get_movies_inserts_document(mongo_db, http_mock, event_loop):
    event_loop.run_until_complete(_run_once(mongo_db, http_mock))
    doc = mongo_db.movies.find_one({"movie_id": "the-matrix"})
    assert doc is not None
    assert doc["movie_title"] == "The Matrix"
    assert doc["imdb_id"] == "tt0133093"
    assert doc["tmdb_id"] == "603"
