import data_processing.get_ratings as get_ratings


async def _run_once(mongo_db, http_mock, html):
    """
    Run one end-to-end pass that:
        - fetches a single ratings page for 'samtestacct'
        - parses it
        - writes ratings/movies/users into the provided (mongomock) DB
    """
    username = "samtestacct"
    # This is exactly the URL pattern used inside get_user_ratings()
    url = f"https://letterboxd.com/{username}/films/by/date/page/1/"

    # Serve our deterministic HTML for that page
    http_mock.get(url, status=200, body=html)

    # Tell the orchestrator to scrape exactly 1 page for our test user
    pages_by_user = {username: 1}

    # Kick off the pipeline. This will call:
    #   - get_user_ratings(..., num_pages=1)
    #   - generate_ratings_operations(...)
    #   - bulk write into db.ratings, db.movies, db.users
    await get_ratings.get_ratings(
        [username],
        pages_by_user,
        mongo_db=mongo_db,
        store_in_db=True,
    )


def test_get_ratings_inserts_docs(mongo_db, http_mock, html_sample_path, event_loop):
    # Load the sample page you already use in unit tests
    html = (html_sample_path / "sample_letterboxd_user_ratings_page.html").read_text()

    event_loop.run_until_complete(_run_once(mongo_db, http_mock, html))

    # --- Assertions ---
    # Ratings were inserted for our user
    ratings_count = mongo_db.ratings.count_documents({"user_id": "samtestacct"})

    # Your unit test says 63 rated items on this sample page
    assert ratings_count == 63

    # Movies skeleton upserts were also written
    # (one per rating; duplicates are unlikely on a single page)
    movies_count = mongo_db.movies.estimated_document_count()
    assert movies_count == 63

    # User scrape status should be updated to "ok"
    user_doc = mongo_db.users.find_one({"username": "samtestacct"})
    assert user_doc is not None
    assert user_doc.get("scrape_status") == "ok"
    # sanity: recent_page_count/num_ratings_pages are not set in this path (we bypassed get_page_counts)
