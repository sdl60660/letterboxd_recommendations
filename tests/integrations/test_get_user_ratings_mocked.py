import responses
import asyncio
from aioresponses import aioresponses

import data_processing.get_users as get_users


def _sample_html(html_sample_path):
    return (html_sample_path / "sample_letterboxd_user_list_page.html").read_text()


@responses.activate
def test_get_users_inserts_documents(mongo_db, html_sample_path):
    base_url = "https://letterboxd.com/members/popular/this/week/page/{}/"
    html = _sample_html(html_sample_path)
    users_coll = mongo_db.users

    with aioresponses() as m:
        m.get(base_url.format(1), status=200, body=html)

        # Run one-page async scrape
        asyncio.run(
            get_users.run_async_scrape(
                users_coll=users_coll,
                base_url=base_url,
                total_pages=1,
                concurrency=2,
                send_to_db=True,
            )
        )

    # DB assertions
    assert mongo_db.users.count_documents({}) == 30
    doc = mongo_db.users.find_one({"username": "schaffrillas"})
    assert doc is not None
    assert "display_name" in doc
    assert isinstance(doc["num_reviews"], int)


@responses.activate
def test_get_users_idempotent(mongo_db, html_sample_path):
    base_url = "https://letterboxd.com/members/popular/this/week/page/{}/"
    html = _sample_html(html_sample_path)
    users_coll = mongo_db.users

    # First run
    with aioresponses() as m:
        m.get(base_url.format(2), status=200, body=html)
        asyncio.run(
            get_users.run_async_scrape(
                users_coll=users_coll,
                base_url=base_url,
                total_pages=1,  # we’ll use page=2 by formatting below
                concurrency=2,
                send_to_db=True,
            )
        )
    first_count = mongo_db.users.count_documents({})

    # Second run (same page & body) — add another mock call
    with aioresponses() as m:
        m.get(base_url.format(2), status=200, body=html)
        asyncio.run(
            get_users.run_async_scrape(
                users_coll=users_coll,
                base_url=base_url,
                total_pages=1,
                concurrency=2,
                send_to_db=True,
            )
        )
    second_count = mongo_db.users.count_documents({})

    assert first_count == 30
    assert second_count == 30
