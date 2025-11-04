import responses

import data_processing.get_users as get_users


def _sample_html(html_sample_path):
    return (html_sample_path / "sample_letterboxd_user_list_page.html").read_text()


@responses.activate
def test_get_users_inserts_documents(mongo_db, html_sample_path):
    base_url = "https://letterboxd.com/members/popular/this/week/page/{}/"
    html = _sample_html(html_sample_path)

    # Mock the exact GET URL your code calls
    responses.add(
        responses.GET,
        base_url.format(1),
        body=html,
        status=200,
        content_type="text/html",
    )

    # Run the page processor, inserting into mongomock
    result = get_users.process_user_page(base_url, 1, mongo_db.users, send_to_db=True)

    # It returns both data and ops for inspection
    assert "data" in result and "ops" in result
    assert len(result["data"]) == 30
    assert len(result["ops"]) == 30

    # DB assertions
    assert mongo_db.users.count_documents({}) == 30
    # this is the first user in the sample DOM data I pulled down for testing
    doc = mongo_db.users.find_one({"username": "schaffrillas"})
    assert doc is not None
    assert "display_name" in doc
    assert isinstance(doc["num_reviews"], int)


@responses.activate
def test_get_users_idempotent(mongo_db, html_sample_path):
    base_url = "https://letterboxd.com/members/popular/this/week/page/{}/"
    html = _sample_html(html_sample_path)

    # First run
    responses.add(
        responses.GET,
        base_url.format(2),
        body=html,
        status=200,
        content_type="text/html",
    )
    get_users.process_user_page(base_url, 2, mongo_db.users, send_to_db=True)
    first_count = mongo_db.users.count_documents({})

    # Second run (same response)
    responses.add(
        responses.GET,
        base_url.format(2),
        body=html,
        status=200,
        content_type="text/html",
    )
    get_users.process_user_page(base_url, 2, mongo_db.users, send_to_db=True)
    second_count = mongo_db.users.count_documents({})

    # Upsert behavior: document count should not increase
    assert first_count == 30
    assert second_count == 30
