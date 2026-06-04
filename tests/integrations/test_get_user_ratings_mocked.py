from types import SimpleNamespace

import data_processing.get_users as get_users


def _sample_html(html_sample_path):
    return (html_sample_path / "sample_letterboxd_user_list_page.html").read_text()


def _mock_cffi_get(monkeypatch, expected_url, html):
    def fake_cffi_get(url):
        assert url == expected_url
        return SimpleNamespace(text=html)

    monkeypatch.setattr(get_users, "cffi_get", fake_cffi_get)


def test_get_users_inserts_documents(mongo_db, html_sample_path, monkeypatch):
    base_url = "https://letterboxd.com/members/popular/this/week/page/{}/"
    html = _sample_html(html_sample_path)

    _mock_cffi_get(monkeypatch, base_url.format(1), html)

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


def test_get_users_idempotent(mongo_db, html_sample_path, monkeypatch):
    base_url = "https://letterboxd.com/members/popular/this/week/page/{}/"
    html = _sample_html(html_sample_path)

    _mock_cffi_get(monkeypatch, base_url.format(2), html)
    get_users.process_user_page(base_url, 2, mongo_db.users, send_to_db=True)
    first_count = mongo_db.users.count_documents({})

    get_users.process_user_page(base_url, 2, mongo_db.users, send_to_db=True)
    second_count = mongo_db.users.count_documents({})

    # Upsert behavior: document count should not increase
    assert first_count == 30
    assert second_count == 30
