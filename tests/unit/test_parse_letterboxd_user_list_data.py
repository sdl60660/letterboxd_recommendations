import pymongo

import data_processing.get_users as get_users


def test_parse_user_list_page__item_count(html_sample_path):
    html = (html_sample_path / "sample_letterboxd_user_list_page.html").read_text()
    out = get_users.parse_user_list_page(html)

    assert len(out) == 30


def test_parse_user_list_page__content(html_sample_path):
    html = (html_sample_path / "sample_letterboxd_user_list_page.html").read_text()
    out = get_users.parse_user_list_page(html)

    assert all([type(x) is pymongo.operations.UpdateOne for x in out])
    assert out[0]._doc["$set"]["username"] == "schaffrillas"
    # assert all([type(x) is pymongo.operations.UpdateOne for x in out])
