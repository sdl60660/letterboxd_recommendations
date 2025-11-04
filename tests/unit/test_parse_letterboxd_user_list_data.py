import pymongo

import data_processing.get_users as get_users


def test_parse_user_list_page__item_count(html_sample_path):
    html = (html_sample_path / "sample_letterboxd_user_list_page.html").read_text()
    out = get_users.parse_user_list_page(html)

    assert len(out) == 30


def test_parse_user_list_page__content(html_sample_path):
    html = (html_sample_path / "sample_letterboxd_user_list_page.html").read_text()
    out = get_users.parse_user_list_page(html)

    assert all([type(x) is dict for x in out])

    # this is the first user in the sample DOM data I pulled down for testing
    assert out[0]["username"] == "schaffrillas"


def test_form_user_ops(html_sample_path):
    html = (html_sample_path / "sample_letterboxd_user_list_page.html").read_text()
    user_data = get_users.parse_user_list_page(html)

    out = [get_users.form_user_upsert_op(user) for user in user_data]
    assert all([type(x) is pymongo.operations.UpdateOne for x in out])
