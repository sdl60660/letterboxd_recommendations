import pymongo

import data_processing.get_ratings as get_ratings


def test_parse_num_pages(html_sample_path):
    html = (html_sample_path / "sample_letterboxd_user_ratings_page.html").read_text()
    out = get_ratings.parse_num_pages(html)

    assert out == 2


def test_parse_user_ratings_to_raw_ratings_full(html_sample_path):
    html = (html_sample_path / "sample_letterboxd_user_ratings_page.html").read_text()
    out = get_ratings.generate_ratings_operations(
        [html, {"username": "samtestacct"}],
        send_to_db=False,
        return_unrated=True,
        attach_liked_flag=False,
    )

    assert len(out[0]) == 72


def test_parse_user_ratings_to_raw_ratings_rated(html_sample_path):
    html = (html_sample_path / "sample_letterboxd_user_ratings_page.html").read_text()
    out = get_ratings.generate_ratings_operations(
        [html, {"username": "samtestacct"}],
        send_to_db=False,
        return_unrated=False,
        attach_liked_flag=False,
    )

    assert len(out[0]) == 63
    assert "liked" not in out[0][0].keys()


def test_parse_user_ratings_to_raw_ratings_liked(html_sample_path):
    html = (html_sample_path / "sample_letterboxd_user_ratings_page.html").read_text()
    out = get_ratings.generate_ratings_operations(
        [html, {"username": "samtestacct"}],
        send_to_db=False,
        return_unrated=False,
        attach_liked_flag=True,
    )

    assert len(out[0]) == 63
    assert "liked" in out[0][0].keys()
    assert out[0][0]["liked"] is False


def test_parse_user_ratings_to_ops(html_sample_path):
    html = (html_sample_path / "sample_letterboxd_user_ratings_page.html").read_text()
    out = get_ratings.generate_ratings_operations(
        [html, {"username": "samtestacct"}],
        send_to_db=True,
        return_unrated=False,
        attach_liked_flag=True,
    )

    assert len(out[0]) == 63
    assert len(out[1]) == 63

    assert all([type(x) is pymongo.operations.UpdateOne for x in out[0]])
    assert all([type(x) is pymongo.operations.UpdateOne for x in out[1]])
