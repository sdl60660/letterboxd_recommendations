import data_processing.get_movies as get_movies


def test_parse_basic_card(html_sample_path):
    html = (html_sample_path / "sample_letterboxd_film.html").read_text()
    out = get_movies.parse_letterboxd_page_data(html, movie_id="test-slug")

    assert out["movie_id"] == "test-slug"
    assert isinstance(out["movie_title"], str)

    # robust fallbacks:
    assert "image_url" in out  # set as "" if missing poster
    assert "tmdb_id" in out
    assert out["scrape_status"] == "ok"


def test_parse_missing_bits(html_sample_path):
    html = (html_sample_path / "sample_letterboxd_film_missing_data.html").read_text()
    out = get_movies.parse_letterboxd_page_data(html, movie_id="test-slug")
    assert out["movie_title"] is not None
    assert out["image_url"] == ""
    assert "letterboxd_genres" not in out.keys()
