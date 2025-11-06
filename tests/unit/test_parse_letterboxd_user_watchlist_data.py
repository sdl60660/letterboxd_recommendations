import pytest

import data_processing.get_user_watchlist as get_user_watchlist
import data_processing.utils.utils as utils_mod
from data_processing.utils.http_utils import default_request_timeout


@pytest.mark.asyncio
async def test_parse_watchlist_page_extracts_ids(html_sample_path):
    """parse_watchlist_page should return a list of movie_id slugs from the watchlist grid."""
    html = (html_sample_path / "sample_letterboxd_user_watchlist_page.html").read_text()
    # parse_watchlist_page expects a tuple/list like (html_bytes_or_str, meta)
    out = await get_user_watchlist.parse_watchlist_page(
        (html, {"username": "samtestacct"})
    )

    # basic shape checks
    assert isinstance(out, list)
    assert all(isinstance(mid, str) for mid in out)

    assert len(out) == 20
    assert out[0].strip().lower() == "gozu"


@pytest.mark.asyncio
async def test_parse_watchlist_page_ignores_tiles_without_react_component():
    """Tiles missing the react-component should be skipped without crashing."""
    html = """
    <ul>
    <li class="griditem">
        <div class="react-component" data-item-slug="a-real-movie"></div>
    </li>
    <li class="griditem">
        <!-- no react-component here -->
        <div class="something-else"></div>
    </li>
    </ul>
    """
    out = await get_user_watchlist.parse_watchlist_page((html, {"username": "x"}))
    assert out == ["a-real-movie"]


class DummyResp:
    def __init__(self, text):
        self.text = text


def test_get_page_count_parses_pagination(monkeypatch, html_sample_path):
    """get_page_count should parse the pagination links and return an integer >= 1."""
    html = (html_sample_path / "sample_letterboxd_user_watchlist_page.html").read_text()

    # Monkeypatch requests.get to return our sample HTML
    monkeypatch.setattr(
        utils_mod.requests,
        "get",
        lambda url, headers=None, timeout=default_request_timeout: DummyResp(html),
    )

    n_pages, _ = utils_mod.get_page_count(
        "samtestacct", url="https://letterboxd.com/{}/watchlist"
    )
    assert isinstance(n_pages, int)
    assert n_pages >= 1


def test_get_page_count_handles_error_page(monkeypatch):
    """If the page looks like an error page, get_page_count should return -1."""
    error_html = '<html><body class="error">Not found</body></html>'

    monkeypatch.setattr(
        utils_mod.requests,
        "get",
        lambda url, headers=None, timeout=default_request_timeout: DummyResp(
            error_html
        ),
    )

    n_pages, _ = utils_mod.get_page_count(
        "nope-user", url="https://letterboxd.com/{}/watchlist"
    )

    assert n_pages == -1
