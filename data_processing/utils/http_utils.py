from curl_cffi import requests as cffi_requests

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

default_request_timeout = 20

IMPERSONATE = "chrome"


def cffi_get(url, **kwargs):
    """Sync GET with browser TLS fingerprint."""
    return cffi_requests.get(url, impersonate=IMPERSONATE, timeout=default_request_timeout, **kwargs)


def cffi_async_session(**kwargs):
    """Async session with browser TLS fingerprint."""
    return cffi_requests.AsyncSession(impersonate=IMPERSONATE, **kwargs)
