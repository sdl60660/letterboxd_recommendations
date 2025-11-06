#!/usr/local/bin/python3.12

import os
import re
import asyncio
from typing import List, Dict, Any

from aiohttp import ClientSession, TCPConnector
from bs4 import BeautifulSoup
from pymongo.operations import UpdateOne
from tqdm import tqdm

if os.getcwd().endswith("/data_processing"):
    from utils.db_connect import connect_to_db
    from utils.http_utils import BROWSER_HEADERS, default_request_timeout
    from utils.mongo_utils import safe_commit_ops
    from utils.selectors import LBX_USER_ROW, LBX_USER_TABLE
else:
    from data_processing.utils.db_connect import connect_to_db
    from data_processing.utils.http_utils import (
        BROWSER_HEADERS,
        default_request_timeout,
    )
    from data_processing.utils.mongo_utils import safe_commit_ops
    from data_processing.utils.selectors import LBX_USER_ROW, LBX_USER_TABLE


def parse_user_tile(user_item) -> Dict[str, Any]:
    link = user_item.find("a")["href"]
    username = link.strip("/").lower()
    display_name = user_item.find("a", class_="name").text.strip()
    reviews_link = user_item.select_one('small.metadata a[href$="/reviews/"]')

    txt = reviews_link.get_text(" ", strip=True) if reviews_link else ""
    m = re.search(r"([\d,]+)\s*reviews", txt, flags=re.I)
    num_reviews = int(m.group(1).replace(",", "")) if m else 0

    return {
        "username": username,
        "display_name": display_name,
        "num_reviews": num_reviews,
        # "last_updated": datetime.datetime.now(datetime.timezone.utc),
    }


def parse_user_list_page(html: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find(*LBX_USER_TABLE)
    if not table:
        return []
    table_items = table.find_all(*LBX_USER_ROW)

    user_data_list = []
    for user_tile in table_items:
        user = parse_user_tile(user_tile)
        user_data_list.append(user)

    return user_data_list


def form_user_upsert_op(record: Dict[str, Any]) -> UpdateOne:
    return UpdateOne({"username": record["username"]}, {"$set": record}, upsert=True)


async def _fetch(session: ClientSession, url: str) -> str:
    async with session.get(url, timeout=default_request_timeout) as r:
        r.raise_for_status()
        return await r.text()


async def _process_one_page(
    session: ClientSession,
    url: str,
    users_coll,
    send_to_db: bool = True,
) -> Dict[str, Any]:
    html = await _fetch(session, url)
    user_data = parse_user_list_page(html)
    ops = [form_user_upsert_op(u) for u in user_data]

    if send_to_db and ops:
        safe_commit_ops(users_coll, ops)

    return {"data": user_data, "ops": ops}


async def run_async_scrape(
    users_coll,
    base_url: str,
    total_pages: int = 128,
    concurrency: int = 6,
    send_to_db: bool = True,
) -> List[Dict[str, Any]]:
    """
    Scrape all user-list pages concurrently with a connection limit.
    Returns a list of per-page results: {"data": [...], "ops": [UpdateOne, ...]}
    """
    results: List[Dict[str, Any]] = []

    connector = TCPConnector(limit=concurrency)
    async with ClientSession(headers=BROWSER_HEADERS, connector=connector) as session:
        tasks = []
        for page in range(1, total_pages + 1):
            url = base_url.format(page)
            tasks.append(_process_one_page(session, url, users_coll, send_to_db))

        # progress bar over completions (not just submissions)
        out = []
        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"Scraping {total_pages} pages of top users",
        ):
            try:
                out.append(await coro)
            except Exception as e:
                # Don't crash whole run if one page fails; collect as empty result
                out.append({"data": [], "ops": [], "error": str(e)})

    results.extend(out)
    return results


def main():
    # Connect to MongoDB client
    db_name, client = connect_to_db()
    db = client[db_name]
    users = db.users

    base_url = "https://letterboxd.com/members/popular/this/week/page/{}/"

    TOTAL_PAGES = 128
    CONCURRENCY = 6

    asyncio.run(
        run_async_scrape(
            users_coll=users,
            base_url=base_url,
            total_pages=TOTAL_PAGES,
            concurrency=CONCURRENCY,
            send_to_db=True,
        )
    )


if __name__ == "__main__":
    main()
