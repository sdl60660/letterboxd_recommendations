# tests/integrations/test_db_safety_quick.py
from mongomock import MongoClient
from pymongo import UpdateOne

from data_processing.utils.mongo_utils import safe_commit_ops


def test_upsert_is_idempotent():
    client = MongoClient()
    db = client["_test"]
    coll = db["ratings"]

    doc = {"user_id": "u", "movie_id": "m", "rating_val": 7}
    op = UpdateOne({"user_id": "u", "movie_id": "m"}, {"$set": doc}, upsert=True)

    # first write
    safe_commit_ops(coll, [op])
    assert coll.count_documents({}) == 1
    first = coll.find_one({"user_id": "u", "movie_id": "m"})
    assert first["rating_val"] == 7

    # repeat the same write; count should remain 1, data stable
    safe_commit_ops(coll, [op])
    assert coll.count_documents({}) == 1
    second = coll.find_one({"user_id": "u", "movie_id": "m"})
    assert second["rating_val"] == 7


def test_bulk_write_not_called_for_empty_batches(monkeypatch):
    client = MongoClient()
    db = client["_test"]
    coll = db["movies"]

    called = {"bulk_write": False}

    def _bw(ops, **kwargs):
        called["bulk_write"] = True
        return None

    # On mongomock, bulk_write does exist; we verify our helper does nothing for []
    monkeypatch.setattr(coll, "bulk_write", _bw)
    safe_commit_ops(coll, [])
    assert called["bulk_write"] is False
