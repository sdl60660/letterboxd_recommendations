from pymongo import DeleteMany, DeleteOne, InsertOne, ReplaceOne, UpdateOne
from pymongo.errors import BulkWriteError


def _is_mongomock_collection(coll) -> bool:
    return coll.__class__.__module__.startswith("mongomock")


def bulk_write_compat(coll, ops, **kwargs):
    """
    Use bulk_write on real Mongo. On mongomock, replay ops individually
    to avoid unsupported kwargs (sort/array_filters/collation).
    """
    if not ops:
        return

    if not _is_mongomock_collection(coll):
        # Real MongoDB
        return coll.bulk_write(ops, **kwargs)

    if _is_mongomock_collection(coll):
        for op in ops:
            if isinstance(op, UpdateOne):
                # Access UpdateOne internals (fine for test-only fallback)
                coll.update_one(op._filter, op._doc, upsert=op._upsert)
            elif isinstance(op, ReplaceOne):
                coll.replace_one(op._filter, op._replacement, upsert=op._upsert)
            elif isinstance(op, InsertOne):
                coll.insert_one(op._doc)
            elif isinstance(op, DeleteOne):
                coll.delete_one(op._filter)
            elif isinstance(op, DeleteMany):
                coll.delete_many(op._filter)
            else:
                # Add other op types if you use them (InsertOne, ReplaceOne, etc.)
                raise NotImplementedError(
                    f"Unsupported op in mongomock fallback: {type(op)}"
                )
        return


def safe_commit_ops(collection, upsert_operations):
    """Write a batch of update ops to a target collection (safe for mongomock)."""
    if not upsert_operations or len(upsert_operations) == 0:
        return 0

    try:
        # this is basically just a simple bulk_write() op with the upsert_operations (see below)
        # but for compatibility with the mongomock stuff in testing, I need to wrap it in this util
        bulk_write_compat(collection, upsert_operations, ordered=False)
        # collection.bulk_write(upsert_operations, ordered=False)
        return len(upsert_operations)
    except BulkWriteError as bwe:
        print(f"[WARN] Bulk write error in {collection.name}: {bwe.details}")
        return 0
