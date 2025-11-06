from typing import Iterable, List
from pymongo import DeleteMany, DeleteOne, InsertOne, ReplaceOne, UpdateOne
from pymongo.errors import (
    BulkWriteError,
    AutoReconnect,
    NetworkTimeout,
    WriteConcernError,
)
from tqdm.auto import tqdm
import time, random


def _is_mongomock_collection(coll) -> bool:
    return coll.__class__.__module__.startswith("mongomock")


_TRANSIENT_CODES = {
    11600,
    11602,
    91,
}  # Interrupted, InterruptedDueToReplStateChange, ShutdownInProgress


def run_agg_with_retries(
    coll, pipeline, *, allowDiskUse=True, comment=None, max_retries=5, base_sleep=1.0
):
    """Run aggregate with retries for transient repl/election errors."""
    attempt = 0
    while True:
        try:
            return list(
                coll.aggregate(pipeline, allowDiskUse=allowDiskUse, comment=comment)
            )
        except WriteConcernError as e:
            if getattr(e, "code", None) in _TRANSIENT_CODES and attempt < max_retries:
                attempt += 1
                delay = base_sleep * (2 ** (attempt - 1)) + random.random() * 0.25
                print(
                    f"[agg-retry] WriteConcernError {e.code}; retry {attempt}/{max_retries} in {delay:.2f}s"
                )
                time.sleep(delay)
                continue
            raise
        except (AutoReconnect, NetworkTimeout) as e:
            if attempt < max_retries:
                attempt += 1
                delay = base_sleep * (2 ** (attempt - 1)) + random.random() * 0.25
                print(
                    f"[agg-retry] {type(e).__name__}; retry {attempt}/{max_retries} in {delay:.2f}s"
                )
                time.sleep(delay)
                continue
            raise


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


def _chunked(seq: List, size: int) -> Iterable[List]:
    """Yield successive chunks from a list."""
    if size <= 0:
        raise ValueError("chunk size must be > 0")
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def safe_commit_ops_chunked(
    collection,
    upsert_operations: List,
    batch_size: int = 2000,
    desc: str = "Committing ops",
) -> int:
    """
    Write update ops to a target collection in batches with a tqdm progress bar.
    Uses bulk_write_compat for mongomock compatibility. Returns the number of
    operations attempted (sum of chunk sizes).
    """
    n_ops = len(upsert_operations) if upsert_operations else 0
    if n_ops == 0:
        return 0

    committed = 0
    with tqdm(total=n_ops, desc=desc, unit="op") as pbar:
        for chunk in _chunked(upsert_operations, batch_size):
            try:
                # ordered=False lets MongoDB continue past individual errors
                bulk_write_compat(collection, chunk, ordered=False)
            except BulkWriteError as bwe:
                # Log and carry on; with ordered=False many ops still apply
                print(f"[WARN] Bulk write error in {collection.name}: {bwe.details}")
            finally:
                committed += len(chunk)
                pbar.update(len(chunk))

    return committed
