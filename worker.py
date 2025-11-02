import os
import sys

import redis
from rq import Queue, Worker

listen = ["high", "default", "low"]

redis_url = (
    os.getenv("REDISCLOUD_URL")
    or os.getenv("REDIS_URL")  # Heroku sets REDIS_URL by default
    or "redis://localhost:6379"
)

conn = redis.from_url(redis_url)
queues = [Queue(x, connection=conn) for x in listen]

if __name__ == "__main__":
    # On macOS host runs, SimpleWorker avoids fork/ObjC issues.
    # In Docker/Linux (compose/Heroku), normal Worker is fine.
    if sys.platform == "darwin":
        from rq import SimpleWorker

        worker = SimpleWorker(queues, connection=conn)
    else:
        worker = Worker(queues, connection=conn)

    worker.work()

    # send_kill_horse_command(redis, worker.name)
