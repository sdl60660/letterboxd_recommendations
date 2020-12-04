import os

import redis
from redis import Redis
from rq import Worker, Queue, Connection
from rq.command import send_kill_horse_command

# from autoworker import AutoWorker


listen = ['high', 'default', 'low']
redis_url = os.getenv('REDISTOGO_URL', 'redis://localhost:6379')
conn = redis.from_url(redis_url)
redis = Redis()


if __name__ == '__main__':
    with Connection(conn):
        worker = Worker(map(Queue, listen))
        send_kill_horse_command(redis, worker.name)
        worker.work()

        # for channel in listen:
        #     worker = Worker([Queue(channel)], connection=conn, name=f"{channel}_worker")
        #     worker.work()
        
        # aw = AutoWorker(map(Queue, listen), max_procs=6, skip_failed=False)
        # aw.work()
        