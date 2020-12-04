import os

import redis
# from redis import Redis
from rq import Worker, Queue, Connection
# from rq.command import send_kill_horse_command


listen = ['high', 'default', 'low']
redis_url = os.getenv('REDISTOGO_URL', 'redis://localhost:6379')
conn = redis.from_url(redis_url)
# redis = Redis()


if __name__ == '__main__':
    with Connection(conn):
        worker = Worker(map(Queue, listen))
        # send_kill_horse_command(redis, worker.name)
        worker.work()
        