import os
from urllib.parse import urlparse
import redis
# from redis import Redis
from rq import Worker, Queue
# from rq.command import send_kill_horse_command


listen = ['high', 'default', 'low']

redis_url = os.getenv('REDISCLOUD_URL', 'redis://localhost:6379')
conn = redis.from_url(redis_url)
queues = [Queue(x, connection=conn) for x in listen]

if __name__ == '__main__':
    worker = Worker(queues, connection=conn)
    worker.work()
    # with Connection(conn):
        # worker = Worker(map(Queue, listen))
        # worker.work()
        # send_kill_horse_command(redis, worker.name)
