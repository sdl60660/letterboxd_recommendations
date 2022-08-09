import os
from urllib.parse import urlparse
import redis
# from redis import Redis
from rq import Worker, Queue, Connection
# from rq.command import send_kill_horse_command


listen = ['high', 'default', 'low']

# redis_url = os.getenv('REDISTOGO_URL', 'redis://localhost:6379')
redis_url = 'redis://localhost:6379'
if os.getenv('REDISTOGO_URL'):
    redis_url = urlparse.urlparse(os.getenv('REDISCLOUD_URL'))

print('here', redis_url, os.getenv('REDISCLOUD_URL'))

conn = redis.from_url(redis_url)

if __name__ == '__main__':
    with Connection(conn):
        worker = Worker(map(Queue, listen))
        worker.work()
        # send_kill_horse_command(redis, worker.name)
