from flask import Flask, render_template, request, jsonify
# from classes.database import Database, CursorFromConnectionFromPool
# from psycopg2.extensions import AsIs
from urllib.parse import urlparse

import json
import datetime
import os

import pandas as pd
import pickle

from rq import Queue
from rq.job import Job
from worker import conn

from handle_recs import get_recommendations


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('app_config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    q = Queue(connection=conn)

    @app.route('/')
    def homepage():
        return render_template('index.html')

    @app.route('/get_recs')
    def get_recs():
        username = request.args.get('username')

        job = q.enqueue(get_recommendations, args=(username,), description=f"Recs for {request.args.get('username')}")
        print(job.get_id())
        return jsonify({"redis_job_id": job.get_id()})
       
    
    @app.route("/results/<job_key>", methods=['GET'])
    def get_results(job_key):

        job = Job.fetch(job_key, connection=conn)
        print(job)

        if job.is_finished:
            return jsonify(job.result), 200
        else:
            return "Nay!", 202


    return app

app = create_app()
SECRET_KEY = os.getenv('SECRET_KEY', '12345')
app.secret_key = SECRET_KEY


if __name__ == "__main__":
    app = create_app()
    app.run(port=5453, debug=True)


