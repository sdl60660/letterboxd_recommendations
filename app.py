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

from handle_recs import get_recommendations, add_nums


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('app_config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)
    
    # if os.getcwd().endswith("flaskr"):
    df = pd.read_csv('data_processing/data/training_data.csv')
    with open("data_processing/models/threshold_movie_list.txt", "rb") as fp:
        threshold_movie_list = pickle.load(fp)
    # else:
    #     df = pd.read_csv('flaskr/data_processing/data/training_data.csv')
    #     with open("flaskr/data_processing/models/threshold_movie_list.txt", "rb") as fp:
    #         threshold_movie_list = pickle.load(fp)
    

    q = Queue(connection=conn)

    @app.route('/')
    def homepage():
        return render_template('index.html')

    @app.route('/get_recs')
    def get_recs():
        
        job = q.enqueue(get_recommendations, args=(request, df, threshold_movie_list,))
        print(job.get_id())
        return jsonify({"redis_job_id": job.get_id()})
       
        # if os.getenv('REDISTOGO_URL'):
        # else:
        #     recs = get_recommendations(request, df, threshold_movie_list)
        #     return jsonify(recs)
    
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


