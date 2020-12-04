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
from rq.exceptions import NoSuchJobError
from rq.job import Job
from rq.registry import DeferredJobRegistry

from worker import conn

from handle_recs import get_client_user_data, build_client_model


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('app_config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    queue_pool = [Queue(channel, connection=conn) for channel in ['high', 'default', 'low']]

    @app.route('/')
    def homepage():
        return render_template('index.html')

    @app.route('/get_recs', methods=['GET', 'POST'])
    def get_recs():
        username = request.args.get('username')
        training_data_size = int(request.args.get('training_data_size'))
        if request.args.get('exclude_popular') == "true":
            exclude_popular = True
        else:
            exclude_popular = False

        num_items = 30

        ordered_queues = sorted(queue_pool, key=lambda queue: DeferredJobRegistry(queue=queue).count)
        print([(q, DeferredJobRegistry(queue=q).count) for q in ordered_queues])
        q = ordered_queues[0]
        
        job_get_user_data = q.enqueue(get_client_user_data, args=(username,), description=f"Scraping user data for {request.args.get('username')}", result_ttl=30)
        # job_create_df = q.enqueue(create_training_data, args=(training_data_size, exclude_popular,), depends_on=job_get_user_data, description=f"Creating training dataframe for {request.args.get('username')}", result_ttl=5)
        job_build_model = q.enqueue(build_client_model, args=(username, training_data_size, exclude_popular,num_items,), depends_on=job_get_user_data, description=f"Building model for {request.args.get('username')}", result_ttl=15)
        # job_run_model = q.enqueue(run_client_model, args=(username,num_items,), depends_on=job_build_model, description=f"Running model for {request.args.get('username')}", result_ttl=5)

        return jsonify({
            "redis_get_user_data_job_id": job_get_user_data.get_id(),
            # "redis_create_df_job_id": job_create_df.get_id(),
            "redis_build_model_job_id": job_build_model.get_id()
            # "redis_run_model_job_id": job_run_model.get_id()
            })
       
    
    @app.route("/results", methods=['GET'])
    def get_results():
        job_ids = request.args.to_dict()
        job_statuses = {}
        for key, job_id in job_ids.items():
            # print(key, job_id)
            try:
                job_statuses[key.replace('_id', '_status')] = Job.fetch(job_id, connection=conn).get_status()
            except NoSuchJobError:
                job_statuses[key.replace('_id', '_status')] = "finished"

        end_job = Job.fetch(job_ids['redis_build_model_job_id'], connection=conn)
        # print(job_statuses)

        if end_job.is_finished:
            return jsonify({"statuses": job_statuses, "result": end_job.result}), 200
        else:
            return jsonify({"statuses": job_statuses}), 202


    return app

app = create_app()
SECRET_KEY = os.getenv('SECRET_KEY', '12345')
app.secret_key = SECRET_KEY


if __name__ == "__main__":
    app = create_app()
    app.run(port=5453, debug=True)


