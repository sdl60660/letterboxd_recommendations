"""
THIS FILE IS NOW OUTDATED, USING FASTAPI IN main.py INSTEAD OF FLASK
WILL DELETE IT IN THE NEAR FUTURE AND CLEAR OUT SOME OTHER OUTDATE FILES
"""

from flask import Flask, render_template, request, jsonify, redirect
from flask_talisman import Talisman
from flask_cors import CORS
# from classes.database import Database, CursorFromConnectionFromPool
# from psycopg2.extensions import AsIs
from urllib.parse import urlparse, urlunparse

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


FROM_DOMAIN = "letterboxd-recommendations.herokuapp.com"
TO_DOMAIN = "letterboxd.samlearner.com"


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    csp = {
        'default-src': ['*', "'unsafe-inline'", "'unsafe-eval'"]
        # 'script-src': ["'unsafe-inline'", "'nonce-allow'"],
        # 'style-src': ["'unsafe-inline'", "'nonce-allow'"],
        # 'connect-src': ["'unsafe-inline'", "'nonce-allow'"],
        # 'img-src': ["*"],
        # 'default-src': [
        #     '\'self\'',
        #     "'nonce-allow'",
        #     "d3js.org"
        # ]
    }
    Talisman(app, content_security_policy=csp)
    CORS(app)

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('app_config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    queue_pool = [Queue(channel, connection=conn)
                  for channel in ['high', 'default', 'low']]
    # queue_pool = [Queue(channel, connection=conn) for channel in ['high']]
    popularity_thresholds_500k_samples = [
        2500, 2000, 1500, 1000, 700, 400, 250, 150]

    @app.before_request
    def redirect_to_new_domain():
        urlparts = urlparse(request.url)

        if urlparts.netloc == FROM_DOMAIN and urlparts.path == "/":
            urlparts_list = list(urlparts)
            urlparts_list[1] = TO_DOMAIN
            return redirect(urlunparse(urlparts_list), code=301)

    @app.route('/')
    def homepage():
        return render_template('index.html')

    @app.route('/get_recs', methods=['GET', 'POST'])
    def get_recs():
        username = request.args.get('username').lower().strip()
        training_data_size = int(request.args.get('training_data_size'))
        popularity_filter = int(request.args.get("popularity_filter"))
        data_opt_in = (request.args.get("data_opt_in") == "true")

        if popularity_filter >= 0:
            popularity_threshold = popularity_thresholds_500k_samples[popularity_filter]
        else:
            popularity_threshold = None

        num_items = 1200

        ordered_queues = sorted(
            queue_pool, key=lambda queue: DeferredJobRegistry(queue=queue).count)
        print([(q, DeferredJobRegistry(queue=q).count)
              for q in ordered_queues])
        q = ordered_queues[0]

        job_get_user_data = q.enqueue(get_client_user_data, args=(
            username, data_opt_in,), description=f"Scraping user data for {request.args.get('username')} (sample: {training_data_size}, popularity_filter: {popularity_threshold}, data_opt_in: {data_opt_in})", result_ttl=45, ttl=300)
        # job_create_df = q.enqueue(create_training_data, args=(training_data_size, exclude_popular,), depends_on=job_get_user_data, description=f"Creating training dataframe for {request.args.get('username')}", result_ttl=5)
        job_build_model = q.enqueue(build_client_model, args=(username, training_data_size, popularity_threshold, num_items,), depends_on=job_get_user_data,
                                    description=f"Building model for {request.args.get('username')} (sample: {training_data_size}, popularity_filter: {popularity_threshold})", result_ttl=30, ttl=300)
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
            try:
                job_statuses[key.replace('_id', '_status')] = Job.fetch(
                    job_id, connection=conn).get_status()
            except NoSuchJobError:
                job_statuses[key.replace('_id', '_status')] = "finished"

        end_job = Job.fetch(
            job_ids['redis_build_model_job_id'], connection=conn)
        execution_data = {"build_model_stage": end_job.meta.get('stage')}

        try:
            user_job = Job.fetch(
                job_ids['redis_get_user_data_job_id'], connection=conn)
            execution_data |= {"num_user_ratings": user_job.meta.get(
                'num_user_ratings'), "user_status": user_job.meta.get('user_status')}
        except NoSuchJobError:
            pass

        if end_job.is_finished:
            return jsonify({"statuses": job_statuses, "execution_data": execution_data, "result": end_job.result}), 200
        else:
            return jsonify({"statuses": job_statuses, "execution_data": execution_data}), 202

    return app


app = create_app()
SECRET_KEY = os.getenv('SECRET_KEY', '12345')
app.secret_key = SECRET_KEY


if __name__ == "__main__":
    app = create_app()
    app.run(port=5453, debug=True)
