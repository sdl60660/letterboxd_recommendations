from flask import Flask, render_template, request, jsonify
# from classes.database import Database, CursorFromConnectionFromPool
# from psycopg2.extensions import AsIs
from urllib.parse import urlparse

import json
import datetime
import os

import pandas as pd
import pickle

from flaskr.data_processing.build_model import build_model
from flaskr.data_processing.run_model import run_model


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('app_config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)
    
    # print(os.getcwd())
    df = pd.read_csv('flaskr/data_processing/data/training_data.csv')
    with open("flaskr/data_processing/models/threshold_movie_list.txt", "rb") as fp:
        threshold_movie_list = pickle.load(fp)
    
    # algo, user_watched_list = build_model(df, "faycwalker")
    # recs = run_model("faycwalker", algo, user_watched_list, threshold_movie_list, 25)

    @app.route('/')
    def homepage():
        return render_template('index.html')

    @app.route('/get_recs')
    def get_recommendations():
        username = request.args.get('username')
        # num_items = int(request.args.get('num_items'))
        num_items = 30
        
        training_data_rows = 200000
        model_df = df.head(training_data_rows)

        algo, user_watched_list = build_model(model_df, username)
        recs = run_model(username, algo, user_watched_list, threshold_movie_list, num_items)

        return(jsonify(recs))


    return app

# SECRET_KEY = os.getenv('SECRET_KEY', config['SECRET_KEY'])
# app = Flask(__name__)
# app.secret_key = SECRET_KEY


if __name__ == "__main__":
    app = create_app()
    app.run(port=5453, debug=True)


