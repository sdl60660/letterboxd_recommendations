#!/usr/local/bin/python3.12

import os
import datetime
import json
from bs4 import BeautifulSoup

import pymongo
import pandas as pd

import time
from tqdm import tqdm

from pymongo import UpdateOne
from pymongo.errors import BulkWriteError

from db_connect import connect_to_db



def main():
    # Connect to MongoDB client
    db_name, client = connect_to_db()
    db = client[db_name]
    movies = db.movies


if __name__ == "__main__":
    main()