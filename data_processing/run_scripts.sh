#!/bin/bash

python get_users.py
python get_ratings.py

python create_sample_dataframe.py
python build_model.py
python load_model.py