#!/usr/local/bin/python3.12

import pickle
import random
import numpy as np
import pandas as pd

from surprise import SVD, Reader, Dataset, BaselineOnly, SVDpp

from surprise.model_selection import cross_validate
from surprise.dump import dump

from build_model import train_model, get_dataset
from run_model import get_top_n

params = {
    # "n_factors": 150,
    # "n_epochs": 20,
    # "lr_all": 0.005,
    # "reg_all": 0.02
}

# bsl_options = {"method": "als", "n_epochs": 10, "reg_u": 1, "reg_i": 1}
# algo = BaselineOnly(bsl_options=bsl_options)

def get_datasets(sample_sizes):
  datasets = []
  for sample_size in sample_sizes:
    df = pd.read_parquet(f"data/training_data_samples/training_data_{sample_size}.parquet")
    dataset = get_dataset(df)
    datasets.append({
      'sample_size': sample_size,
      'dataset': dataset
    })

  return datasets

def evaluate_config(dataset, model, params={}):
  data = dataset['dataset']
  sample_size = dataset['sample_size']

  print(f'Testing model: {model['name']} at sample size {sample_size}...')
  algo = train_model(data=data, model=model['model'], params=params, run_cv=False)
  cv = cross_validate(algo, data, measures=['RMSE', 'MAE', 'FCP'], cv=4, verbose=False)

  eval_row = {
    'model': model['name'],
    'sample_size': sample_size,
    'params': params,
    'RMSE': cv['test_rmse'].mean(),
  }
  return eval_row


def main():
  sample_sizes = [500_000, 1_000_000, 2_000_000, 3_000_000]
  # models = [{'name': 'SVD', 'model': SVD}, {'name': 'SVD++', 'model': SVDpp}, {'name': 'ALS', 'model': BaselineOnly}]
  models = [{'name': 'SVD', 'model': SVD}]
  datasets = get_datasets(sample_sizes)

  eval_rows = []
  for dataset in datasets:
    for model in models:
      config_eval = evaluate_config(dataset, model)
      eval_rows.append(config_eval)

  eval_table = pd.DataFrame(eval_rows)
  print(eval_table.head(20))

  
  

if __name__ == "__main__":
    main()

    # dump("models/mini_model.pkl", predictions=None, algo=algo, verbose=1)
    # with open("models/user_watched.txt", "wb") as fp:
    #     pickle.dump(user_watched_list, fp)
