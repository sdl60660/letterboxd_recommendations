#!/usr/local/bin/python3.12

import pickle
import random
from collections import defaultdict

import numpy as np
import pandas as pd

from surprise import SVD, Reader, Dataset, BaselineOnly, SVDpp

from surprise.model_selection import cross_validate, KFold
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

def precision_recall_at_k(predictions, k=20, threshold=6):
  """Return precision and recall at k metrics for each user"""

  # Map predcitions to users
  user_est_true = defaultdict(list)
  for uid, _, true_r, est, _ in predictions:
      user_est_true[uid].append((est, true_r))
  
  precisions = dict()
  recalls = dict()

  for uid, user_ratings in user_est_true.items():

    # Sort users by estimated value
    user_ratings.sort(key=lambda x: x[0], reverse=True)

    # Number of relevant items
    n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

    # Number of recommended items in top k
    n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

    # Number of relevant and recommended items in top k
    n_rel_and_rec_k = sum(
      ((true_r >= threshold) and (est >= threshold))
      for (est, true_r) in user_ratings[:k]
    )

    # Precision@K: Proportion of recommended items that are relevant
    # When n_rec_k is 0, Precision is undefined/set to 0 here
    precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

    # Recall@K: Proportion of relevant items that are recommended
    # When n_rel is 0, Recall is undefined/set to 0 here
    recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
  
  return precisions, recalls


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

def evaluate_config(dataset, model, params={}, cv_folds=4):
  data = dataset['dataset']
  sample_size = dataset['sample_size']

  print(f'Testing model: {model['name']} at sample size {sample_size}...')
  algo = train_model(data=data, model=model['model'], params=params, run_cv=False)

  kf = KFold(cv_folds)
  for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    precisions, recalls = precision_recall_at_k(predictions, k=20, threshold=6)
  
  mean_precision = sum(prec for prec in precisions.values()) / len(precisions)
  mean_recall = sum(rec for rec in recalls.values()) / len(recalls)

  cv = cross_validate(algo, data, measures=['RMSE', 'MAE', 'FCP'], cv=cv_folds, verbose=False)

  eval_row = {
    'model': model['name'],
    'sample_size': sample_size,
    'params': params,
    'RMSE': cv['test_rmse'].mean(),
    'Precision@K': mean_precision,
    'Recall@K': mean_recall
  }
  return eval_row


def main():
  sample_sizes = [500_000, 1_000_000, 2_000_000, 3_000_000]
  # models = [{'name': 'SVD', 'model': SVD}, {'name': 'SVD++', 'model': SVDpp}, {'name': 'ALS', 'model': BaselineOnly}]
  # models = [{'name': 'SVD', 'model': SVD}]
  models = [{'name': 'ALS', 'model': BaselineOnly}]

  bsl_options = {"method": "als", "n_epochs": 10, "reg_u": 1, "reg_i": 1}
  # bsl_options = {"method": "als"}

  datasets = get_datasets(sample_sizes)

  eval_rows = []
  for dataset in datasets:
    for model in models:
      # config_eval = evaluate_config(dataset, model, params={})
      config_eval = evaluate_config(dataset, model, params={"bsl_options": bsl_options})
      eval_rows.append(config_eval)

  eval_table = pd.DataFrame(eval_rows)
  print(eval_table.head(20))

  
  

if __name__ == "__main__":
    main()

    # dump("models/mini_model.pkl", predictions=None, algo=algo, verbose=1)
    # with open("models/user_watched.txt", "wb") as fp:
    #     pickle.dump(user_watched_list, fp)
