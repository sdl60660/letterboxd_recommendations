#!/usr/local/bin/python3.12

import pickle
import json
import random
from collections import defaultdict, namedtuple

import numpy as np
import pandas as pd

import scipy.stats.distributions as dists

from surprise import SVD, Reader, Dataset, BaselineOnly, SVDpp, accuracy
from surprise.model_selection import cross_validate, KFold, RandomizedSearchCV
from surprise.dump import dump
from surprise import Prediction

from tqdm.auto import tqdm

from model import Model

from build_model import train_model, get_dataset
from run_model import run_model, get_movie_data


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
  algo, training_set = train_model(data=data, model=model['model'], params=params, run_cv=False)

  kf = KFold(cv_folds)
  for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    precisions, recalls = precision_recall_at_k(predictions, k=50, threshold=7)
  
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


def create_user_test_train_sets(user_data_df, test_users, user_train_test_split=0.7):
  output_set = []

  for user_id in test_users:
    user_data = user_data_df[user_data_df['user_id'] == user_id]

    user_train_data = user_data.sample(frac=user_train_test_split, random_state=12)
    user_test_data = user_data.loc[~user_data.index.isin(user_train_data.index)]

    all_movies = user_data['movie_id'].unique()

    output_set.append({
      "username": user_id,
      "train": user_train_data.to_dict(orient="records"),
      "test": user_test_data.to_dict(orient="records"),
      "test_key": user_test_data.set_index('movie_id').to_dict(orient='index'),
      "movie_set": set(all_movies)
    })
  
  return output_set


def eval_fold_in_user(user_data_set, model):
  username = user_data_set['username']
  user_data = user_data_set['train']
  sample_movie_list = user_data_set['movie_set']
  test_key = user_data_set['test_key']

  results = run_model(username=username, algo=model, user_data=user_data, sample_movie_list=sample_movie_list, movie_data=None, num_recommendations=len(sample_movie_list), fold_in=True)
  predictions = [Prediction(uid=username, iid=x['movie_id'], r_ui=test_key[x['movie_id']]['rating_val'], est=x['predicted_rating'], details=None) for x in results]

  rmse_value = accuracy.rmse(predictions, verbose=False)
  precisions, recalls = precision_recall_at_k(predictions, k=50, threshold=7)

  precision = sum(prec for prec in precisions.values()) / len(precisions)
  recall = sum(rec for rec in recalls.values()) / len(recalls)

  user_metrics = {
    'rmse': rmse_value,
    'precision': precision,
    'recall': recall,
    'total_test_predictions': len(predictions)
  }

  return user_metrics

def eval_param_set_fold_in(params,  training_dataset, user_data_sets):
  algo = train_model(training_dataset, SVD, params=params, run_cv=False)
  model = Model.from_surprise(algo)

  all_user_evals = []
  for j, user_data_set in enumerate(user_data_sets):
    user_metrics = eval_fold_in_user(user_data_set, model)
    all_user_evals.append(user_metrics)
  
  # not actually sure if these should be weighted means or unweighted means (treat each user's...
  # error equally regardless of number of items). for now, they are weighted
  sum_total_predictions = sum([x['total_test_predictions'] for x in all_user_evals])

  mean_rmse = sum([(x['rmse'] * x['total_test_predictions']) for x in all_user_evals]) / sum_total_predictions
  mean_precision = sum([(x['precision'] * x['total_test_predictions']) for x in all_user_evals]) / sum_total_predictions
  mean_recall = sum([(x['recall'] * x['total_test_predictions']) for x in all_user_evals]) / sum_total_predictions

  param_set_metrics = {
    'fold_in_rmse': mean_rmse,
    'fold_in_precision@k': mean_precision,
    'fold_in_recall@k': mean_recall
  }

  print(param_set_metrics)

  return param_set_metrics 


def eval_fold_in(df, param_set_df, num_test_users=1000):  
  params_set = [x for x in param_set_df['params'].str.replace("\'", "\"").apply(json.loads).to_list()]

  # manually split out folds by complete users (or just split into one 1000 user test set and the rest in training)
  # split train/test data by *users*, leaving out a set of 1000 users to use for fold-in testing
  all_users = df['user_id'].unique()
  test_users = all_users[(-1*num_test_users):]

  training_data = df[~df['user_id'].isin(test_users)]
  test_data = df[df['user_id'].isin(test_users)]

  # build dataset on training data
  training_dataset = get_dataset(training_data)

  # create train/test splits or CV folds for each user's ratings
  user_data_sets = create_user_test_train_sets(user_data_df=test_data, test_users=test_users)

  # for each candidate set of params
  #   run user fold-in run_model on with the user's train ratings
  #   evaluate error (RMSE/precision/etc) for the user's test ratings
  #   maybe do this over multiple CV folds
  #   find mean error across all 1000 test users and attach to the larger param output data

  fold_in_eval_rows = []
  for i, params in enumerate(params_set):
    print(f"Working on param set {i+1} of {len(params_set)}")
    param_metrics = eval_param_set_fold_in(params, training_dataset, user_data_sets)
    fold_in_eval_rows.append(fold_in_eval_rows)

    print(f'Model {i+1} -- RMSE: {param_set_df['mean_test_rmse'][i]}, Fold-in RMSE: {param_metrics['fold_in_rmse']}')

  fold_in_cols_df = pd.DataFrame(fold_in_eval_rows)
  return fold_in_cols_df


def run_grid_search(model, dataset):
  param_dists = {'n_factors': dists.randint(100, 250), 'n_epochs': dists.randint(40, 80), 'lr_all': dists.uniform(0.003, 0.008), 'reg_qi': dists.uniform(0.01, 0.04), 'reg_pu': dists.uniform(0.01, 0.04), 'reg_bu': dists.uniform(0.03, 0.08), 'reg_bi': dists.uniform(0.08, 0.25), 'init_std_dev': dists.uniform(0.05, 0.25)}

  rand_search = RandomizedSearchCV(model, param_dists, n_iter = 60, measures=['rmse', 'mae'], cv=4, n_jobs=4, joblib_verbose = 1000)
  rand_search.fit(dataset)

  results_df = pd.DataFrame.from_dict(rand_search.cv_results)
  # for index, row in results_df.iterrows():
  #   eval_fold_in(model, dataset, params=row['params']) 

  results_df = results_df[['mean_test_rmse', 'std_test_rmse', 'rank_test_rmse', 'mean_test_mae', 'std_test_mae', 'rank_test_mae', 'mean_fit_time', 'params']]
  results_df.to_csv('./models/model_param_test_results.csv', index=False)

  best_params = rand_search.best_params["rmse"]
  with open('./models/best_svd_params.json', 'w') as f:
    json.dump(best_params, f)
  
  return best_params, results_df

def main():
  sample_sizes = [500_000, 1_000_000, 2_000_000, 3_000_000]
  models = [{'name': 'SVD', 'model': SVD}]

  sample_size_index = 1

  # datasets = get_datasets(sample_sizes)
  # best_params, param_eval_df = run_grid_search(models[0]['model'], datasets[sample_size_index]['dataset'])

  param_eval_df = pd.read_csv('./models/model_param_test_results.csv')

  df = pd.read_parquet(f"data/training_data_samples/training_data_{sample_sizes[sample_size_index]}.parquet")
  fold_in_cols_df = eval_fold_in(df, param_set_df=param_eval_df)

  rich_param_eval_df = pd.concat([param_eval_df.reset_index(drop=True), fold_in_cols_df.reset_index(drop=True)], axis=1)
  rich_param_eval_df.to_csv('./models/model_param_test_results_with_foldin.csv', index=False)

  # eval_rows = []
  # for dataset in datasets:
  #   for model in models:
  #     config_eval = evaluate_config(dataset, model, params=best_params)
  #     eval_rows.append(config_eval)

  # pd.set_option('display.max_columns', 10)
  # eval_table = pd.DataFrame(eval_rows)
  # print(eval_table.head(20))
  

if __name__ == "__main__":
    main()