#!/usr/local/bin/python3.12

import os
import pymongo
import pickle
import pandas as pd
import numpy as np
import random

from collections import defaultdict, namedtuple

from types import SimpleNamespace, MethodType

# from surprise import Dataset, SVD, Reader
# from surprise.model_selection import GridSearchCV
from surprise.dump import load
from surprise.utils import get_rng
# from surprise.predictions import PredictionImpossible


if os.getcwd().endswith("data_processing"):
    from db_connect import connect_to_db
    from utils.utils import explicit_exclude_list
    from get_user_ratings import get_user_data

else:
    from data_processing.db_connect import connect_to_db
    from data_processing.utils.utils import explicit_exclude_list
    from data_processing.get_user_ratings import get_user_data


Prediction = namedtuple("Prediction", "uid iid r_ui est details")

def _estimate_like(self, iu, ii):
    """iu, ii are *inner* indices (ints) or None if unknown."""
    biased = getattr(self, "biased", True)
    mu = getattr(self, "mu", 0.0)

    # Known flags
    ku = (iu is not None) and (iu < self.pu.shape[0])
    ki = (ii is not None) and (ii < self.qi.shape[0])

    if biased:
        est = mu
        if ku:
            est += self.bu[iu]
        if ki:
            est += self.bi[ii]
        if ku and ki:
            est += float(np.dot(self.qi[ii], self.pu[iu]))
        return est
    else:
        # Unbiased: only dot(pu, qi) when both are known; otherwise fall back to 0
        if ku and ki:
            return float(np.dot(self.qi[ii], self.pu[iu]))
        return 0.0

def _predict_like(self, uid_raw, iid_raw, r_ui=None, clip=True):
    iu = self.user_index.get(uid_raw, None)
    ii = self.item_index.get(iid_raw, None)

    est = _estimate_like(self, iu, ii)
    details = {"was_impossible": False, "inner_uid": iu, "inner_iid": ii}

    if clip:
        lo = getattr(self, "rating_min", 1.0)
        hi = getattr(self, "rating_max", 10.0)
        est = min(hi, max(lo, est))

    return Prediction(uid_raw, iid_raw, r_ui, float(est), details)

def _test_like(self, testset, verbose=False):
    """testset: iterable of (uid_raw, iid_raw, r_ui_trans)."""
    preds = [_predict_like(self, uid, iid, r_ui, clip=True) for (uid, iid, r_ui) in testset]
    if verbose:
        for p in preds:
            print(p)
    return preds

def attach_surprise_like_methods(model_ns):
    """Bind the helpers as methods on your SimpleNamespace model."""
    model_ns.estimate = MethodType(_estimate_like, model_ns)
    model_ns.predict  = MethodType(_predict_like,  model_ns)
    model_ns.test     = MethodType(_test_like,     model_ns)
    return model_ns

def get_top_n(predictions, n=20):
    top_n = [(iid, est) for uid, iid, true_r, est, _ in predictions]
    top_n.sort(key=lambda x: (x[1], random.random()), reverse=True)

    return top_n[:n]

def get_prediction_set(username, user_watched_list, sample_movie_list):
    exclude_list = user_watched_list + explicit_exclude_list
    # unwatched_movies = [x for x in sample_movie_list if x not in user_watched_list]
    valid_unwatched_movies = [x for x in sample_movie_list if x not in exclude_list]
    prediction_set = [(username, x, 0) for x in valid_unwatched_movies]

    return prediction_set


def get_movie_data(top_n):
    movie_fields = [
        "image_url",
        "movie_id",
        "movie_title",
        "year_released",
        "genres",
        "original_language",
        "popularity",
        "runtime",
        "release_date",
        "content_type"
    ]

    db_name, client = connect_to_db()
    db = client[db_name]
    movie_data = {
        x["movie_id"]: {k: v for k, v in x.items() if k in movie_fields}
        for x in db.movies.find({"movie_id": {"$in": [x[0] for x in top_n]}})
    }

    return movie_data


def update_trainset_user_data(model, new_ratings_set, uid, username):
    # update ur entry in trainset (which should in turn allow the user to be found when predict/estimate searches for id)
    ur_dict_entry = []
    for u,i,r in new_ratings_set:
        ur_dict_entry.append((i, r))

    model.trainset.ur[uid] = ur_dict_entry
    model.trainset._raw2inner_id_users[username] = uid
    return model

def adjust_model_for_user(model, new_ratings_set, uid, username):
    user_in_training_set = (uid != model.pu.shape[0])
    rng = get_rng(model.random_state)

    # model = update_trainset_user_data(model, new_ratings_set, uid, username)

    # append a new zero-value item to the end of the user-bias numpy array for the new user
    bu = model.bu
    if user_in_training_set:
        bu[uid] = 0
    else:
        bu = np.append(model.bu, 0)

    # append a new slice onto the user-factors array for the new user
    new_user_factor_slice = rng.normal(model.init_mean, model.init_std_dev, size=(1, model.n_factors))
    pu = model.pu

    if user_in_training_set:
        pu[uid] = new_user_factor_slice
    else:
        pu = np.concatenate((model.pu, new_user_factor_slice), axis=0)

    # This is a modified version of the way the SVD model trains off of the full dataset in the surprise library...
    # which can be found here: https://github.com/NicolasHug/Surprise/blob/2381fb11d0c4bf917cc4b9126f205d0013649966/surprise/prediction_algorithms/matrix_factorization.pyx#L159-L252
    # this allows us to fold-in a new user without fully retraining. Technically, the global_mean/item biases/item factors...
    # aren't being adjusted here as they would be during training, but the adjustments from this one individual user would...
    # be tiny and in this case, it's not clear it's best to be making them anyway, as that's fitting to the user that we're...
    # about to estimate for

    bi = model.bi
    qi = model.qi

    global_mean = model.trainset.global_mean if hasattr(model, 'trainset') else model.mu

    # don't need to iterate because we won't really worry about re-adjusting global_mean, etc. based...
    # on this user's ratings, which aren't likely to impact it much at all
    for current_epoch in range(model.n_epochs):
        for u, i, r in new_ratings_set:
            # compute current error
            dot = 0  # <q_i, p_u>

            for f in range(model.n_factors):
                dot += qi[i, f] * pu[u, f]

            err = r - (global_mean + bu[u] + bi[i] + dot)

            # update biases
            if model.biased:
                bu[u] += model.lr_bu * (err - model.reg_bu * bu[u])
                # model.bi[i] += model.lr_bi * (err - model.reg_bi * model.bi[i])

            # update factors
            for f in range(model.n_factors):
                puf = pu[u, f]
                qif = qi[i, f]
                pu[u, f] += model.lr_pu * (err * qif - model.reg_pu * puf)
                # qi[i, f] += model.lr_qi * (err * puf - model.reg_qi * qif)
    
    model.bu = np.asarray(bu)
    model.bi = np.asarray(bi)
    model.pu = np.asarray(pu)
    model.qi = np.asarray(qi)
    
    return model


def update_algo(algo, username, user_data):
    # training_set = algo.trainset

    try:
        uid = algo.user_index[username]
    except KeyError:
        # new user id should be the total exising users (index off-by-one)
        uid = algo.pu.shape[0]

    new_ratings_set = []
    for item in user_data:
        try:
            iid = algo.item_index[item['movie_id']]
            new_ratings_set.append((uid, iid, float(item['rating_val'])))
        except KeyError:
            # print(f"Cannot find a corresponding item ID in the training set for {item['movie_id']}")
            pass
        
    updated_model = adjust_model_for_user(algo, new_ratings_set, uid, username)
    algo.user_index[username] = uid

    return updated_model

def run_model(
    username, algo, user_data, user_watched_list, sample_movie_list, num_recommendations=20, fold_in=True
):
    if fold_in:
        algo = update_algo(algo, username, user_data)

    prediction_set = get_prediction_set(username, user_watched_list, sample_movie_list)
    predictions = algo.test(prediction_set)
    top_n = get_top_n(predictions, num_recommendations)
    
    movie_data = get_movie_data(top_n)

    return_object = [
        {
            "movie_id": x[0],
            "predicted_rating": round(x[1], 3),
            "unclipped_rating": round(x[1], 3),
            "movie_data": movie_data[x[0]],
        }
        for x in top_n
    ]

    for i, prediction in enumerate(return_object):
        if prediction["predicted_rating"] == 10:
            return_object[i]["unclipped_rating"] = float(
                algo.predict(username, prediction["movie_id"], clip=False).est
            )

    # filter out any tv shows (based on TMDB data)
    # this conditional is a little funky for now because the "content_type" field hasn't finished backfilling in the movies collection
    return_object = [x for x in return_object if 'content_type' not in x['movie_data'].keys() or x['movie_data']['content_type'] != 'tv']
    return_object.sort(key=lambda x: (x["unclipped_rating"]), reverse=True)

    return return_object

def load_compressed_model(path):
    blob = np.load(path, allow_pickle=True)
    
    model_dict = {
        "qi": blob["qi"],
        "bi": blob["bi"],
        "pu": blob["pu"],
        "bu": blob["bu"],
        "mu": float(blob["mu"]),
        "user_ids": list(blob["user_ids"]),
        "item_ids": list(blob["item_ids"]),
        "n_factors": int(blob["n_factors"]),
        "n_epochs": int(blob["n_epochs"]),
        "reg_bu": float(blob["reg_bu"]),
        "reg_pu": float(blob["reg_pu"]),
        "reg_qi": float(blob["reg_qi"]),
        "reg_bi": float(blob["reg_bi"]),
        "lr_bu": float(blob["lr_bu"]),
        "lr_pu": float(blob["lr_pu"]),
        "lr_qi": float(blob["lr_qi"]),
        "lr_bi": float(blob["lr_bi"]),
        "rating_min": float(blob["rating_min"]),
        "rating_max": float(blob["rating_max"]),
        "random_state": int(blob["random_state"]),
        "init_mean": float(blob["init_mean"]),
        "init_std_dev": float(blob["init_std_dev"]),
        "biased": bool(blob["biased"])
    }

    model_dict["item_index"] = {mid: i for i, mid in enumerate(model_dict["item_ids"])}
    model_dict["user_index"] = {uid: i for i, uid in enumerate(model_dict["user_ids"])}

    return SimpleNamespace(**model_dict)


def main(username, sample_size = 1000000, fold_in=False, num_recommendations=25):
    # algo = load(f"models/model_{sample_size}.npz")[1]
    algo = load_compressed_model(f"models/model_{sample_size}.npz")
    attach_surprise_like_methods(algo)

    with open(f"data/movie_lists/sample_movie_list_{sample_size}.txt", "rb") as fp:
        sample_movie_list = pickle.load(fp)

    if fold_in == True:
        user_data = get_user_data(username)[0]
        user_watched_list = [x["movie_id"] for x in user_data]

    else:
        with open("models/user_watched.txt", "rb") as fp:
            user_watched_list = pickle.load(fp)

        with open("models/user_data.txt", "rb") as fp:
            user_data = pickle.load(fp)
        
    recs = run_model(username, algo, user_data, user_watched_list, sample_movie_list, num_recommendations, fold_in)
    print([{'movie': x['movie_id'], 'rating': x['predicted_rating']} for x in recs[:10]])
    return recs


if __name__ == "__main__":
    main("samlearner", sample_size=1000000, fold_in=True, num_recommendations=25)
