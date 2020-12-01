#!/usr/local/bin/python3.9

from collections import defaultdict

from surprise import Dataset
from surprise import SVD
from surprise import Reader
from surprise.model_selection import GridSearchCV
from surprise import SVD
from surprise.dump import load

import pickle


def get_top_n(predictions, n=20):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


algo = load("models/mini_model.pkl")[1]
prediction = algo.predict('5fc52b1c22862e5421d36cea', "get-out-2017")
print(prediction.est)

data = pickle.load(open("models/mini_model_data.pkl", "rb"))
# First train an SVD algorithm on the movielens dataset.
# data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
# algo = SVD()
# algo.fit(trainset)

# Than predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()

username = "samlearner"
user_set = [x for x in user_set if x[0] === username]
print(user_set)

# print(trainset.ur.keys(), testset[1])
# predictions = algo.test(testset)

# top_n = get_top_n(predictions, n=20)

# # Print the recommended items for each user
# for uid, user_ratings in top_n.items():
#     if str(uid) == "5fc52b1c22862e5421d36cea":
#         print(uid, [(iid, _) for (iid, _) in user_ratings])