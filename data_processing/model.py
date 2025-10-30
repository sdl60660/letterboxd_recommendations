# model.py
from __future__ import annotations
import numpy as np
from collections import namedtuple
from surprise.utils import get_rng

Prediction = namedtuple("Prediction", "uid iid r_ui est details")

class Model:
    """
    Backward-compatible wrapper for compressed SVD state.
    - Loads from NPZ saved by export (with pu/bu, qi/bi, lr_*, reg_*, etc.).
    - Provides Surprise-like predict/test/estimate (w/ adjustments to avoid failing on conditionals w/ data not stored)
    """

    # ---------- construction ----------

    def __init__(self, **kw):
        # core params
        self.qi = kw["qi"]               # (n_items, k)
        self.bi = kw["bi"]               # (n_items,)
        self.pu = kw["pu"]               # (n_users, k)
        self.bu = kw["bu"]               # (n_users,)
        self.mu = float(kw["mu"])

        self.item_ids = list(kw["item_ids"])
        self.user_ids = list(kw["user_ids"])

        self.n_factors = int(kw["n_factors"])
        self.n_epochs  = int(kw["n_epochs"])
        self.rating_min = float(kw["rating_min"])
        self.rating_max = float(kw["rating_max"])
        self.biased = bool(kw.get("biased", True))

        # learning rates / regs (kept for compatibility with your current fold-in)
        self.lr_bu = float(kw.get("lr_bu", 0.005))
        self.lr_bi = float(kw.get("lr_bi", 0.005))
        self.lr_pu = float(kw.get("lr_pu", 0.005))
        self.lr_qi = float(kw.get("lr_qi", 0.005))

        self.reg_bu = float(kw.get("reg_bu", 0.02))
        self.reg_bi = float(kw.get("reg_bi", 0.02))
        self.reg_pu = float(kw.get("reg_pu", 0.02))
        self.reg_qi = float(kw.get("reg_qi", 0.02))

        self.random_state = int(kw.get("random_state", 0))
        self.init_mean = float(kw.get("init_mean", 0.0))
        self.init_std_dev = float(kw.get("init_std_dev", 0.1))

        # indices
        self.item_index = {mid: i for i, mid in enumerate(self.item_ids)}
        self.user_index = {uid: i for i, uid in enumerate(self.user_ids)}

    @classmethod
    def from_npz(cls, path: str) -> "Model":
        blob = np.load(path, allow_pickle=True)

        d = {
            "qi": blob["qi"],
            "bi": blob["bi"],
            "pu": blob["pu"],
            "bu": blob["bu"],
            "mu": float(blob["mu"]),
            "item_ids": list(blob["item_ids"]),
            "user_ids": list(blob["user_ids"]),
            "n_factors": int(blob["n_factors"]),
            "n_epochs": int(blob["n_epochs"]),
            "rating_min": float(blob["rating_min"]),
            "rating_max": float(blob["rating_max"]),
            "biased": bool(blob.get("biased", True)),

            "lr_bu": float(blob.get("lr_bu", 0.005)),
            "lr_bi": float(blob.get("lr_bi", 0.005)),
            "lr_pu": float(blob.get("lr_pu", 0.005)),
            "lr_qi": float(blob.get("lr_qi", 0.005)),

            "reg_bu": float(blob.get("reg_bu", 0.02)),
            "reg_bi": float(blob.get("reg_bi", 0.02)),
            "reg_pu": float(blob.get("reg_pu", 0.02)),
            "reg_qi": float(blob.get("reg_qi", 0.02)),

            "random_state": int(blob.get("random_state", 0)),
            "init_mean": float(blob.get("init_mean", 0.0)),
            "init_std_dev": float(blob.get("init_std_dev", 0.1)),
        }
        return cls(**d)

    def to_npz(self, path: str) -> None:
        np.savez_compressed(
            path,
            qi=self.qi, bi=self.bi, pu=self.pu, bu=self.bu,
            mu=np.float32(self.mu),
            item_ids=np.array(self.item_ids, dtype=object),
            user_ids=np.array(self.user_ids, dtype=object),
            n_factors=np.int16(self.n_factors),
            n_epochs=np.int16(self.n_epochs),
            rating_min=np.float32(self.rating_min),
            rating_max=np.float32(self.rating_max),
            biased=np.bool_(self.biased),
            lr_bu=np.float32(self.lr_bu), lr_bi=np.float32(self.lr_bi),
            lr_pu=np.float32(self.lr_pu), lr_qi=np.float32(self.lr_qi),
            reg_bu=np.float32(self.reg_bu), reg_bi=np.float32(self.reg_bi),
            reg_pu=np.float32(self.reg_pu), reg_qi=np.float32(self.reg_qi),
            random_state=np.int16(self.random_state),
            init_mean=np.float32(self.init_mean),
            init_std_dev=np.float32(self.init_std_dev),
        )

    # ---------- Surprise-like API ----------
    # Methods on Surprise's SVD/AlgoBase have some conditionals that the folded-in data
    # will fail. Such as the `self.trainset.knows_user(u)` in `estimate()`, where, even though
    # the new user's data is folded in, I'm specifically not storing the trainset with the model.
    # I could override something/work around that, but as long as I'm overriding, may as well just
    # override the methods

    def estimate_inner(self, iu: int | None, ii: int | None) -> float:
        ku = (iu is not None) and (iu < self.pu.shape[0])
        ki = (ii is not None) and (ii < self.qi.shape[0])

        if self.biased:
            est = self.mu
            if ku: est += self.bu[iu]
            if ki: est += self.bi[ii]
            if ku and ki: est += float(self.qi[ii].dot(self.pu[iu]))
            return est
        else:
            if ku and ki:
                return float(self.qi[ii].dot(self.pu[iu]))
            return 0.0

    def predict(self, uid_raw, iid_raw, r_ui=None, clip=True):
        iu = self.user_index.get(uid_raw, None)
        ii = self.item_index.get(iid_raw, None)
        est = self.estimate_inner(iu, ii)
        if clip:
            est = min(self.rating_max, max(self.rating_min, est))
        details = {"was_impossible": False, "inner_uid": iu, "inner_iid": ii}
        return Prediction(uid_raw, iid_raw, r_ui, float(est), details)

    def test(self, testset, verbose=False):
        preds = [self.predict(uid, iid, r_ui, clip=True) for (uid, iid, r_ui) in testset]
        if verbose:
            for p in preds: print(p)
        return preds

    # ---------- Will adjust this method soon, maintaining like this for now for compatibility ----------

    def adjust_model_for_user(self, new_ratings_set, uid: int, username: str):
        """
        new_ratings_set: list of (u_inner, i_inner, rating_float) triples using *inner* ids.
        """
        user_in_training_set = (uid != self.pu.shape[0])
        rng = get_rng(self.random_state)

        # prepare bu/pu rows
        bu = self.bu
        if user_in_training_set:
            bu[uid] = 0.0
        else:
            bu = np.append(self.bu, 0.0)

        new_user_slice = rng.normal(self.init_mean, self.init_std_dev, size=(1, self.n_factors))
        pu = self.pu
        if user_in_training_set:
            pu[uid] = new_user_slice
        else:
            pu = np.concatenate((self.pu, new_user_slice), axis=0)

        bi = self.bi
        qi = self.qi
        global_mean = self.mu

        for _ in range(self.n_epochs):
            for u, i, r in new_ratings_set:
                # dot(qi[i], pu[u])
                dot = float(np.dot(qi[i], pu[u]))
                err = r - (global_mean + bu[u] + bi[i] + dot)

                # biases
                if self.biased:
                    bu[u] += self.lr_bu * (err - self.reg_bu * bu[u])
                    # self.bi[i] += self.lr_bi * (err - self.reg_bi * self.bi[i])  # intentionally frozen

                # factors (user-only updates)
                grad_pu = err * qi[i] - self.reg_pu * pu[u]
                pu[u] += self.lr_pu * grad_pu
                # qi[i] frozen for now

        self.bu = np.asarray(bu)
        self.pu = np.asarray(pu)

        # bi/qi unchanged
        self.user_index[username] = uid  # ensure predict() sees this user
        if not user_in_training_set:
            self.user_ids.append(username)

        return self

    # helper to map raw user ratings to inner ids and call adjust
    def update_algo(self, username: str, user_data: list[dict]) -> "Model":
        """
        user_data: list of {'movie_id': str, 'rating_val': float}
        """
        uid = self.user_index.get(username, None)
        if uid is None:
            uid = self.pu.shape[0]  # next row

        triples = []
        for item in user_data:
            ii = self.item_index.get(item["movie_id"])
            if ii is not None:
                triples.append((uid, ii, float(item["rating_val"])))

        return self.adjust_model_for_user(triples, uid, username)


