# model.py
from __future__ import annotations
import numpy as np
from collections import namedtuple

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
        self.pu = kw.get("pu", None)     # (n_users, k)
        self.bu = kw.get("bu", None)     # (n_users,)
        self.mu = float(kw["mu"])

        self.item_ids = list(kw["item_ids"])
        self.user_ids = list(kw.get("user_ids", []))

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

        self.reg = float(kw.get("reg", kw.get("reg_pu", kw.get("reg_qi", 0.02))))
        self._user_cache = {}  # uid_raw -> (p_u, b_u)
    
    @classmethod
    def from_surprise(cls, algo) -> "Model":
        """Create a Model from a trained Surprise SVD instance."""
        # core tensors (keep float32 for size/speed)
        qi = algo.qi.astype(np.float32)             # (n_items, k)
        bi = algo.bi.astype(np.float32)             # (n_items,)
        pu = algo.pu.astype(np.float32)             # (n_users, k)
        bu = algo.bu.astype(np.float32)             # (n_users,)
        mu = float(algo.trainset.global_mean)

        # raw id lists (inner -> raw)
        inner2raw_item = {i: algo.trainset.to_raw_iid(i) for i in range(algo.trainset.n_items)}
        inner2raw_user = {u: algo.trainset.to_raw_uid(u) for u in range(algo.trainset.n_users)}
        item_ids = [inner2raw_item[i] for i in range(len(inner2raw_item))]
        user_ids = [inner2raw_user[u] for u in range(len(inner2raw_user))]

        return cls(
            qi=qi, bi=bi, pu=pu, bu=bu, mu=mu,
            item_ids=item_ids, user_ids=user_ids,
            n_factors=int(algo.n_factors),
            n_epochs=int(algo.n_epochs),
            rating_min=float(algo.trainset.rating_scale[0]),
            rating_max=float(algo.trainset.rating_scale[1]),
            biased=bool(getattr(algo, "biased", True)),

            lr_bu=float(getattr(algo, "lr_bu", 0.005)),
            lr_bi=float(getattr(algo, "lr_bi", 0.005)),
            lr_pu=float(getattr(algo, "lr_pu", 0.005)),
            lr_qi=float(getattr(algo, "lr_qi", 0.005)),

            reg_bu=float(getattr(algo, "reg_bu", 0.02)),
            reg_bi=float(getattr(algo, "reg_bi", 0.02)),
            reg_pu=float(getattr(algo, "reg_pu", 0.02)),
            reg_qi=float(getattr(algo, "reg_qi", 0.02)),

            random_state=int(getattr(algo, "random_state", 0)),
            init_mean=float(getattr(algo, "init_mean", 0.0)),
            init_std_dev=float(getattr(algo, "init_std_dev", 0.1)),
        )

    @classmethod
    def from_npz(cls, path: str) -> "Model":
        blob = np.load(path, allow_pickle=True)

        d = {
            "qi": blob["qi"],
            "bi": blob["bi"],
            "pu": blob["pu"] if "pu" in blob.files else None,
            "bu": blob["bu"] if "bu" in blob.files else None,
            "mu": float(blob["mu"]),
            "item_ids": list(blob["item_ids"]),
            "user_ids": list(blob["user_ids"]) if "user_ids" in blob.files else [],
            "n_factors": int(blob["n_factors"]),
            "n_epochs": int(blob["n_epochs"]) if "n_epochs" in blob.files else 0,
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

    def to_npz(self, path: str, items_only: bool = False) -> None:
        out = dict(
            qi=self.qi, bi=self.bi,
            mu=np.float32(self.mu),
            item_ids=np.array(self.item_ids, dtype=object),
            n_factors=np.int16(self.n_factors),
            rating_min=np.float32(self.rating_min),
            rating_max=np.float32(self.rating_max),
            biased=np.bool_(self.biased),
            lr_bu=np.float32(self.lr_bu),
            lr_bi=np.float32(self.lr_bi),
            lr_pu=np.float32(self.lr_pu),
            lr_qi=np.float32(self.lr_qi),
            reg_bu=np.float32(self.reg_bu),
            reg_bi=np.float32(self.reg_bi),
            reg_pu=np.float32(self.reg_pu),
            reg_qi=np.float32(self.reg_qi),
            random_state=np.int16(self.random_state),
            init_mean=np.float32(self.init_mean),
            init_std_dev=np.float32(self.init_std_dev),
        )
        if not items_only:
            out.update(
                pu=self.pu if self.pu is not None else np.zeros((0, self.n_factors), self.qi.dtype),
                bu=self.bu if self.bu is not None else np.zeros((0,), np.float32),
                user_ids=np.array(self.user_ids, dtype=object),
                n_epochs=np.int16(self.n_epochs),
            )
        np.savez_compressed(path, **out)

    # ---------- Surprise-like API ----------
    # Methods on Surprise's SVD/AlgoBase have some conditionals that the folded-in data
    # will fail. Such as the `self.trainset.knows_user(u)` in `estimate()`, where, even though
    # the new user's data is folded in, I'm specifically not storing the trainset with the model.
    # I could override something/work around that, but as long as I'm overriding, may as well just
    # override the methods

    def estimate_inner(self, iu: int | None, ii: int | None) -> float:
        ku = (iu is not None) and (self.pu is not None) and (iu < self.pu.shape[0])
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
        ii = self.item_index.get(iid_raw, None)

        if uid_raw in self._user_cache and ii is not None:
            p_u, b_u = self._user_cache[uid_raw]
            est = self.mu + b_u

            if ii is not None:
                est += self.bi[ii] + float(self.qi[ii].dot(p_u))

            if clip:
                est = min(self.rating_max, max(self.rating_min, est))

            details = {"was_impossible": False, "inner_uid": None, "inner_iid": ii}
            return Prediction(uid_raw, iid_raw, r_ui, float(est), details)

        iu = self.user_index.get(uid_raw, None)
        ii = self.item_index.get(iid_raw, None)
        est = self.estimate_inner(iu, ii)

        if clip:
            est = min(self.rating_max, max(self.rating_min, est))

        details = {"was_impossible": False, "inner_uid": iu, "inner_iid": ii}
        return Prediction(uid_raw, iid_raw, r_ui, float(est), details)

    def _scores_unclipped(self, username: str, item_ids: list[str]) -> tuple[list[str], np.ndarray]:
        p_u, b_u = self._user_cache.get(username, (np.zeros(self.n_factors, self.qi.dtype), 0.0))
        idx = [self.item_index[iid] for iid in item_ids if iid in self.item_index]

        I = np.asarray(idx, dtype=np.int64)
        scores = self.mu + b_u + self.bi[I] + (self.qi[I] @ p_u)  # unclipped

        # Preserve original order alignment
        filtered_items = [iid for iid in item_ids if iid in self.item_index]
        return filtered_items, scores

    
    def test(self, testset, clip_ratings: bool=False, verbose: bool=False):
        uids = {uid for (uid, _, _) in testset}

        # If test set is for single user, this will be faster
        if len(uids) == 1:
            username = next(iter(uids))
            items = [iid for (_, iid, _) in testset]
            items_filt, scores = self._scores_unclipped(username, items)

            preds = []
            it = iter(scores)

            for (uid, iid, r_ui) in testset:
                if iid in self.item_index:
                    score = float(next(it))
                    est = min(self.rating_max, max(self.rating_min, score)) if clip_ratings else score

                else:
                    # unknown item: fall back to baseline
                    est = float(self.mu)

                pred = Prediction(uid, iid, r_ui, est, {"was_impossible": False})

                if verbose: print(pred)
                preds.append(pred)

            return preds

        # Fallback: mixed users → use per-item predict
        preds = [self.predict(uid, iid, r_ui, clip=clip_ratings) for (uid, iid, r_ui) in testset]

        if verbose:
            for p in preds: print(p)

        return preds

    def _fold_in_from_pairs(self, uid_raw: str, pairs: list[tuple[int, float]], reg: float | None = None):
        if not pairs:
            self._user_cache[uid_raw] = (np.zeros(self.n_factors, dtype=self.qi.dtype), 0.0)
            return

        I, r = zip(*pairs)
        I = np.asarray(I, dtype=np.int64)
        r = np.asarray(r, dtype=np.float32)

        Q = self.qi[I, :]
        y = r - (self.mu + self.bi[I])

        m = Q.shape[0]
        lam_p = (self.reg_pu if reg is None else float(reg)) * max(1, m + self.n_factors)
        A = (Q.T @ Q).astype(np.float32) + lam_p * np.eye(self.n_factors, dtype=np.float32)
        b = (Q.T @ y).astype(np.float32)
        p_u = np.linalg.solve(A, b).astype(self.qi.dtype)

        if self.pu is not None and self.pu.size:
            tau_cap = float(np.median(np.linalg.norm(self.pu, axis=1)))
        else:
            qnorm90 = float(np.percentile(np.linalg.norm(self.qi, axis=1), 90))
            target_dot = 0.9 * (self.rating_max - self.rating_min)
            tau_cap = target_dot / max(qnorm90, 1e-6)
        norm_p = float(np.linalg.norm(p_u))
        if norm_p > 0 and norm_p > tau_cap:
            p_u = (p_u * (tau_cap / norm_p)).astype(self.qi.dtype)

        lam_b = self.reg_bu * m
        resid = y - Q @ p_u
        b_u = float(resid.sum() / (lam_b + m))

        self._user_cache[uid_raw] = (p_u, b_u)
        
    def _ensure_user_row(self, uid_raw: str) -> int:
        if self.pu is None:
            self.pu = np.zeros((0, self.n_factors), dtype=self.qi.dtype)
            self.bu = np.zeros((0,), dtype=np.float32)

        iu = self.user_index.get(uid_raw)
        if iu is not None:
            return iu
        
        iu = self.pu.shape[0]

        self.pu = np.vstack([self.pu, np.zeros((1, self.n_factors), dtype=self.qi.dtype)])
        self.bu = np.hstack([self.bu, np.zeros((1,), dtype=np.float32)])
        self.user_ids.append(uid_raw)
        self.user_index[uid_raw] = iu

        return iu


    def adjust_model_for_user(self, new_ratings_set, uid: int, username: str):
        pairs = [(i, float(r)) for (_u, i, r) in new_ratings_set]
        self._fold_in_from_pairs(username, pairs)
        return self


    def update_algo(self, username: str, user_data: list[dict]) -> "Model":
        pairs = []

        for item in user_data:
            ii = self.item_index.get(item["movie_id"])
            if ii is not None:
                pairs.append((ii, float(item["rating_val"])))

        self._fold_in_from_pairs(username, pairs)
        return self

    def debug_foldin_user(self, username: str, user_data: list[dict], candidate_ids=None, n_show=10):
        pairs = []
        for it in user_data:
            v = float(it["rating_val"])
            if self.rating_min <= v <= self.rating_max:
                ii = self.item_index.get(it["movie_id"])
                if ii is not None:
                    pairs.append((ii, float(it["rating_val"])))

        # fold-in (reuses your current impl)
        self._fold_in_from_pairs(username, pairs)
        p_u, b_u = self._user_cache[username]

        # scales
        q_norms = np.linalg.norm(self.qi, axis=1)
        dot_all = self.qi @ p_u
        bi = self.bi
        mu = self.mu

        # optional candidate restriction
        if candidate_ids is not None:
            mask = np.array([mid in self.item_index for mid in candidate_ids], dtype=bool)
            I = np.fromiter((self.item_index[mid] for mid in candidate_ids if mid in self.item_index), dtype=np.int64)
        else:
            I = np.arange(self.qi.shape[0], dtype=np.int64)

        s = mu + b_u + bi[I] + dot_all[I]

        def pct(a, ps=[5, 25, 50, 75, 95]):
            return {p: float(np.percentile(a, p)) for p in ps}

        stats = {
            "m_user_ratings": len(pairs),
            "p_u_norm": float(np.linalg.norm(p_u)),
            "b_u": float(b_u),
            "bi_percentiles": pct(bi[I]),
            "dot_percentiles": pct(dot_all[I]),
            "score_percentiles_unclipped": pct(s),
            "mu": float(mu),
            "q_norm_percentiles": pct(q_norms[I]),
        }

        # show top/bottom few for quick smell test
        top_idx = I[np.argsort(-s)[:n_show]]
        bot_idx = I[np.argsort(s)[:n_show]]
        top = [(self.item_ids[i], float(s[I.tolist().index(i)] if candidate_ids else s[np.where(I==i)[0][0]])) for i in top_idx]
        bot = [(self.item_ids[i], float(s[I.tolist().index(i)] if candidate_ids else s[np.where(I==i)[0][0]])) for i in bot_idx]

        return stats, {"top": top, "bottom": bot}



