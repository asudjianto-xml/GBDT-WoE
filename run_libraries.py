"""Library comparison on the bureau-style simulation: XGBoost, LightGBM and CatBoost.

usage: run_libraries.py SEED
Reuses the data, queries, declined applicants and score-band references of run_sim.py for the
same seed (read from results/sim_seed{SEED}.npz) and fits full and block-excluded models with
each library. Reports direct conditional evidence and grouped baseline-Shapley attribution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import lightgbm as lgb
import catboost as cb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from scipy.special import expit

import bureau_sim as bs
import estimators as es
from run_sim import SPLITS, BAND, N_REF

OUT = Path(__file__).resolve().parent / "results"


class Lib:
    def __init__(self, kind, columns, seed):
        self.kind, self.columns, self.seed, self.a, self.b = kind, list(columns), seed, 0.0, 1.0

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        val = rng.random(len(y)) < 0.2
        Xc = X[:, self.columns]
        if self.kind == "lightgbm":
            self.m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05, num_leaves=15, max_depth=4,
                                        subsample=0.8, subsample_freq=1, min_child_samples=20,
                                        random_state=self.seed, n_jobs=es.NTHREAD, verbose=-1)
            self.m.fit(Xc[~val], y[~val], eval_set=[(Xc[val], y[val])],
                       callbacks=[lgb.early_stopping(50, verbose=False)])
        else:
            self.m = cb.CatBoostClassifier(iterations=3000, learning_rate=0.05, depth=4, random_seed=self.seed,
                                           thread_count=es.NTHREAD, verbose=False, early_stopping_rounds=50)
            self.m.fit(Xc[~val], y[~val], eval_set=(Xc[val], y[val]))
        return self

    def margin(self, X):
        Xc = X[:, self.columns]
        if self.kind == "lightgbm":
            return self.m.predict(Xc, raw_score=True)
        return self.m.predict(Xc, prediction_type="RawFormulaVal")

    def calibrate(self, X, y):
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(self.margin(X)[:, None], y)
        self.a, self.b = float(lr.intercept_[0]), float(lr.coef_[0, 0])
        return self

    def logit(self, X):
        return self.a + self.b * self.margin(X)


def main(seed):
    r = np.load(OUT / f"sim_seed{seed}.npz")
    rng = np.random.default_rng(seed)
    X, y, ptrue, booked = bs.generate(sum(SPLITS.values()), rng)
    o = np.cumsum([0] + list(SPLITS.values()))
    sl = {k: slice(o[i], o[i + 1]) for i, k in enumerate(SPLITS)}
    Xd, yd, Xc, yc = X[sl["dev"]], y[sl["dev"]], X[sl["cal"]], y[sl["cal"]]
    Xr, Xt, yt = X[sl["ref"]], X[sl["test"]], y[sl["test"]]
    Xq, Xa = Xt[r["qsel"]], Xt[r["dsel"]]
    J, blocks, allm = bs.J, bs.BLOCKS, (1 << bs.J) - 1
    S = es.shapley_matrix(J)
    W_true = r["W_true"]
    out, arrays = {}, {}
    for kind in ("lightgbm", "catboost"):
        F = Lib(kind, range(X.shape[1]), seed).fit(Xd, yd).calibrate(Xc, yc)
        R = [Lib(kind, es.cols(blocks, allm & ~(1 << j)), seed + 1 + j).fit(Xd, yd).calibrate(Xc, yc) for j in range(J)]
        lt = F.logit(Xt)
        lq = F.logit(Xq)
        W = np.stack([lq - R[j].logit(Xq) for j in range(J)], 1)
        # same applicants and score-band references as the XGBoost study; cutoff at the same approval rate
        refs = r["refs_score"]
        cut = np.quantile(F.logit(Xr), 0.80)
        C = np.empty((len(Xa), J))
        for i in range(len(Xa)):
            phi, _ = es.baseline_shapley(F.logit, Xa[i], Xr[refs[i]], blocks, S)
            C[i] = phi.mean(0)
        arrays[f"{kind}_W"], arrays[f"{kind}_C"] = W, C
        out[kind] = dict(auc=roc_auc_score(yt, lt), logloss=log_loss(yt, expit(lt)),
                         rmse=np.sqrt(((W - W_true) ** 2).mean(0)).tolist(),
                         declined_by_this_model=float(np.mean(F.logit(Xa) > cut)),
                         refs_passing=float(np.mean(F.logit(Xr[refs.ravel()]) <= cut)))
        print(seed, kind, out[kind]["auc"], flush=True)
    np.savez_compressed(OUT / f"lib_seed{seed}.npz", **arrays)
    (OUT / f"lib_seed{seed}.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    for s in sys.argv[1:]:
        main(int(s))
