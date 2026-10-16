"""Calibration of the fitted models: against outcomes and, in the simulation, against true probabilities.

usage: calib_check.py sim SEED ... | heloc REP ...
Appends to results/calibration.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression, LinearRegression

import bureau_sim as bs
import estimators as es

RES = Path(__file__).resolve().parent / "results"


def outcome_cal(lp, y):
    """Calibration intercept (offset) and slope, calibration-in-the-large, decile ECE."""
    p = expit(lp)
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(lp[:, None], y)
    # calibration intercept with slope fixed at 1: solve by Newton on the offset
    a = 0.0
    for _ in range(50):
        q = expit(lp + a)
        a += (y - q).sum() / (q * (1 - q)).sum()
    dec = np.quantile(p, np.linspace(0, 1, 11))
    g = np.clip(np.digitize(p, dec[1:-1]), 0, 9)
    ece = sum(abs(p[g == k].mean() - y[g == k].mean()) * (g == k).mean() for k in range(10))
    return dict(mean_pred=float(p.mean()), mean_obs=float(y.mean()), cal_intercept=float(a),
                cal_slope=float(slope.coef_[0, 0]), ece=float(ece),
                deciles=[[float(p[g == k].mean()), float(y[g == k].mean())] for k in range(10)])


def truth_cal(lp, lt):
    """Agreement of fitted with true log odds: mean bias, slope of true on fitted, RMSE; and on probability scale."""
    lr = LinearRegression().fit(lp[:, None], lt)
    return dict(bias_logit=float((lp - lt).mean()), slope_true_on_fit=float(lr.coef_[0]),
                rmse_logit=float(np.sqrt(((lp - lt) ** 2).mean())),
                bias_prob=float((expit(lp) - expit(lt)).mean()),
                mean_abs_prob=float(np.abs(expit(lp) - expit(lt)).mean()))


def sim(seed):
    from run_revision import data
    r = np.load(RES / f"sim_seed{seed}.npz")
    D = data(seed)
    Xd, yd = D["dev"][:2]
    Xc, yc = D["cal"][:2]
    Xt, yt, pt = D["test"][:3]
    J, blocks, allm = bs.J, bs.BLOCKS, (1 << bs.J) - 1
    lt_true = logit(pt)
    out = {}
    F = es.fit_model(Xd, yd, list(range(Xd.shape[1])), seed, Xc, yc)
    out["refit_maxdiff"] = float(np.abs(F.logit(Xt[r["qsel"]]) - r["lF"]).max())
    raw = F.margin(Xt)
    lf = F.logit(Xt)
    out["full"] = dict(platt_a=F.a, platt_b=F.b, raw_outcome=outcome_cal(raw, yt), outcome=outcome_cal(lf, yt),
                       truth=truth_cal(lf, lt_true))
    cut = r["cut"]
    dec = lf > cut
    out["full"]["declined"] = dict(n=int(dec.sum()), mean_pred=float(expit(lf[dec]).mean()),
                                   mean_true=float(pt[dec].mean()), mean_obs=float(yt[dec].mean()),
                                   truth=truth_cal(lf[dec], lt_true[dec]))
    near = np.abs(lf - cut) < 0.3
    out["full"]["near_cutoff"] = dict(n=int(near.sum()), mean_pred=float(expit(lf[near]).mean()),
                                      mean_true=float(pt[near].mean()), mean_obs=float(yt[near].mean()))
    out["reduced"] = []
    for j in range(J):
        R = es.fit_model(Xd, yd, es.cols(blocks, allm & ~(1 << j)), seed + 1 + j, Xc, yc)
        lr_ = R.logit(Xt)
        out["reduced"].append(dict(platt_a=R.a, platt_b=R.b, outcome=outcome_cal(lr_, yt)))
    # block-excluded posteriors against the oracle P(Y=1 | x_-j), on the evaluated queries
    lq_true_red = r["eta_q"][:, None] - r["W_true"]
    lq_dir = r["lF"][:, None] - r["W_dir"]
    lq_proj = r["lF"][:, None] - r["W_proj"]
    out["reduced_vs_oracle"] = [dict(direct=truth_cal(lq_dir[:, j], lq_true_red[:, j]),
                                     projection=truth_cal(lq_proj[:, j], lq_true_red[:, j])) for j in range(J)]
    return out


def heloc(rep):
    import run_heloc as rh
    rng = np.random.default_rng(1000 + rep)
    X, y = rh.load()
    perm = rng.permutation(len(y))
    idx, a = {}, 0
    for k, f in rh.FRAC.items():
        m = int(round(f * len(y))) if k != "test" else len(y) - a
        idx[k] = perm[a:a + m]
        a += m
    seed = 5000 + 100 * rep
    F = es.fit_model(X[idx["dev"]], y[idx["dev"]], list(range(X.shape[1])), seed, X[idx["cal"]], y[idx["cal"]])
    r = np.load(RES / f"heloc_rep{rep}.npz")
    Xt, yt = X[idx["test"]], y[idx["test"]]
    return dict(refit_maxdiff=float(np.abs(F.logit(Xt) - r["lF"]).max()), platt_a=F.a, platt_b=F.b,
                raw_outcome=outcome_cal(F.margin(Xt), yt), outcome=outcome_cal(F.logit(Xt), yt))


if __name__ == "__main__":
    kind, ids = sys.argv[1], [int(s) for s in sys.argv[2:]]
    path = RES / "calibration.json"
    allres = json.load(open(path)) if path.exists() else {}
    for i in ids:
        allres[f"{kind}{i}"] = sim(i) if kind == "sim" else heloc(i)
        print(kind, i, "done", flush=True)
    path.write_text(json.dumps(allres, indent=1))
