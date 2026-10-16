"""Estimators of conditional weight of evidence and grouped score attribution for GBDTs."""
from __future__ import annotations

import os
from math import factorial

import numpy as np
import xgboost as xgb
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

EPS = 1e-4
# Thread count for XGBoost/LightGBM/CatBoost; override with the GBDT_WOE_THREADS environment variable.
NTHREAD = int(os.environ.get("GBDT_WOE_THREADS", min(10, os.cpu_count() or 1)))
BASE = dict(tree_method="hist", max_depth=4, eta=0.05, subsample=0.8, min_child_weight=5,
            reg_lambda=1.0, nthread=NTHREAD, verbosity=0)


def cols(blocks, mask):
    return sorted(c for k, b in enumerate(blocks) if mask >> k & 1 for c in b)


def fit_xgb(X, y, seed, objective="binary:logistic", rounds=3000, extra=None, val_frac=0.2):
    """Fit with early stopping on an internal random split of the supplied sample."""
    rng = np.random.default_rng(seed)
    val = rng.random(len(y)) < val_frac
    params = dict(BASE, objective=objective, seed=seed, **(extra or {}))
    if objective == "multi:softprob":
        params["eval_metric"] = "mlogloss"
    elif objective == "reg:logistic":
        params["eval_metric"] = "rmse"
    else:
        params["eval_metric"] = "logloss"
    dtr, dva = xgb.DMatrix(X[~val], label=y[~val]), xgb.DMatrix(X[val], label=y[val])
    bst = xgb.train(params, dtr, rounds, evals=[(dva, "val")], early_stopping_rounds=50, verbose_eval=False)
    return bst.copy() if bst.best_iteration is None else bst[: bst.best_iteration + 1]


class Model:
    """A GBDT on a column subset with a Platt recalibration of its margin."""

    def __init__(self, columns, booster, a=0.0, b=1.0):
        self.columns, self.booster, self.a, self.b = list(columns), booster, a, b

    def margin(self, X):
        return self.booster.predict(xgb.DMatrix(X[:, self.columns]), output_margin=True)

    def logit(self, X):
        return self.a + self.b * self.margin(X)

    def prob(self, X):
        return np.clip(expit(self.logit(X)), EPS, 1 - EPS)

    def calibrate(self, X, y):
        m = LogisticRegression(C=1e6, max_iter=1000).fit(self.margin(X)[:, None], y)
        self.a, self.b = float(m.intercept_[0]), float(m.coef_[0, 0])
        return self

    def leaves(self, X):
        return self.booster.predict(xgb.DMatrix(X[:, self.columns]), pred_leaf=True).reshape(len(X), -1).astype(np.int64)


def fit_model(X, y, columns, seed, Xcal=None, ycal=None, **kw):
    m = Model(columns, fit_xgb(X[:, columns], y, seed, **kw))
    return m.calibrate(Xcal, ycal) if Xcal is not None else m


def clip_logit(p):
    return logit(np.clip(p, EPS, 1 - EPS))


# ---------------------------------------------------------------- block bins
class BlockBinner:
    """Coarse classes for a block: quantiles of a logistic block index fitted on development data."""

    def __init__(self, design, n_bins=5):
        self.design, self.n_bins = design, n_bins
        self.coef, self.icpt, self.edges = {}, {}, {}

    def fit(self, xdict, y, J):
        for j in range(J):
            D = self.design(xdict, j)
            mu, sd = D.mean(0), D.std(0) + 1e-12
            m = LogisticRegression(C=1.0, max_iter=2000).fit((D - mu) / sd, y)
            self.coef[j] = m.coef_[0] / sd
            self.icpt[j] = m.intercept_[0] - (m.coef_[0] * mu / sd).sum()
            score = D @ self.coef[j] + self.icpt[j]
            e = np.unique(np.quantile(score, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
            # drop edges that leave an empty or near-empty class on the development sample
            keep = [t for t in e if 0.01 < (score <= t).mean() < 0.99]
            self.edges[j] = np.array(keep)
        return self

    def score(self, xdict, j):
        return self.design(xdict, j) @ self.coef[j] + self.icpt[j]

    def __call__(self, xdict, j):
        return np.searchsorted(self.edges[j], self.score(xdict, j), side="left")

    def nbins(self, j):
        return len(self.edges[j]) + 1


# ---------------------------------------------------------------- leaf-local evidence
class LeafTally:
    """Class-by-bin counts in the leaves of a feature-excluded ensemble.

    For query x the pseudo-count n_{yb}(x) = T^{-1} sum_t #{i : L_t(x_i) = L_t(x), Y_i = y, b_i = b}
    is the kernel-weighted count with K(x, x_i) equal to the fraction of trees in which x and x_i
    share a leaf. Its scale is that of an ordinary count of estimation records."""

    def __init__(self, leaves_ev, y, b, nb):
        self.nb = nb
        n, T = leaves_ev.shape
        self.T = T
        self.width = leaves_ev.max() + 1
        idx = (np.arange(T)[None, :] * self.width + leaves_ev) * (2 * nb) + (y * nb + b)[:, None]
        self.tab = np.bincount(idx.ravel(), minlength=T * self.width * 2 * nb).reshape(T, self.width, 2, nb)
        self.glob = np.array([np.bincount(b[y == c], minlength=nb) / (y == c).sum() for c in (0, 1)])
        self.leaves_ev, self.y = leaves_ev, y

    def counts(self, leaves_q):
        lq = np.minimum(leaves_q, self.width - 1)
        return self.tab[np.arange(self.T)[None, :], lq].mean(1)   # (Q, 2, nb)

    def woe(self, leaves_q, bq, lam):
        n = self.counts(leaves_q)
        ny = n.sum(2)
        q = self.glob
        nb_ = n[np.arange(len(bq)), :, bq]                       # (Q, 2)
        p = (nb_ + lam * q[:, bq].T) / (ny + lam)
        return np.log(p[:, 1] / p[:, 0]), ny

    def heldout_loglik(self, leaves_q, bq, yq, lam):
        n = self.counts(leaves_q)
        ny = n.sum(2)
        r = np.arange(len(bq))
        p = (n[r, yq, bq] + lam * self.glob[yq, bq]) / (ny[r, yq] + lam)
        return np.log(p).mean()

    def global_woe(self, bq):
        return np.log(self.glob[1, bq] / self.glob[0, bq])

    def ess(self, leaves_q):
        """Class-specific effective sample sizes (sum K)^2 / sum K^2."""
        T = self.T
        out = np.empty((len(leaves_q), 2))
        for a in range(0, len(leaves_q), 200):
            lq = leaves_q[a:a + 200]
            K = np.zeros((len(lq), len(self.y)), dtype=np.float32)
            for t in range(T):
                K += (lq[:, t:t + 1] == self.leaves_ev[None, :, t])
            K /= T
            for c in (0, 1):
                Kc = K[:, self.y == c]
                out[a:a + 200, c] = Kc.sum(1) ** 2 / np.maximum((Kc ** 2).sum(1), 1e-12)
        return out


class KernelEvidence:
    """Leaf-proximity weights a_i(x) proportional to K(x, x_i)^kappa, computed on the GPU.

    K(x, x') is the fraction of trees in which x and x' share a leaf (equal tree weights).
    kappa = 1 is the proximity average; larger kappa concentrates weight on records that share
    leaves in many trees. Pseudo-counts n_{yb}(x) = sum_i K(x, x_i)^kappa 1{Y_i = y, b_i = b}
    are on the scale of estimation-record counts."""

    def __init__(self, leaves_ev, y, b, nb, device=None):
        import torch
        if device is None:
            device = os.environ.get("GBDT_WOE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        # Leaf co-membership counts are small integers, exact in float16; CPU matmul uses float32.
        self.dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self.torch, self.dev, self.nb = torch, device, nb
        n, T = leaves_ev.shape
        self.T = T
        self.lut, off = [], 0
        for t in range(T):
            u = np.unique(leaves_ev[:, t])
            lut = np.full(leaves_ev[:, t].max() + 64, -1, dtype=np.int64)
            lut[u] = off + np.arange(len(u))
            self.lut.append(lut)
            off += len(u)
        self.L = off
        self.Ae = self._onehot(leaves_ev)
        Yb = np.zeros((n, 2 * nb), dtype=np.float32)
        Yb[np.arange(n), y * nb + b] = 1
        self.Yb = torch.tensor(Yb, device=device)
        self.Yc = torch.tensor(np.stack([y == 0, y == 1], 1).astype(np.float32), device=device)
        self.glob = np.array([np.bincount(b[y == c], minlength=nb) / (y == c).sum() for c in (0, 1)])

    def _onehot(self, leaves):
        torch = self.torch
        cols = np.stack([self.lut[t][np.minimum(leaves[:, t], len(self.lut[t]) - 1)] for t in range(self.T)], 1)
        A = torch.zeros((len(leaves), self.L), dtype=self.dtype, device=self.dev)
        r, c = np.nonzero(cols >= 0)
        A[torch.tensor(r, device=self.dev), torch.tensor(cols[r, c], device=self.dev)] = 1
        return A

    def stats(self, leaves_q, kappas, chunk=512):
        """Returns n (len(kappas), Q, 2, nb) and ess (len(kappas), Q, 2)."""
        torch = self.torch
        Q = len(leaves_q)
        n = np.empty((len(kappas), Q, 2, self.nb))
        ess = np.empty((len(kappas), Q, 2))
        for a in range(0, Q, chunk):
            Aq = self._onehot(leaves_q[a:a + chunk])
            K = (Aq @ self.Ae.T).float() / self.T
            for i, kap in enumerate(kappas):
                Wk = K ** kap
                n[i, a:a + chunk] = (Wk @ self.Yb).reshape(-1, 2, self.nb).cpu().numpy()
                s1 = (Wk @ self.Yc).cpu().numpy()
                s2 = ((Wk * Wk) @ self.Yc).cpu().numpy()
                ess[i, a:a + chunk] = s1 ** 2 / np.maximum(s2, 1e-12)
            del K, Aq
        return n, ess

    def woe(self, n, bq, lam):
        ny = n.sum(-1)
        r = np.arange(len(bq))
        p = (n[r, :, bq] + lam * self.glob[:, bq].T) / (ny + lam)
        return np.log(p[:, 1] / p[:, 0])

    def loglik(self, n, bq, yq, lam):
        ny = n.sum(-1)
        r = np.arange(len(bq))
        p = (n[r, yq, bq] + lam * self.glob[yq, bq]) / (ny[r, yq] + lam)
        # class-balanced: the evidence ratio depends on both class-conditional distributions
        return 0.5 * (np.log(p[yq == 0]).mean() + np.log(p[yq == 1]).mean())

    def global_woe(self, bq):
        return np.log(self.glob[1, bq] / self.glob[0, bq])


# ---------------------------------------------------------------- Shapley
def shapley_matrix(J):
    """S with phi = V @ S.T for V indexed by bitmask over J players."""
    M = np.zeros((J, 1 << J))
    for r in range(J):
        for mask in range(1 << J):
            if mask >> r & 1:
                continue
            s = bin(mask).count("1")
            w = factorial(s) * factorial(J - s - 1) / factorial(J)
            M[r, mask | (1 << r)] += w
            M[r, mask] -= w
    return M


def hybrids(x, refs, blocks):
    """All 2^J block hybrids of applicant x (blocks in mask) and each reference row."""
    J = len(blocks)
    H = np.repeat(refs[:, None, :], 1 << J, axis=1).copy()
    for mask in range(1 << J):
        for k in range(J):
            if mask >> k & 1:
                H[:, mask, list(blocks[k])] = x[list(blocks[k])]
    return H   # (R, 2^J, p)


def baseline_shapley(score_fn, x, refs, blocks, S=None):
    """Exact grouped baseline Shapley of score_fn(x) - score_fn(r) for each reference r."""
    J = len(blocks)
    S = shapley_matrix(J) if S is None else S
    H = hybrids(x, refs, blocks)
    v = score_fn(H.reshape(-1, H.shape[-1])).reshape(len(refs), 1 << J)
    v = v - v[:, :1]
    return v @ S.T, v[:, -1]   # (R, J), gaps
