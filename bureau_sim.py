"""Credit-bureau-style simulation with a computable conditional-evidence oracle.

Three correlated latent borrower factors (financial stress s, credit maturity m and
credit seeking q) generate six characteristic blocks that are conditionally independent
given the factors. Default depends only on the observed characteristics through a known
log-odds eta(x), so P(Y=1 | x) = sigmoid(eta(x)) exactly. Conditional probabilities given
a subset of blocks are computed by integrating the remaining blocks over the latent
posterior, which is evaluated on a fine grid because every block likelihood is available
in closed form.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logit, gammaln

FEATURES = [
    "revolving_limit", "revolving_balance", "revolving_utilization",
    "trades_30plus_dpd", "trades_60plus_dpd", "trades_90plus_dpd", "months_since_delinquency",
    "oldest_trade_months", "average_trade_months",
    "inquiries_6m", "new_accounts_6m",
    "installment_original", "installment_balance", "installment_fraction",
    "public_record_flag",
]
BLOCKS = [(0, 1, 2), (3, 4, 5, 6), (7, 8), (9, 10), (11, 12, 13), (14,)]
BLOCK_NAMES = ["Revolving", "Delinquency", "File age", "Credit seeking", "Installment", "Public record"]
J = len(BLOCKS)

SIGMA = np.array([[1.0, -0.20, 0.35], [-0.20, 1.0, -0.15], [0.35, -0.15, 1.0]])
CHOL = np.linalg.cholesky(SIGMA)
INTERCEPT = -5.2
INSTALL_COEF = 0.8      # direct effect of the installment fraction on default log odds
LEGACY_T = 1.55           # legacy approval threshold (about 70% approval)
LEGACY_SD = 0.5

# latent-mean functions; z has trailing axis (s, m, q)
def _means(z):
    s, m, q = z[..., 0], z[..., 1], z[..., 2]
    return dict(
        mu_lim=np.log(14000) + 0.45 * m - 0.15 * s,
        mu_lu=-0.7 + s + 0.3 * q,
        lam_dq=0.25 * np.exp(0.8 * s + 0.25 * q),
        mu_age=np.log(70) + 0.8 * m - 0.15 * s,
        lam_inq=np.exp(0.1 + 0.4 * s + 0.5 * q),
        mu_org=np.log(18000) + 0.4 * m,
        mu_lf=0.1 + 0.5 * s,
        p_pub=expit(-3.3 + 0.8 * s),
    )

SD_LIM, SD_LU, SD_AGE, SD_ORG, SD_LF = 0.4, 0.6, 0.35, 0.4, 0.5


def sample_blocks(z, blocks, rng):
    """Draw the listed blocks given latent z (shape (..., 3)). Returns dict feature->array."""
    mu = _means(z)
    shape = z.shape[:-1]
    out = {}
    if 0 in blocks:
        lim = np.exp(mu["mu_lim"] + SD_LIM * rng.standard_normal(shape))
        u = expit(mu["mu_lu"] + SD_LU * rng.standard_normal(shape))
        out.update(revolving_limit=lim, revolving_balance=lim * u, revolving_utilization=u)
    if 1 in blocks:
        n30 = rng.poisson(mu["lam_dq"]).astype(float)
        n60 = rng.binomial(n30.astype(np.int64), 0.4).astype(float)
        n90 = rng.binomial(n60.astype(np.int64), 0.4).astype(float)
        msd = np.where(n30 > 0, np.minimum(rng.geometric(0.08, shape), 60), 999).astype(float)
        out.update(trades_30plus_dpd=n30, trades_60plus_dpd=n60, trades_90plus_dpd=n90,
                   months_since_delinquency=msd)
    if 2 in blocks:
        oldest = 6 + np.exp(mu["mu_age"] + SD_AGE * rng.standard_normal(shape))
        avg = oldest * (0.25 + 0.55 * rng.beta(3, 2, shape))
        out.update(oldest_trade_months=oldest, average_trade_months=avg)
    if 3 in blocks:
        inq = rng.poisson(mu["lam_inq"]).astype(float)
        new = rng.binomial(inq.astype(np.int64), 0.35).astype(float)
        out.update(inquiries_6m=inq, new_accounts_6m=new)
    if 4 in blocks:
        org = np.exp(mu["mu_org"] + SD_ORG * rng.standard_normal(shape))
        frac = expit(mu["mu_lf"] + SD_LF * rng.standard_normal(shape))
        out.update(installment_original=org, installment_balance=org * frac, installment_fraction=frac)
    if 5 in blocks:
        out.update(public_record_flag=rng.binomial(1, mu["p_pub"]).astype(float))
    return out


def _lognorm(x, mu, sd):
    return -0.5 * ((x - mu) / sd) ** 2


def block_loglik(x, z):
    """Log p(x_k | z) up to z-free constants, for each block k.

    x: dict of arrays of shape (Q,); z: (G, 3). Returns array (Q, J, G).
    """
    mu = _means(z)
    Q, G = len(x["revolving_limit"]), z.shape[0]
    ll = np.empty((Q, J, G))
    c = lambda a: a[:, None]
    ll[:, 0] = (_lognorm(c(np.log(x["revolving_limit"])), mu["mu_lim"], SD_LIM)
                + _lognorm(c(logit(np.clip(x["revolving_utilization"], 1e-12, 1 - 1e-12))), mu["mu_lu"], SD_LU))
    n30 = c(x["trades_30plus_dpd"])
    ll[:, 1] = n30 * np.log(mu["lam_dq"]) - mu["lam_dq"]
    ll[:, 2] = _lognorm(c(np.log(x["oldest_trade_months"] - 6)), mu["mu_age"], SD_AGE)
    inq = c(x["inquiries_6m"])
    ll[:, 3] = inq * np.log(mu["lam_inq"]) - mu["lam_inq"]
    ll[:, 4] = (_lognorm(c(np.log(x["installment_original"])), mu["mu_org"], SD_ORG)
                + _lognorm(c(logit(x["installment_fraction"])), mu["mu_lf"], SD_LF))
    pub = c(x["public_record_flag"])
    ll[:, 5] = pub * np.log(mu["p_pub"]) + (1 - pub) * np.log1p(-mu["p_pub"])
    return ll


def thin_file(oldest):
    return expit((48.0 - oldest) / 8.0)


def eta(x):
    """True default log-odds given the observed record."""
    u = x["revolving_utilization"]
    thin = thin_file(x["oldest_trade_months"])
    n30, n60, n90 = x["trades_30plus_dpd"], x["trades_60plus_dpd"], x["trades_90plus_dpd"]
    recent = (n30 > 0) * np.exp(-np.minimum(x["months_since_delinquency"], 60) / 12.0)
    return (INTERCEPT + (2.6 - 3.4 * thin) * u + 1.0 * u ** 2
            + 0.30 * n30 + 0.40 * n60 + 0.60 * n90 + 0.8 * recent
            + 0.30 * np.log1p(x["inquiries_6m"]) + 0.20 * x["new_accounts_6m"]
            + 0.9 * x["public_record_flag"] + INSTALL_COEF * x["installment_fraction"]
            - 0.25 * np.log(x["revolving_limit"] / 14000.0) + 0.6 * thin
            + 1.0 * u * (n60 > 0))


def legacy_score(x):
    """Score of an earlier approval policy; applicants with legacy < LEGACY_T are booked."""
    return (2.0 * x["revolving_utilization"] + 0.5 * x["trades_30plus_dpd"]
            + 0.8 * x["public_record_flag"] + 0.3 * np.log1p(x["inquiries_6m"]))


def approve_prob(x):
    from scipy.stats import norm
    return norm.cdf((LEGACY_T - legacy_score(x)) / LEGACY_SD)


def generate(n, rng):
    z = rng.standard_normal((n, 3)) @ CHOL.T
    x = sample_blocks(z, set(range(J)), rng)
    p = expit(eta(x))
    y = rng.binomial(1, p)
    booked = rng.random(n) < approve_prob(x)
    X = np.column_stack([x[f] for f in FEATURES])
    return X, y, p, booked


def to_dict(X):
    return {f: X[:, i] for i, f in enumerate(FEATURES)}


# ---------------------------------------------------------------- block bins
# Each block is coarse-classed by quantiles of a one-dimensional block index, the linear
# predictor of a logistic regression of default on transformed block fields fitted on the
# development sample. The transforms are fixed before fitting.

def block_design(x, j):
    if j == 0:
        u = x["revolving_utilization"]
        return np.stack([u, u ** 2, np.log(x["revolving_limit"])], -1)
    if j == 1:
        rec = (x["trades_30plus_dpd"] > 0) * np.exp(-np.minimum(x["months_since_delinquency"], 60) / 12.0)
        return np.stack([x["trades_30plus_dpd"], x["trades_60plus_dpd"], x["trades_90plus_dpd"], rec], -1)
    if j == 2:
        return np.stack([np.log(x["oldest_trade_months"]), np.log(x["average_trade_months"])], -1)
    if j == 3:
        return np.stack([np.log1p(x["inquiries_6m"]), x["new_accounts_6m"]], -1)
    if j == 4:
        return np.stack([x["installment_fraction"], np.log(x["installment_original"])], -1)
    return x["public_record_flag"][..., None]


# ---------------------------------------------------------------- oracle
GRID_N = 34
GRID_LIM = 4.3


def latent_grid(n=None, lim=None):
    n, lim = n or GRID_N, lim or GRID_LIM
    g = np.linspace(-lim, lim, n)
    U = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    return U, g[1] - g[0], -0.5 * (U ** 2).sum(1)


class Oracle:
    """Monte Carlo integration of P(Y=1 | x_S) through the latent posterior on a grid."""

    def __init__(self, M=8192, seed=0, binner=None, grid_n=None, grid_lim=None):
        self.U, self.h, self.logprior = latent_grid(grid_n, grid_lim)
        self.Z = self.U @ CHOL.T
        self.M = M
        self.rng = np.random.default_rng(seed)
        self.binner = binner   # callable (x_dict, j) -> bin index array

    def _draw_latent(self, logw):
        """logw: (Q, G). Returns z draws (Q, M, 3) with uniform jitter inside the grid cell."""
        logw = logw - logw.max(1, keepdims=True)
        w = np.exp(logw)
        cdf = np.cumsum(w, 1)
        cdf /= cdf[:, -1:]
        r = self.rng.random((logw.shape[0], self.M))
        idx = np.stack([np.searchsorted(cdf[i], r[i]) for i in range(len(r))])
        idx = np.minimum(idx, logw.shape[1] - 1)
        u = self.U[idx] + (self.rng.random(idx.shape + (3,)) - 0.5) * self.h
        return u @ CHOL.T

    def _compose(self, xq, drawn):
        out = {}
        for f in FEATURES:
            out[f] = drawn[f] if f in drawn else np.broadcast_to(xq[f][:, None], (len(xq[f]), self.M))
        return out

    def leave_one_out(self, X, chunk=24):
        """Per query and block j: E[sig], E[sig*s], E[s], E[1_B sig], E[1_B (1-sig)], E[1_B],
        all conditional on x_{-j} with X_j integrated out. Returns dict of (Q, J) arrays."""
        x = to_dict(X)
        Q = X.shape[0]
        keys = ["sig", "sig_s", "s", "b_sig", "b_nsig", "b", "b_sig_s", "b_nsig_s", "b_s"]
        res = {k: np.empty((Q, J)) for k in keys}
        own_bin = np.stack([self.binner(x, j) for j in range(J)], 1) if self.binner else None
        for a in range(0, Q, chunk):
            sl = slice(a, min(a + chunk, Q))
            xq = {f: v[sl] for f, v in x.items()}
            ll = block_loglik(xq, self.Z)
            tot = ll.sum(1)
            for j in range(J):
                z = self._draw_latent(self.logprior[None] + tot - ll[:, j])
                xx = self._compose(xq, sample_blocks(z, {j}, self.rng))
                sig = expit(eta(xx))
                s = approve_prob(xx)
                res["sig"][sl, j] = sig.mean(1)
                res["sig_s"][sl, j] = (sig * s).mean(1)
                res["s"][sl, j] = s.mean(1)
                if self.binner is not None:
                    ib = (self.binner(xx, j) == own_bin[sl, j][:, None])
                    res["b_sig"][sl, j] = (ib * sig).mean(1)
                    res["b_nsig"][sl, j] = (ib * (1 - sig)).mean(1)
                    res["b"][sl, j] = ib.mean(1)
                    res["b_sig_s"][sl, j] = (ib * sig * s).mean(1)
                    res["b_nsig_s"][sl, j] = (ib * (1 - sig) * s).mean(1)
                    res["b_s"][sl, j] = (ib * s).mean(1)
        return res

    def coalition(self, X, chunk=16):
        """E[sig | x_S] for every subset S of blocks (bitmask), shape (Q, 2^J).
        The full set uses sigmoid(eta(x)); the empty set is left as NaN (use pi)."""
        x = to_dict(X)
        Q = X.shape[0]
        out = np.full((Q, 1 << J), np.nan)
        out[:, (1 << J) - 1] = expit(eta(x))
        for a in range(0, Q, chunk):
            sl = slice(a, min(a + chunk, Q))
            xq = {f: v[sl] for f, v in x.items()}
            ll = block_loglik(xq, self.Z)
            for mask in range(1, (1 << J) - 1):
                inS = [k for k in range(J) if mask >> k & 1]
                miss = {k for k in range(J) if not mask >> k & 1}
                z = self._draw_latent(self.logprior[None] + ll[:, inS].sum(1))
                xx = self._compose(xq, sample_blocks(z, miss, self.rng))
                out[sl, mask] = expit(eta(xx)).mean(1)
        return out


_SHARED = {}


def _run_part(args):
    method, X, seed = args
    o = _SHARED["oracle"]
    o.rng = np.random.default_rng(seed)
    return getattr(o, method)(X)


def parallel(oracle, method, X, n_jobs=16, seed=0):
    """Run oracle.<method> on query blocks in forked worker processes."""
    import multiprocessing as mp
    _SHARED["oracle"] = oracle
    parts = [p for p in np.array_split(np.arange(len(X)), n_jobs) if len(p)]
    with mp.get_context("fork").Pool(len(parts)) as pool:
        res = pool.map(_run_part, [(method, X[p], seed + 7919 * i) for i, p in enumerate(parts)])
    if isinstance(res[0], dict):
        return {k: np.concatenate([r[k] for r in res]) for k in res[0]}
    return np.concatenate(res)


def population_rate(rng, n=4_000_000):
    tot = 0.0
    for _ in range(n // 500_000):
        z = rng.standard_normal((500_000, 3)) @ CHOL.T
        tot += expit(eta(sample_blocks(z, set(range(J)), rng))).sum()
    return tot / n
