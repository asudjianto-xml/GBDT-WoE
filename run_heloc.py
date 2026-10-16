"""One random-split replicate of the FICO HELOC study (OpenML 45023).

usage: run_heloc.py REP
Default is RiskPerformance == 0 ("Bad": 90+ days past due within the performance window).
Records with no bureau information (all fields -9) are removed. ExternalRiskEstimate is a
consolidated bureau risk score rather than a credit characteristic and is excluded.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, log_loss

import estimators as es

OUT = Path(__file__).resolve().parent / "results"
GROUPS = {
    "Delinquency": ["NumTrades60Ever2DerogPubRec", "NumTrades90Ever2DerogPubRec", "PercentTradesNeverDelq",
                    "MSinceMostRecentDelq", "MaxDelq2PublicRecLast12M"],
    "File age": ["MSinceOldestTradeOpen", "AverageMInFile", "MSinceMostRecentTradeOpen"],
    "Trade volume": ["NumSatisfactoryTrades", "NumTotalTrades", "NumTradesOpeninLast12M", "PercentInstallTrades"],
    "Inquiries": ["MSinceMostRecentInqexcl7days", "NumInqLast6M", "NumInqLast6Mexcl7days"],
    "Revolving": ["NetFractionRevolvingBurden", "NumRevolvingTradesWBalance",
                  "NumBank2NatlTradesWHighUtilization", "PercentTradesWBalance"],
    "Installment": ["NetFractionInstallBurden", "NumInstallTradesWBalance"],
}
NAMES = list(GROUPS)
FEATS = [f for g in GROUPS.values() for f in g]
BLOCKS = []
_a = 0
for g in GROUPS.values():
    BLOCKS.append(tuple(range(_a, _a + len(g))))
    _a += len(g)
J = len(BLOCKS)
FRAC = dict(dev=0.40, cal=0.10, ev=0.25, ref=0.10, test=0.15)
APPROVAL = 0.50
BAND = 0.30
N_REF = 30
N_DECLINE = 400
N_BOOT = 10
LAMBDAS = [0.1, 0.5, 2, 8, 32]
KAPPAS = [1, 2, 4, 8, 16, 32]


def design(X, j):
    """Block design for coarse classing: nonnegative value and special-value indicator per field."""
    V = X[..., list(BLOCKS[j])] if not isinstance(X, dict) else X["X"][..., list(BLOCKS[j])]
    return np.concatenate([np.maximum(V, 0), (V < 0).astype(float)], -1)


def load():
    path = OUT / "heloc_openml45023.csv"
    if not path.exists():
        from sklearn.datasets import fetch_openml
        OUT.mkdir(exist_ok=True)
        d = fetch_openml(data_id=45023, as_frame=True).frame
        d["RiskPerformance"] = pd.to_numeric(d["RiskPerformance"])
        d.to_csv(path, index=False)
    d = pd.read_csv(path)
    d = d[~(d.drop(columns="RiskPerformance") == -9).all(1)]
    return d[FEATS].to_numpy(float), (d["RiskPerformance"].to_numpy() == 0).astype(int)


def main(rep):
    rng = np.random.default_rng(1000 + rep)
    X, y = load()
    perm = rng.permutation(len(y))
    idx, a = {}, 0
    for k, f in FRAC.items():
        m = int(round(f * len(y))) if k != "test" else len(y) - a
        idx[k] = perm[a:a + m]
        a += m
    Xd, yd = X[idx["dev"]], y[idx["dev"]]
    Xc, yc = X[idx["cal"]], y[idx["cal"]]
    Xe, ye = X[idx["ev"]], y[idx["ev"]]
    Xr, Xt, yt = X[idx["ref"]], X[idx["test"]], y[idx["test"]]
    allm = (1 << J) - 1
    seed = 5000 + 100 * rep
    summary = dict(rep=rep, n=len(y), default_rate=float(y.mean()))
    binner = es.BlockBinner(lambda x, j: design(x, j), n_bins=4).fit({"X": Xd}, yd, J)
    B = lambda Z, j: binner({"X": Z}, j)

    F = es.fit_model(Xd, yd, list(range(X.shape[1])), seed, Xc, yc)
    lt = F.logit(Xt)
    summary["auc"], summary["logloss"] = roc_auc_score(yt, lt), log_loss(yt, expit(lt))
    R = [es.fit_model(Xd, yd, es.cols(BLOCKS, allm & ~(1 << j)), seed + 1 + j, Xc, yc) for j in range(J)]
    pFe = F.prob(Xe)
    P = {mask: es.Model(es.cols(BLOCKS, mask), es.fit_xgb(Xe[:, es.cols(BLOCKS, mask)], pFe, seed + 100 + mask,
                                                          objective="reg:logistic"))
         for mask in range(1, allm)}
    pi_hat = float(ye.mean())
    predp = lambda mdl, Z: es.clip_logit(mdl.booster.predict(es.xgb.DMatrix(Z[:, mdl.columns])))

    Xq = Xt
    bq = np.stack([B(Xq, j) for j in range(J)], 1)
    lF = F.logit(Xq)
    W_dir = np.stack([lF - R[j].logit(Xq) for j in range(J)], 1)
    W_proj = np.stack([lF - predp(P[allm & ~(1 << j)], Xq) for j in range(J)], 1)
    W_leafY, W_leafX, W_glob = (np.empty((len(Xq), J)) for _ in range(3))
    ess_Y, ny_Y = np.empty((len(Xq), J, 2)), np.empty((len(Xq), J, 2))
    sel = {}
    for j in range(J):
        be, bc, bd, nb = B(Xe, j), B(Xc, j), B(Xd, j), binner.nbins(j)
        GX = es.fit_model(Xd, bd, es.cols(BLOCKS, allm & ~(1 << j)), seed + 200 + j,
                          objective="multi:softprob", extra=dict(num_class=nb, eta=0.1))
        for name, G in (("Y", R[j]), ("X", GX)):
            ke = es.KernelEvidence(G.leaves(Xe), ye, be, nb)
            nc, _ = ke.stats(G.leaves(Xc), KAPPAS)
            ll = np.array([[ke.loglik(nc[i], bc, yc, lam) for lam in LAMBDAS] for i in range(len(KAPPAS))])
            i_s, l_s = np.unravel_index(np.argmax(ll), ll.shape)
            sel[f"{name}{j}"] = (KAPPAS[i_s], LAMBDAS[l_s])
            nq, eq = ke.stats(G.leaves(Xq), [KAPPAS[i_s]])
            w = ke.woe(nq[0], bq[:, j], LAMBDAS[l_s])
            if name == "Y":
                W_leafY[:, j], ess_Y[:, j], ny_Y[:, j] = w, eq[0], nq[0].sum(-1)
                W_glob[:, j] = ke.global_woe(bq[:, j])
            else:
                W_leafX[:, j] = w
    summary["kernel_selected"] = sel

    # coalition evidence and adverse action comparisons for declined test applicants
    lr = F.logit(Xr)
    cut = np.quantile(lr, APPROVAL)
    declined = np.flatnonzero(lt > cut)
    dsel = rng.choice(declined, min(N_DECLINE, len(declined)), replace=False)
    Xa = Xt[dsel]
    S = es.shapley_matrix(J)
    Vh = np.zeros((len(Xa), 1 << J))
    for mask in range(1, allm):
        Vh[:, mask] = predp(P[mask], Xa) - logit(pi_hat)
    Vh[:, allm] = F.logit(Xa) - logit(pi_hat)
    E_hat = Vh @ S.T
    band = np.flatnonzero((lr <= cut) & (lr > cut - BAND))
    summary["band_size"] = int(len(band))
    refs = np.stack([rng.choice(band, min(N_REF, len(band)), replace=False) for _ in range(len(Xa))])
    scorers = {"prod": F.logit}
    for b in range(N_BOOT):
        ii = rng.integers(0, len(yd), len(yd))
        scorers[f"boot{b}"] = es.fit_model(Xd[ii], yd[ii], list(range(X.shape[1])), seed + 300 + b, Xc, yc).logit
    attr = {}
    for name, fn in scorers.items():
        C, Dg = np.empty((len(Xa), J)), np.empty(len(Xa))
        for i in range(len(Xa)):
            phi, gap = es.baseline_shapley(fn, Xa[i], Xr[refs[i]], BLOCKS, S)
            C[i], Dg[i] = phi.mean(0), gap.mean()
        attr[f"score_{name}"], attr[f"score_{name}_gap"] = C, Dg
    np.savez_compressed(OUT / f"heloc_rep{rep}.npz", W_dir=W_dir, W_proj=W_proj, W_leafY=W_leafY,
                        W_leafX=W_leafX, W_glob=W_glob, ess_Y=ess_Y, ny_Y=ny_Y,
                        ids_test=idx["test"], ids_decl=idx["test"][dsel], bq=bq, lF=lF, yq=yt, dsel=dsel,
                        E_hat=E_hat, Vh=Vh, **attr)
    (OUT / f"heloc_rep{rep}.json").write_text(json.dumps(summary, indent=2, default=float))
    print(time.strftime("%H:%M:%S"), "rep", rep, "auc", round(summary["auc"], 4), flush=True)


if __name__ == "__main__":
    for r in sys.argv[1:]:
        main(int(r))
