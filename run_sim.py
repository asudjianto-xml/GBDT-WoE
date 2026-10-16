"""One replicate of the bureau-style simulation study.

usage: run_sim.py SEED [--quick]
Writes results/sim_seed{SEED}.npz and .json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

import bureau_sim as bs
import estimators as es

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

SPLITS = dict(dev=100_000, cal=25_000, ev=100_000, ref=50_000, test=25_000)
DECLINE_RATE = 0.20
BAND = 0.30          # passing band width on the logit scale
N_REF = 50
N_DECLINE = 400
N_QUERY = 3000
N_CIV = 1000         # random test applicants with oracle coalition evidence (information value)
N_BOOT = 10
LAMBDAS = [0.1, 0.5, 2, 8, 32]
KAPPAS = [1, 2, 4, 8, 16, 32]
N_CALSEL = 4000


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main(seed, quick=False):
    global SPLITS, N_QUERY, N_DECLINE, N_BOOT
    if quick:
        SPLITS = {k: v // 5 for k, v in SPLITS.items()}
        N_QUERY, N_DECLINE, N_BOOT = 300, 60, 2
    rng = np.random.default_rng(seed)
    n = sum(SPLITS.values())
    X, y, ptrue, booked = bs.generate(n, rng)
    idx, a = {}, 0
    for k, m in SPLITS.items():
        idx[k] = np.arange(a, a + m)
        a += m
    D = {k: (X[i], y[i], ptrue[i], booked[i]) for k, i in idx.items()}
    Xd, yd = D["dev"][:2]
    Xc, yc = D["cal"][:2]
    Xe, ye = D["ev"][:2]
    Xr = D["ref"][0]
    Xt, yt, pt = D["test"][:3]
    J, blocks = bs.J, bs.BLOCKS
    allm = (1 << J) - 1
    import platform, xgboost
    summary = dict(seed=seed, splits=SPLITS, default_rate_dev=float(yd.mean()), host=platform.node(),
                   xgboost=xgboost.__version__, numpy=np.__version__)

    binner = es.BlockBinner(bs.block_design).fit(bs.to_dict(Xd), yd, J)
    summary["n_bins"] = [binner.nbins(j) for j in range(J)]

    # --- production (full) model and feature-excluded models
    log("fit full")
    F = es.fit_model(Xd, yd, list(range(X.shape[1])), seed, Xc, yc)
    lt = F.logit(Xt)
    summary["full_test"] = dict(auc=roc_auc_score(yt, lt), logloss=log_loss(yt, expit(lt)),
                                brier=brier_score_loss(yt, expit(lt)), auc_true=roc_auc_score(yt, pt),
                                logloss_true=log_loss(yt, pt))
    log("full", summary["full_test"])
    R = [es.fit_model(Xd, yd, es.cols(blocks, allm & ~(1 << j)), seed + 1 + j, Xc, yc) for j in range(J)]

    # --- projection models: E[p_F(X) | x_S] for every proper nonempty S, fitted on the evidence sample
    log("projections")
    pFe = F.prob(Xe)
    P = {}
    for mask in range(1, allm):
        P[mask] = es.Model(es.cols(blocks, mask), es.fit_xgb(Xe[:, es.cols(blocks, mask)], pFe, seed + 100 + mask,
                                                             objective="reg:logistic"))
    pi_hat = float(ye.mean())

    # --- queries: random test records plus sampled declines
    cut = np.quantile(F.logit(Xr), 1 - DECLINE_RATE)
    declined = np.flatnonzero(lt > cut)
    dsel = rng.choice(declined, N_DECLINE, replace=False)
    qsel = np.unique(np.concatenate([rng.choice(len(Xt), N_QUERY, replace=False), dsel]))
    Xq = Xt[qsel]
    xq = bs.to_dict(Xq)
    bq = np.stack([binner(xq, j) for j in range(J)], 1)

    # --- oracle leave-one-block-out quantities
    log("oracle loo", len(Xq))
    O = bs.Oracle(M=8192, seed=seed + 7, binner=binner)
    orc = bs.parallel(O, "leave_one_out", Xq, seed=seed + 11)
    eta_q = bs.eta(xq)
    W_true = eta_q[:, None] - es.clip_logit(orc["sig"])
    W_booked = eta_q[:, None] - es.clip_logit(orc["sig_s"] / orc["s"])
    with np.errstate(divide="ignore", invalid="ignore"):
        Wb_true = np.log((orc["b_sig"] / orc["sig"]) / (orc["b_nsig"] / (1 - orc["sig"])))
    bin_mc = orc["b"] * O.M

    # --- estimates
    log("estimates")
    lF = F.logit(Xq)
    W_dir = np.stack([lF - R[j].logit(Xq) for j in range(J)], 1)
    W_proj = np.stack([lF - es.clip_logit(P[allm & ~(1 << j)].booster.predict(
        es.xgb.DMatrix(Xq[:, P[allm & ~(1 << j)].columns]))) for j in range(J)], 1)

    xd, xe, xc = bs.to_dict(Xd), bs.to_dict(Xe), bs.to_dict(Xc)
    csel = rng.choice(len(Xc), min(N_CALSEL, len(Xc)), replace=False)
    nk, nl = len(KAPPAS), len(LAMBDAS)
    leaf = {g: dict(grid=np.empty((len(Xq), J, nk, nl)), sel=np.empty((len(Xq), J)), k1=np.empty((len(Xq), J)),
                    ny=np.empty((len(Xq), J, 2)), ess=np.empty((len(Xq), J, 2)), ny1=np.empty((len(Xq), J, 2)),
                    ess1=np.empty((len(Xq), J, 2))) for g in ("Y", "X")}
    W_glob = np.empty((len(Xq), J))
    sel_par, sel_curves, ntrees = {}, {}, {}
    for j in range(J):
        be, bc, bd = binner(xe, j), binner(xc, j)[csel], binner(xd, j)
        nb = binner.nbins(j)
        GX = es.fit_model(Xd, bd, es.cols(blocks, allm & ~(1 << j)), seed + 200 + j,
                          objective="multi:softprob", extra=dict(num_class=nb, eta=0.1))
        for name, G in (("Y", R[j]), ("X", GX)):
            ke = es.KernelEvidence(G.leaves(Xe), ye, be, nb)
            ntrees[f"{name}{j}"] = ke.T
            nc, _ = ke.stats(G.leaves(Xc[csel]), KAPPAS)
            ll = np.array([[ke.loglik(nc[i], bc, yc[csel], lam) for lam in LAMBDAS] for i in range(nk)])
            i_s, l_s = np.unravel_index(np.argmax(ll), ll.shape)
            l_1 = int(np.argmax(ll[0]))
            sel_par[f"{name}{j}"] = (KAPPAS[i_s], LAMBDAS[l_s], LAMBDAS[l_1])
            sel_curves[f"{name}{j}"] = ll.tolist()
            nq, eq = ke.stats(G.leaves(Xq), KAPPAS)
            L = leaf[name]
            for i in range(nk):
                for l, lam in enumerate(LAMBDAS):
                    L["grid"][:, j, i, l] = ke.woe(nq[i], bq[:, j], lam)
            L["sel"][:, j], L["k1"][:, j] = L["grid"][:, j, i_s, l_s], L["grid"][:, j, 0, l_1]
            L["ny"][:, j], L["ess"][:, j] = nq[i_s].sum(-1), eq[i_s]
            L["ny1"][:, j], L["ess1"][:, j] = nq[0].sum(-1), eq[0]
            if name == "Y":
                W_glob[:, j] = ke.global_woe(bq[:, j])
            del ke
        log("leaf block", j, sel_par[f"Y{j}"], sel_par[f"X{j}"], ntrees[f"Y{j}"], ntrees[f"X{j}"])

    # --- coalition (Shapley) evidence for the declined applicants
    log("coalition oracle")
    Xa = Xt[dsel]
    S = es.shapley_matrix(J)
    pi_true = bs.population_rate(np.random.default_rng(seed + 9))
    Vt = es.clip_logit(bs.parallel(O, "coalition", Xa, seed=seed + 13)) - logit(pi_true)
    Vt[:, 0] = 0.0
    E_true = Vt @ S.T
    Vh = np.zeros((len(Xa), 1 << J))
    for mask in range(1, allm):
        Vh[:, mask] = es.clip_logit(P[mask].booster.predict(es.xgb.DMatrix(Xa[:, P[mask].columns]))) - logit(pi_hat)
    Vh[:, allm] = F.logit(Xa) - logit(pi_hat)
    E_hat = Vh @ S.T
    W_loo_true_a = Vt[:, [allm]] - Vt[:, [allm & ~(1 << j) for j in range(J)]]
    W_loo_hat_a = Vh[:, [allm]] - Vh[:, [allm & ~(1 << j) for j in range(J)]]
    summary["pi_true"], summary["pi_hat"] = pi_true, pi_hat

    # --- adverse action: references, exact grouped attribution, refits
    log("attribution")
    lr = F.logit(Xr)
    band = np.flatnonzero((lr <= cut) & (lr > cut - BAND))
    summary["band_size"] = int(len(band))
    leaves_full_band = F.leaves(Xr[band])
    leaves_full_a = F.leaves(Xa)
    refs = {"score": np.stack([rng.choice(band, N_REF, replace=False) for _ in range(len(Xa))])}
    prox = []
    for i in range(len(Xa)):
        sim = (leaves_full_band == leaves_full_a[i]).mean(1)
        prox.append(band[np.argsort(-sim, kind="stable")[:N_REF]])
    refs["proximity"] = np.stack(prox)

    boot = []
    for b in range(N_BOOT):
        ii = rng.integers(0, len(yd), len(yd))
        boot.append(es.fit_model(Xd[ii], yd[ii], list(range(X.shape[1])), seed + 300 + b, Xc, yc))
    scorers = {"prod": F.logit, "true": lambda Z: bs.eta(bs.to_dict(Z))}
    for b, m in enumerate(boot):
        scorers[f"boot{b}"] = m.logit
    attr = {}
    for pol, rr in refs.items():
        for name, fn in scorers.items():
            C = np.empty((len(Xa), J))
            Dg = np.empty(len(Xa))
            for i in range(len(Xa)):
                phi, gap = es.baseline_shapley(fn, Xa[i], Xr[rr[i]], blocks, S)
                C[i], Dg[i] = phi.mean(0), gap.mean()
            attr[f"{pol}_{name}"] = C
            attr[f"{pol}_{name}_gap"] = Dg
        log("policy", pol)

    # comparative evidence at a fixed context with the direct estimator and the single-swap contrast
    CW, swap = np.empty((len(Xa), J)), np.empty((len(Xa), J))
    for i in range(len(Xa)):
        Rr = Xr[refs["score"][i]]
        for j in range(J):
            H = np.repeat(Xa[i][None], len(Rr), 0)
            H[:, list(blocks[j])] = Rr[:, list(blocks[j])]
            wx = F.logit(Xa[i][None])[0] - R[j].logit(Xa[i][None])[0]
            wr = F.logit(H) - R[j].logit(H)
            CW[i, j] = np.mean(wx - wr)
            swap[i, j] = np.mean(F.logit(Xa[i][None])[0] - F.logit(H))
    summary["comparative_identity_maxabs"] = float(np.abs(CW - swap).max())

    # --- booked-only estimation (selection by a legacy policy)
    log("booked models")
    bd_, bc_ = D["dev"][3], D["cal"][3]
    Fb = es.fit_model(Xd[bd_], yd[bd_], list(range(X.shape[1])), seed + 400, Xc[bc_], yc[bc_])
    Rb = [es.fit_model(Xd[bd_], yd[bd_], es.cols(blocks, allm & ~(1 << j)), seed + 401 + j, Xc[bc_], yc[bc_])
          for j in range(J)]
    W_dir_booked = np.stack([Fb.logit(Xq) - Rb[j].logit(Xq) for j in range(J)], 1)

    # --- coalition evidence on random test applicants, for conditional information value.
    # Runs last with its own generator, so the quantities above are unchanged.
    log("coalition oracle, random applicants")
    csel_civ = np.random.default_rng(seed + 17).choice(len(Xt), N_CIV, replace=False)
    Xv = Xt[csel_civ]
    Vt_civ = es.clip_logit(bs.parallel(O, "coalition", Xv, seed=seed + 15)) - logit(pi_true)
    Vt_civ[:, 0] = 0.0
    Vh_civ = np.zeros((len(Xv), 1 << J))
    for mask in range(1, allm):
        Vh_civ[:, mask] = es.clip_logit(P[mask].booster.predict(es.xgb.DMatrix(Xv[:, P[mask].columns]))) - logit(pi_hat)
    Vh_civ[:, allm] = F.logit(Xv) - logit(pi_hat)

    is_decl = np.isin(qsel, dsel)
    a_pos = np.array([np.flatnonzero(qsel == d)[0] for d in dsel])
    np.savez_compressed(
        OUT / f"sim_seed{seed}{'_quick' if quick else ''}.npz",
        qsel=qsel, dsel=dsel, is_decl=is_decl, a_pos=a_pos, bq=bq, eta_q=eta_q, lF=lF, ptrue_q=pt[qsel],
        W_true=W_true, W_booked=W_booked, Wb_true=Wb_true, bin_mc=bin_mc,
        W_dir=W_dir, W_proj=W_proj, W_glob=W_glob, W_dir_booked=W_dir_booked,
        **{f"leaf{g}_{k}": v for g, d in leaf.items() for k, v in d.items()},
        legacy_q=bs.legacy_score(xq),
        E_true=E_true, E_hat=E_hat, Vt=Vt, Vh=Vh, W_loo_true_a=W_loo_true_a, W_loo_hat_a=W_loo_hat_a,
        CW=CW, swap=swap, Xa=Xa, refs_score=refs["score"], refs_proximity=refs["proximity"],
        band=band, cut=cut, csel_civ=csel_civ, Vt_civ=Vt_civ, Vh_civ=Vh_civ, ptrue_civ=pt[csel_civ],
        y_civ=yt[csel_civ], **attr)
    summary["kernel_selected"] = sel_par
    summary["kernel_loglik"] = sel_curves
    summary["kappas"], summary["lambdas"], summary["ntrees"] = KAPPAS, LAMBDAS, ntrees
    (OUT / f"sim_seed{seed}{'_quick' if quick else ''}.json").write_text(json.dumps(summary, indent=2, default=float))
    log("done")


if __name__ == "__main__":
    main(int(sys.argv[1]), quick="--quick" in sys.argv)
