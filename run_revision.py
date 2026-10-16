"""Additional analyses requested in the review of the first draft.

usage: run_revision.py PART [SEED ...]
  oracle      numerical accuracy of the oracle (independent draws; finer, wider grid and more draws)
  install     oracle with the direct installment term removed
  logodds     applicant-level and hybrid-record log-odds error of the production model; coalition offset
  refits      30 additional bootstrap refits per seed (score-band references), for H3
  samplesize  leaf-local error against evidence-sample size (seed 1)
  case        reference-policy changes and an applicant case (seed 1)
Outputs go to results/revision_*.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

import bureau_sim as bs
import estimators as es
from run_sim import SPLITS, KAPPAS, LAMBDAS, N_CALSEL

RES = Path(__file__).resolve().parent / "results"
J, BLOCKS, ALLM = bs.J, bs.BLOCKS, (1 << bs.J) - 1


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def data(seed):
    rng = np.random.default_rng(seed)
    X, y, p, booked = bs.generate(sum(SPLITS.values()), rng)
    o = np.cumsum([0] + list(SPLITS.values()))
    return {k: (X[o[i]:o[i + 1]], y[o[i]:o[i + 1]], p[o[i]:o[i + 1]], booked[o[i]:o[i + 1]])
            for i, k in enumerate(SPLITS)}


def production(seed, D):
    Xd, yd = D["dev"][:2]
    Xc, yc = D["cal"][:2]
    return es.fit_model(Xd, yd, list(range(Xd.shape[1])), seed, Xc, yc)


def binner_for(D):
    return es.BlockBinner(bs.block_design).fit(bs.to_dict(D["dev"][0]), D["dev"][1], J)


def summarize_diff(a, b):
    d = np.abs(a - b)
    return dict(rmse=np.sqrt(np.nanmean(d ** 2, 0)).tolist(), q99=np.nanquantile(d, 0.99, axis=0).tolist(),
                max=np.nanmax(d, 0).tolist())


# ------------------------------------------------------------------ oracle accuracy
def part_oracle(seed=1, nq=300, na=100):
    r = np.load(RES / f"sim_seed{seed}.npz")
    D = data(seed)
    Xt = D["test"][0]
    binner = binner_for(D)
    rnd = np.flatnonzero(~r["is_decl"])[:nq]
    Xq = Xt[r["qsel"][rnd]]
    eta = bs.eta(bs.to_dict(Xq))
    out = {}

    def loo(o, tag):
        res = bs.parallel(o, "leave_one_out", Xq, seed=seed + 1000 + len(tag))
        W = eta[:, None] - logit(res["sig"])
        with np.errstate(divide="ignore", invalid="ignore"):
            Wb = np.log((res["b_sig"] / res["sig"]) / (res["b_nsig"] / (1 - res["sig"])))
        return W, Wb, res["b"]

    base_W, base_Wb = r["W_true"][rnd], r["Wb_true"][rnd]
    for tag, kw in (("replicate", dict(M=8192)), ("fine", dict(M=32768, grid_n=48, grid_lim=5.0))):
        o = bs.Oracle(seed=seed + 77, binner=binner, **kw)
        W, Wb, pb = loo(o, tag)
        out[tag] = dict(W=summarize_diff(W, base_W), Wb=summarize_diff(Wb, base_Wb),
                        min_class_prob=float(np.nanmin(pb)), n_nan_Wb=int(np.isnan(Wb).sum()))
        log("oracle", tag, out[tag]["W"]["rmse"])
    # coalition evidence for declined applicants
    Xa = r["Xa"][:na]
    o = bs.Oracle(seed=seed + 99, M=32768, grid_n=48, grid_lim=5.0)
    pi = bs.population_rate(np.random.default_rng(seed + 9))
    V = es.clip_logit(bs.parallel(o, "coalition", Xa, seed=seed + 2000)) - logit(pi)
    V[:, 0] = 0.0
    E = V @ es.shapley_matrix(J).T
    out["fine_E"] = summarize_diff(E, r["E_true"][:na])
    out["min_conditional_prob"] = float(np.nanmin(expit(r["Vt"][:, 1:] + logit(pi))))
    out["sd_W_true"] = np.std(base_W, 0).tolist()
    (RES / "revision_oracle.json").write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------ installment counterfactual
def part_install(seed=1, nq=1000, na=200):
    r = np.load(RES / f"sim_seed{seed}.npz")
    D = data(seed)
    Xt = D["test"][0]
    rnd = np.flatnonzero(~r["is_decl"])[:nq]
    Xq, Xa = Xt[r["qsel"][rnd]], r["Xa"][:na]
    out = {}
    for coef in (0.8, 0.0):
        bs.INSTALL_COEF = coef
        o = bs.Oracle(seed=seed + 55)
        res = bs.parallel(o, "leave_one_out", Xq, seed=seed + 3000)
        W = bs.eta(bs.to_dict(Xq))[:, None] - logit(res["sig"])
        pi = bs.population_rate(np.random.default_rng(seed + 9))
        V = es.clip_logit(bs.parallel(o, "coalition", Xa, seed=seed + 4000)) - logit(pi)
        V[:, 0] = 0.0
        E = V @ es.shapley_matrix(J).T
        Wa = V[:, [ALLM]] - V[:, [ALLM & ~(1 << j) for j in range(J)]]
        out[str(coef)] = dict(W_inst_sd=float(W[:, 4].std()), W_inst_mean=float(W[:, 4].mean()),
                              W_inst_meanabs=float(np.abs(W[:, 4]).mean()),
                              E_inst_mean_declined=float(E[:, 4].mean()), E_inst_sd_declined=float(E[:, 4].std()),
                              W_inst_mean_declined=float(Wa[:, 4].mean()), W_inst_sd_declined=float(Wa[:, 4].std()),
                              pi=float(pi))
        log("install", coef, out[str(coef)])
    bs.INSTALL_COEF = 0.8
    (RES / "revision_install.json").write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------ production-model log-odds error
def part_logodds(seeds):
    out = {}
    for seed in seeds:
        r = np.load(RES / f"sim_seed{seed}.npz")
        D = data(seed)
        F = production(seed, D)
        Xt, Xr, Xe = D["test"][0], D["ref"][0], D["ev"][0]
        lt = F.logit(Xt)
        same = float(np.abs(lt[r["qsel"]] - r["lF"]).max())
        et = bs.eta(bs.to_dict(Xt))
        err = lt - et
        dec = np.quantile(et, np.linspace(0, 1, 11))
        g = np.clip(np.digitize(et, dec[1:-1]), 0, 9)
        # hybrids of declined applicants and their score-band references
        hyb_err = []
        for i in range(len(r["Xa"])):
            H = es.hybrids(r["Xa"][i], Xr[r["refs_score"][i]], BLOCKS).reshape(-1, Xt.shape[1])
            hyb_err.append(F.logit(H) - bs.eta(bs.to_dict(H)))
        hyb_err = np.concatenate(hyb_err)
        ea = F.logit(r["Xa"]) - bs.eta(bs.to_dict(r["Xa"]))
        pF = F.prob(Xe)
        offset = (logit(D["ev"][1].mean()) - logit(pF.mean())) / J
        out[seed] = dict(refit_maxdiff=same, rmse_test=float(np.sqrt((err ** 2).mean())),
                         rmse_by_true_decile=[float(np.sqrt((err[g == k] ** 2).mean())) for k in range(10)],
                         rmse_declined=float(np.sqrt((ea ** 2).mean())),
                         rmse_hybrids=float(np.sqrt((hyb_err ** 2).mean())),
                         q95_abs_hybrids=float(np.quantile(np.abs(hyb_err), 0.95)),
                         coalition_offset_per_block=float(offset))
        log("logodds", seed, out[seed])
    path = RES / "revision_logodds.json"
    merged = json.load(open(path)) if path.exists() else {}
    merged.update({str(k): v for k, v in out.items()})
    path.write_text(json.dumps(merged, indent=1))


# ------------------------------------------------------------------ additional refits
def part_refits(seeds, n_more=30):
    for seed in seeds:
        r = np.load(RES / f"sim_seed{seed}.npz")
        D = data(seed)
        Xd, yd = D["dev"][:2]
        Xc, yc = D["cal"][:2]
        Xr = D["ref"][0]
        S = es.shapley_matrix(J)
        rng = np.random.default_rng(seed + 12345)
        C = np.empty((n_more, len(r["Xa"]), J))
        for b in range(n_more):
            ii = rng.integers(0, len(yd), len(yd))
            m = es.fit_model(Xd[ii], yd[ii], list(range(Xd.shape[1])), seed + 600 + b, Xc, yc)
            for i in range(len(r["Xa"])):
                phi, _ = es.baseline_shapley(m.logit, r["Xa"][i], Xr[r["refs_score"][i]], BLOCKS, S)
                C[b, i] = phi.mean(0)
            log("refit", seed, b)
        np.savez_compressed(RES / f"revision_refits_seed{seed}.npz", C=C)


# ------------------------------------------------------------------ leaf-local error by evidence-sample size
def part_samplesize(seed=1, sizes=(2500, 10000, 30000, 100000)):
    r = np.load(RES / f"sim_seed{seed}.npz")
    D = data(seed)
    Xd, yd = D["dev"][:2]
    Xc, yc = D["cal"][:2]
    Xe, ye = D["ev"][:2]
    Xq = D["test"][0][r["qsel"]]
    binner = binner_for(D)
    xd, xe, xc, xq = (bs.to_dict(a) for a in (Xd, Xe, Xc, Xq))
    rng = np.random.default_rng(seed + 31)
    csel = rng.choice(len(Xc), N_CALSEL, replace=False)
    out = {}
    for j in range(J):
        nb = binner.nbins(j)
        cols = es.cols(BLOCKS, ALLM & ~(1 << j))
        GY = es.fit_model(Xd, yd, cols, seed + 1 + j)
        GX = es.fit_model(Xd, binner(xd, j), cols, seed + 200 + j, objective="multi:softprob",
                          extra=dict(num_class=nb, eta=0.1))
        bq, bc, be = binner(xq, j), binner(xc, j)[csel], binner(xe, j)
        for gname, G in (("Y", GY), ("X", GX)):
            Le, Lc, Lq = G.leaves(Xe), G.leaves(Xc[csel]), G.leaves(Xq)
            for n in sizes:
                sub = rng.choice(len(Xe), n, replace=False) if n < len(Xe) else np.arange(len(Xe))
                ke = es.KernelEvidence(Le[sub], ye[sub], be[sub], nb)
                nc, _ = ke.stats(Lc, KAPPAS)
                ll = np.array([[ke.loglik(nc[i], bc, yc[csel], lam) for lam in LAMBDAS] for i in range(len(KAPPAS))])
                i_s, l_s = np.unravel_index(np.argmax(ll), ll.shape)
                nq, eq = ke.stats(Lq, [1, KAPPAS[i_s]])
                w_sel = ke.woe(nq[1], bq, LAMBDAS[l_s])
                w1 = ke.woe(nq[0], bq, LAMBDAS[int(np.argmax(ll[0]))])
                wg = ke.global_woe(bq)
                t = r["Wb_true"][:, j]
                rm = lambda w: float(np.sqrt(np.nanmean((w - t) ** 2)))
                out[f"{gname}{j}_{n}"] = dict(rmse_sel=rm(w_sel), rmse_k1=rm(w1), rmse_glob=rm(wg),
                                              kappa=KAPPAS[i_s], lam=LAMBDAS[l_s],
                                              ess1_median=float(np.median(eq[1][:, 1])),
                                              mean_abs_dev_from_global=float(np.mean(np.abs(w_sel - wg))))
                del ke
            log("samplesize", gname, j, {n: round(out[f'{gname}{j}_{n}']['rmse_sel'], 3) for n in sizes})
    (RES / "revision_samplesize.json").write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------ reference policy and applicant case
def part_case(seeds):
    fields = {"utilization": 2, "limit": 0, "balance": 1, "n30": 3, "n60": 4, "n90": 5, "months_since_delq": 6,
              "oldest": 7, "inquiries": 9, "inst_fraction": 13, "public_record": 14}
    agg = {}
    for seed in seeds:
        r = np.load(RES / f"sim_seed{seed}.npz")
        D = data(seed)
        Xr = D["ref"][0]
        ts, tp = r["score_prod"].argmax(1), r["proximity_prod"].argmax(1)
        ch = ts != tp
        Xa = r["Xa"]
        for pol in ("score", "proximity"):
            R = Xr[r[f"refs_{pol}"]]                       # (A, N_REF, p)
            Rm = R.copy()
            Rm[..., 6] = np.where(R[..., 3] > 0, R[..., 6], np.nan)
            m = np.nanmean(Rm, 1)
            for k, c in fields.items():
                agg.setdefault(f"{pol}_{k}_changed", []).append(float(np.nanmean(m[ch, c])))
                agg.setdefault(f"{pol}_{k}_all", []).append(float(np.nanmean(m[:, c])))
        Xa = r["Xa"]
        for k, c in fields.items():
            v = np.where(Xa[:, 3] > 0, Xa[:, 6], np.nan) if k == "months_since_delq" else Xa[:, c]
            agg.setdefault(f"applicant_{k}_changed", []).append(float(np.nanmean(v[ch])))
        agg.setdefault("share_changed", []).append(float(ch.mean()))
        d2r = (ts == 1) & (tp == 0)                         # delinquency -> revolving
        agg.setdefault("d2r_n", []).append(float(d2r.sum()))
        for pol in ("score", "proximity"):
            R = Xr[r[f"refs_{pol}"][d2r]]
            for k, c in (("n30", 3), ("n90", 5), ("utilization", 2), ("limit", 0)):
                agg.setdefault(f"d2r_{pol}_{k}", []).append(float(R[..., c].mean()))
        for k, c in (("n30", 3), ("n90", 5), ("utilization", 2), ("limit", 0)):
            agg.setdefault(f"d2r_applicant_{k}", []).append(float(Xa[d2r, c].mean()))
        trans = np.zeros((J, J))
        np.add.at(trans, (ts, tp), 1)
        agg.setdefault("transitions", []).append(trans.tolist())
        if seed == seeds[0]:
            case = pick_case(r, Xr, D, seed)
    out = {k: (np.mean(v, 0).tolist() if k == "transitions" else float(np.mean(v))) for k, v in agg.items()}
    out["case"] = case
    (RES / "revision_case.json").write_text(json.dumps(out, indent=1))
    log("case", json.dumps(case)[:400])


def pick_case(r, Xr, D, seed):
    """A declined applicant whose leading block changes between reference policies, with revolving
    leading under one policy and delinquency under the other."""
    ts, tp = r["score_prod"].argmax(1), r["proximity_prod"].argmax(1)
    cand = np.flatnonzero(((ts == 1) & (tp == 0)) | ((ts == 0) & (tp == 1)))
    # the clearest change: largest shift of the revolving-minus-delinquency difference between policies
    shift = np.abs((r["score_prod"][cand, 0] - r["score_prod"][cand, 1]) - (r["proximity_prod"][cand, 0] - r["proximity_prod"][cand, 1]))
    i = int(cand[np.argmax(shift)])
    x = r["Xa"][i]
    a = int(r["a_pos"][i])
    rec = {f: float(x[k]) for k, f in enumerate(bs.FEATURES)}
    refs = {}
    for pol in ("score", "proximity"):
        R = Xr[r[f"refs_{pol}"][i]]
        refs[pol] = {f: float(np.mean(R[:, k])) for k, f in enumerate(bs.FEATURES)}
        refs[pol]["share_any_30dpd"] = float(np.mean(R[:, 3] > 0))
        refs[pol]["C"] = r[f"{pol}_prod"][i].tolist()
        refs[pol]["C_true"] = r[f"{pol}_true"][i].tolist() if f"{pol}_true" in r.files else None
        refs[pol]["gap"] = float(r[f"{pol}_prod_gap"][i])
    return dict(index=i, seed=seed, record=rec, refs=refs, W_true=r["W_true"][a].tolist(), W_proj=r["W_proj"][a].tolist(),
                E_true=r["E_true"][i].tolist(), E_hat=r["E_hat"][i].tolist(), f=float(r["lF"][a]),
                eta=float(r["eta_q"][a]), cut=float(r["cut"]))


if __name__ == "__main__":
    part = sys.argv[1]
    seeds = [int(s) for s in sys.argv[2:]] or [1, 2, 3, 4, 5]
    {"oracle": lambda: part_oracle(), "install": lambda: part_install(), "logodds": lambda: part_logodds(seeds),
     "refits": lambda: part_refits(seeds), "samplesize": lambda: part_samplesize(),
     "case": lambda: part_case(seeds)}[part]()
