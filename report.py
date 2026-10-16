"""Aggregate replicate outputs into tables (LaTeX), figures and a JSON summary."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import bureau_sim as bs

ROOT = Path(__file__).resolve().parent
RES, FIG, TAB = ROOT / "results", ROOT / "figures", ROOT / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)
NAMES = bs.BLOCK_NAMES
J = bs.J
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42})


def rmse(a, b, axis=0):
    return np.sqrt(np.nanmean((a - b) ** 2, axis))


def msd(v, d=2):
    v = np.asarray(v, float)
    return f"{v.mean():.{d}f} ({v.std(ddof=1):.{d}f})" if len(v) > 1 else f"{v.mean():.{d}f}"


def load_sim():
    runs = []
    for f in sorted(glob.glob(str(RES / "sim_seed[0-9].npz"))):
        r = dict(np.load(f))
        r["meta"] = json.load(open(f.replace(".npz", ".json")))
        runs.append(r)
    return runs


def top_reason_stats(r, pol="score"):
    C = r[f"{pol}_prod"]
    top = C.argmax(1)
    boots = [k for k in r if k.startswith(f"{pol}_boot") and not k.endswith("_gap")]
    tb = np.stack([r[k].argmax(1) for k in boots], 1)
    agree = (tb == top[:, None]).mean(1)
    srt = np.sort(C, 1)
    margin = (srt[:, -1] - srt[:, -2])
    gap = r[f"{pol}_prod_gap"]
    return top, agree, margin, gap


def h3_auc(rows, feats, outcome, seed=0):
    """Cross-validated AUC of a logistic model predicting the outcome from the named features."""
    X = np.column_stack([rows[f] for f in feats])
    y = rows[outcome].astype(int)
    if y.min() == y.max():
        return np.nan
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        m = LogisticRegression(max_iter=2000).fit((X[tr] - mu) / sd, y[tr])
        p[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
    return roc_auc_score(y, p)


def sim_report(runs):
    S = {}
    seeds = [r["meta"]["seed"] for r in runs]
    S["seeds"] = seeds
    S["n_seeds"] = len(runs)
    ft = [r["meta"]["full_test"] for r in runs]
    S["auc"] = [f["auc"] for f in ft]
    S["auc_true"] = [f["auc_true"] for f in ft]
    S["logloss"] = [f["logloss"] for f in ft]
    S["logloss_true"] = [f["logloss_true"] for f in ft]
    S["default_rate"] = [r["meta"]["default_rate_dev"] for r in runs]
    S["pi_true"] = runs[0]["meta"]["pi_true"]
    S["n_bins"] = runs[0]["meta"]["n_bins"]
    S["band_size"] = [r["meta"]["band_size"] for r in runs]
    S["identity_maxabs"] = max(r["meta"]["comparative_identity_maxabs"] for r in runs)

    # ---- pointwise and bin-level evidence accuracy
    est_pt = {"Direct": "W_dir", "Projection": "W_proj"}
    est_bin = {"Global WoE": "W_glob", "Leaf Y, $\\kappa=1$": "leafY_k1", "Leaf Y, selected": "leafY_sel",
               "Leaf X, $\\kappa=1$": "leafX_k1", "Leaf X, selected": "leafX_sel"}
    acc = {}
    for lab, k in est_pt.items():
        acc[lab] = np.stack([rmse(r[k], r["W_true"]) for r in runs])
    for lab, k in est_bin.items():
        acc[lab] = np.stack([rmse(r[k], r["Wb_true"]) for r in runs])
    sd_pt = np.stack([np.nanstd(r["W_true"], 0) for r in runs])
    sd_bin = np.stack([np.nanstd(r["Wb_true"], 0) for r in runs])
    S["rmse"] = {k: v.mean(0).tolist() for k, v in acc.items()}
    S["rmse_sd"] = {k: v.std(0, ddof=1).tolist() for k, v in acc.items()}
    S["sd_true_pt"], S["sd_true_bin"] = sd_pt.mean(0).tolist(), sd_bin.mean(0).tolist()
    # pointwise vs bin-level oracle difference (coarsening)
    S["coarsening_rmse"] = np.stack([rmse(r["W_true"], r["Wb_true"]) for r in runs]).mean(0).tolist()
    # correlation and sign agreement for material evidence
    def corr(a, b):
        return np.array([np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(J)])
    S["corr"] = {lab: np.stack([corr(r[k], r["W_true"]) for r in runs]).mean(0).tolist() for lab, k in est_pt.items()}
    S["corr"].update({lab: np.stack([corr(np.nan_to_num(r[k]), np.nan_to_num(r["Wb_true"])) for r in runs]).mean(0).tolist()
                      for lab, k in est_bin.items()})
    def sign_acc(a, b, thr=0.25):
        m = np.abs(b) > thr
        return np.array([(np.sign(a[m[:, j], j]) == np.sign(b[m[:, j], j])).mean() if m[:, j].sum() > 20 else np.nan
                         for j in range(J)])
    S["sign"] = {lab: np.nanmean(np.stack([sign_acc(r[k], r["W_true"]) for r in runs]), 0).tolist() for lab, k in est_pt.items()}
    S["sign"].update({lab: np.nanmean(np.stack([sign_acc(r[k], r["Wb_true"]) for r in runs]), 0).tolist()
                      for lab, k in est_bin.items()})
    S["material_share"] = np.stack([(np.abs(r["W_true"]) > 0.25).mean(0) for r in runs]).mean(0).tolist()

    # file age and utilization subgroups (sign reversal of the utilization effect for young files)
    from run_sim import SPLITS
    sub = {}
    for r in runs:
        X, *_ = bs.generate(sum(SPLITS.values()), np.random.default_rng(r["meta"]["seed"]))
        o = np.cumsum([0] + list(SPLITS.values()))
        Xq = X[o[4]:o[5]][r["qsel"]]
        u, thin = Xq[:, 2], bs.thin_file(Xq[:, 7]) > 0.5
        for nm, m in (("young_high_util", thin & (u > 0.5)), ("young_low_util", thin & (u < 0.2)),
                      ("established_high_util", ~thin & (u > 0.5)), ("established_low_util", ~thin & (u < 0.2))):
            d = sub.setdefault(nm, {k: [] for k in ("n", "W_age", "glob_age", "proj_age", "W_rev", "glob_rev", "proj_rev")})
            d["n"].append(int(m.sum()))
            for blk, tag in ((2, "age"), (0, "rev")):
                d[f"W_{tag}"].append(float(r["W_true"][m, blk].mean()))
                d[f"glob_{tag}"].append(float(r["W_glob"][m, blk].mean()))
                d[f"proj_{tag}"].append(float(r["W_proj"][m, blk].mean()))
    S["subgroups"] = {k: {kk: float(np.mean(vv)) for kk, vv in v.items()} for k, v in sub.items()}
    # ---- kernel grid
    S["kappas"], S["lambdas"] = runs[0]["meta"]["kappas"], runs[0]["meta"]["lambdas"]
    for g in ("Y", "X"):
        grid = np.stack([rmse(r[f"leaf{g}_grid"], r["Wb_true"][:, :, None, None]) for r in runs])  # seeds,J,k,l
        S[f"grid_{g}"] = grid.mean(0).tolist()
        S[f"selected_{g}"] = [[r["meta"]["kernel_selected"][f"{g}{j}"][:2] for j in range(J)] for r in runs]
        S[f"ess_{g}_sel"] = np.stack([np.nanmedian(r[f"leaf{g}_ess"], 0) for r in runs]).mean(0).tolist()
        S[f"ess_{g}_k1"] = np.stack([np.nanmedian(r[f"leaf{g}_ess1"], 0) for r in runs]).mean(0).tolist()
    # error of selected leaf-Y by default-class ESS tercile
    terc = []
    for r in runs:
        e = r["leafY_ess"][:, :, 1]
        err = (r["leafY_sel"] - r["Wb_true"]) ** 2
        row = []
        for j in range(J):
            q = np.nanquantile(e[:, j], [1 / 3, 2 / 3])
            g = np.digitize(e[:, j], q)
            row.append([np.sqrt(np.nanmean(err[g == t, j])) for t in range(3)])
        terc.append(row)
    S["leafY_rmse_by_ess_tercile"] = np.array(terc).mean(0).tolist()

    # ---- additivity and coalition evidence (declined applicants)
    add = []
    for r in runs:
        Wfull = r["Vt"][:, -1]
        s_loo = r["W_loo_true_a"].sum(1)
        add.append(dict(under=np.mean(s_loo < Wfull - 0.05), over=np.mean(s_loo > Wfull + 0.05),
                        mean_W=Wfull.mean(), mean_sum=s_loo.mean(),
                        mad=np.mean(np.abs(s_loo - Wfull))))
    S["additivity"] = {k: [a[k] for a in add] for k in add[0]}
    S["E_rmse"] = np.stack([rmse(r["E_hat"], r["E_true"]) for r in runs]).mean(0).tolist()
    S["E_sd"] = np.stack([r["E_true"].std(0) for r in runs]).mean(0).tolist()
    S["E_mean"] = np.stack([r["E_true"].mean(0) for r in runs]).mean(0).tolist()
    S["Wloo_mean_decl"] = np.stack([r["W_loo_true_a"].mean(0) for r in runs]).mean(0).tolist()
    S["redund_mean"] = np.stack([(r["E_true"] - r["W_loo_true_a"]).mean(0) for r in runs]).mean(0).tolist()

    # ---- selection
    sel = []
    for r in runs:
        d = np.abs(r["W_booked"] - r["W_true"])
        legacy_decl = r["legacy_q"] > bs.LEGACY_T
        near = np.abs(r["legacy_q"] - bs.LEGACY_T) < 0.25
        cur = r["lF"] > r["cut"]
        sel.append(dict(all=d.mean(0), legacy_decl=d[legacy_decl].mean(0), near=d[near].mean(0),
                        current_decl=d[cur].mean(0), share_current_decl=cur.mean(),
                        signed_min=(r["W_booked"] - r["W_true"]).min(0),
                        dir_vs_ttd=rmse(r["W_dir_booked"], r["W_true"]),
                        dir_vs_booked=rmse(r["W_dir_booked"], r["W_booked"]),
                        dirttd_vs_ttd=rmse(r["W_dir"], r["W_true"]),
                        share_legacy_decl=legacy_decl.mean()))
    S["selection"] = {k: np.stack([s[k] for s in sel]).mean(0).tolist() for k in sel[0]}

    # ---- adverse action
    aa = {}
    rows = {k: [] for k in ["agree", "margin", "rel_margin", "unique", "shared", "uniq_true", "shared_true",
                            "unstable", "wrong", "seed", "gap"]}
    tops = []
    for r in runs:
        top, agree, margin, gap = top_reason_stats(r, "score")
        top_true = r["score_true"].argmax(1)
        topp, _, _, _ = top_reason_stats(r, "proximity")
        a_pos = r["a_pos"]
        Wp = r["W_proj"][a_pos]
        Eh = r["E_hat"]
        idx = np.arange(len(top))
        rows["agree"].append(agree)
        rows["margin"].append(margin)
        rows["rel_margin"].append(margin / np.maximum(gap, 1e-6))
        rows["unique"].append(Wp[idx, top])
        rows["shared"].append(Eh[idx, top] - r["W_loo_hat_a"][idx, top])
        rows["uniq_true"].append(r["W_loo_true_a"][idx, top])
        rows["shared_true"].append(r["E_true"][idx, top] - r["W_loo_true_a"][idx, top])
        rows["unstable"].append(agree < 0.8)
        rows["wrong"].append(top != top_true)
        rows["seed"].append(np.full(len(top), r["meta"]["seed"]))
        rows["gap"].append(gap)
        rho = np.array([spearmanr(r["score_prod"][i], r["proximity_prod"][i])[0] for i in range(len(top))])
        tops.append(dict(dist_prod=np.bincount(top, minlength=J) / len(top),
                         dist_true=np.bincount(top_true, minlength=J) / len(top),
                         dist_prox=np.bincount(topp, minlength=J) / len(top),
                         agree_true=np.mean(top == top_true), agree_prox=np.mean(top == topp),
                         spearman_prox=np.nanmedian(rho), mean_stability=agree.mean(),
                         share_unstable=np.mean(agree < 0.8), mean_gap=gap.mean(),
                         topk2_true=np.mean([len(set(np.argsort(-r["score_prod"][i])[:2]) &
                                                 set(np.argsort(-r["score_true"][i])[:2])) / 2 for i in range(len(top))]),
                         C_mean=r["score_prod"].mean(0), C_true_mean=r["score_true"].mean(0),
                         C_prox_mean=r["proximity_prod"].mean(0)))
    aa = {k: np.stack([t[k] for t in tops]).tolist() for k in tops[0]}
    S["aa"] = aa
    R = {k: np.concatenate(v) for k, v in rows.items()}
    S["h3"] = {}
    for outcome in ("unstable", "wrong"):
        S["h3"][outcome] = dict(
            rate=float(R[outcome].mean()),
            margin=h3_auc(R, ["rel_margin", "margin"], outcome),
            margin_evidence=h3_auc(R, ["rel_margin", "margin", "unique", "shared"], outcome),
            evidence=h3_auc(R, ["unique", "shared"], outcome),
            margin_oracle=h3_auc(R, ["rel_margin", "margin", "uniq_true", "shared_true"], outcome))
        # seed-level variability of the AUC difference
        diffs = []
        for s in np.unique(R["seed"]):
            m = R["seed"] == s
            sub = {k: v[m] for k, v in R.items()}
            diffs.append(h3_auc(sub, ["rel_margin", "margin", "unique", "shared"], outcome)
                         - h3_auc(sub, ["rel_margin", "margin"], outcome))
        S["h3"][outcome]["diff_by_seed"] = diffs
    # stability by unique-evidence tercile
    q = np.quantile(R["shared"], [1 / 3, 2 / 3])
    g = np.digitize(R["shared"], q)
    S["stability_by_shared_tercile"] = [float(R["agree"][g == t].mean()) for t in range(3)]
    S["shared_tercile_edges"] = q.tolist()
    q = np.quantile(R["rel_margin"], [1 / 3, 2 / 3])
    g = np.digitize(R["rel_margin"], q)
    S["stability_by_margin_tercile"] = [float(R["agree"][g == t].mean()) for t in range(3)]
    return S, R


def lib_report():
    out = {}
    for f in sorted(glob.glob(str(RES / "lib_seed*.json"))):
        s = int(Path(f).stem.replace("lib_seed", ""))
        r = np.load(f.replace(".json", ".npz"))
        sim = np.load(RES / f"sim_seed{s}.npz")
        meta = json.load(open(f))
        topx = sim["score_prod"].argmax(1)
        topt = sim["score_true"].argmax(1)
        Wx = sim["W_dir"]
        for kind in ("lightgbm", "catboost"):
            d = out.setdefault(kind, dict(auc=[], rmse=[], agree_xgb=[], agree_true=[], corr_W_xgb=[], refs_passing=[]))
            d["auc"].append(meta[kind]["auc"])
            d["rmse"].append(meta[kind]["rmse"])
            top = r[f"{kind}_C"].argmax(1)
            d["agree_xgb"].append(float(np.mean(top == topx)))
            d["agree_true"].append(float(np.mean(top == topt)))
            d["corr_W_xgb"].append([float(np.corrcoef(r[f"{kind}_W"][:, j], Wx[:, j])[0, 1]) for j in range(J)])
            d["refs_passing"].append(meta[kind]["refs_passing"])
    return out


def heloc_report():
    import run_heloc as rh
    files = sorted(glob.glob(str(RES / "heloc_rep*.npz")))
    if not files:
        return None
    H = dict(names=rh.NAMES)
    rs = [dict(np.load(f)) for f in files]
    ms = [json.load(open(f.replace(".npz", ".json"))) for f in files]
    H["n_reps"], H["n"] = len(rs), ms[0]["n"]
    H["auc"] = [m["auc"] for m in ms]
    H["band"] = [m["band_size"] for m in ms]
    Jh = len(rh.NAMES)
    cor = lambda a, b: [float(np.corrcoef(a[:, j], b[:, j])[0, 1]) for j in range(Jh)]
    H["corr_dir_proj"] = np.mean([cor(r["W_dir"], r["W_proj"]) for r in rs], 0).tolist()
    H["corr_proj_leafY"] = np.mean([cor(r["W_proj"], r["W_leafY"]) for r in rs], 0).tolist()
    H["corr_proj_leafX"] = np.mean([cor(r["W_proj"], r["W_leafX"]) for r in rs], 0).tolist()
    H["corr_leafY_glob"] = np.mean([cor(r["W_leafY"], r["W_glob"]) for r in rs], 0).tolist()
    H["sd"] = {k: np.mean([r[k].std(0) for r in rs], 0).tolist() for k in ["W_dir", "W_proj", "W_leafY", "W_leafX", "W_glob"]}
    H["meanabs"] = {k: np.mean([np.abs(r[k]).mean(0) for r in rs], 0).tolist() for k in ["W_dir", "W_proj", "W_leafY", "W_glob"]}
    # additivity of estimated evidence on declined
    H["sum_loo_vs_W"] = []
    tops, stab, rows = [], [], {k: [] for k in ["agree", "margin", "rel_margin", "unique", "shared", "unstable"]}
    for r in rs:
        allm = (1 << Jh) - 1
        Wl = r["Vh"][:, [allm]] - r["Vh"][:, [allm & ~(1 << j) for j in range(Jh)]]
        H["sum_loo_vs_W"].append([float(np.mean(Wl.sum(1) < r["Vh"][:, -1] - 0.05)),
                                  float(np.mean(Wl.sum(1) > r["Vh"][:, -1] + 0.05))])
        C = r["score_prod"]
        top = C.argmax(1)
        boots = [k for k in r if k.startswith("score_boot") and not k.endswith("_gap")]
        tb = np.stack([r[k].argmax(1) for k in boots], 1)
        agree = (tb == top[:, None]).mean(1)
        srt = np.sort(C, 1)
        idx = np.arange(len(top))
        rows["agree"].append(agree)
        rows["margin"].append(srt[:, -1] - srt[:, -2])
        rows["rel_margin"].append((srt[:, -1] - srt[:, -2]) / np.maximum(r["score_prod_gap"], 1e-6))
        rows["unique"].append(Wl[idx, top])
        rows["shared"].append(r["E_hat"][idx, top] - Wl[idx, top])
        rows["unstable"].append(agree < 0.8)
        tops.append(np.bincount(top, minlength=Jh) / len(top))
        stab.append(agree.mean())
        H.setdefault("C_mean", []).append(C.mean(0).tolist())
        H.setdefault("E_mean", []).append(r["E_hat"].mean(0).tolist())
        H.setdefault("Wl_mean", []).append(Wl.mean(0).tolist())
        H.setdefault("gap", []).append(float(r["score_prod_gap"].mean()))
    H["top_dist"] = np.mean(tops, 0).tolist()
    H["stability"] = stab
    R = {k: np.concatenate(v) for k, v in rows.items()}
    H["share_unstable"] = float(R["unstable"].mean())
    H["h3"] = dict(margin=h3_auc(R, ["rel_margin", "margin"], "unstable"),
                   margin_evidence=h3_auc(R, ["rel_margin", "margin", "unique", "shared"], "unstable"))
    q = np.quantile(R["shared"], [1 / 3, 2 / 3])
    g = np.digitize(R["shared"], q)
    H["stability_by_shared_tercile"] = [float(R["agree"][g == t].mean()) for t in range(3)]
    return H


# ------------------------------------------------------------------ figures
def fig_kernel(S):
    k = np.array(S["kappas"])
    selected = [json.loads((RES / f"sim_seed{s}.json").read_text())["kernel_selected"] for s in range(1, 6)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, g, title in zip(axes, ("Y", "X"), ("Geometry fitted to default", "Geometry fitted to block class")):
        grid = np.array(S[f"grid_{g}"])        # J, k, l
        best = grid.min(2)
        for j in range(J):
            line, = ax.plot(k, best[j], marker="o", ms=3, label=NAMES[j])
            choices = [m[f"{g}{j}"][0] for m in selected]
            values, counts = np.unique(choices, return_counts=True)
            mode = values[np.argmax(counts)]
            chosen_error = S["rmse"][f"Leaf {g}, selected"][j]
            ax.scatter([mode], [chosen_error], marker="*", s=85, color=line.get_color(), edgecolor="black", linewidth=0.4, zorder=5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(k)
        ax.set_xticklabels([str(v) for v in k])
        ax.set_xlabel(r"kernel exponent $\kappa$")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("RMSE against class-level oracle (logit)")
    axes[1].legend(fontsize=7, frameon=False, loc="upper left")
    fig.savefig(FIG / "kernel_exponent.pdf")
    plt.close(fig)


def fig_attribution(S):
    aa = S["aa"]
    Cm = np.array(aa["C_mean"]).mean(0)
    Ct = np.array(aa["C_true_mean"]).mean(0)
    Cp = np.array(aa["C_prox_mean"]).mean(0)
    Em = np.array(S["E_mean"])
    Wl = np.array(S["Wloo_mean_decl"])
    x = np.arange(J)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    panels = [([(Cm, "Fitted model, score band", "#216a8b"),
                (Cp, "Fitted model, proximity", "#7fb2c9"),
                (Ct, "True model, score band", "#9a9a9a")], "Score gaps to passing references"),
              ([(Em, "Order-averaged evidence", "#df8b36"),
                (Wl, "Last-entry evidence", "#699c65")], "Evidence in the population")]
    for ax, (series, title) in zip(axes, panels):
        w = 0.75 / len(series)
        for i, (v, lab, c) in enumerate(series):
            ax.bar(x + (i - (len(series)-1)/2) * w, v, w, label=lab, color=c)
        ax.set_xticks(x)
        ax.set_xticklabels(NAMES, fontsize=7, rotation=35, ha="right")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
        ax.set_ylim(-0.1, 1.2)
    axes[0].set_ylabel("mean over declined applicants (logit)")
    fig.tight_layout()
    fig.savefig(FIG / "attribution_vs_evidence.pdf")
    plt.close(fig)


def fig_revolving(runs):
    """Oracle and projection estimate of revolving evidence against utilization, by file age."""
    r = runs[0]
    from run_sim import SPLITS
    rng = np.random.default_rng(r["meta"]["seed"])
    X, y, p, b = bs.generate(sum(SPLITS.values()), rng)
    o = np.cumsum([0] + list(SPLITS.values()))
    Xt = X[o[4]:o[5]]
    Xq = Xt[r["qsel"]]
    u = Xq[:, 2]
    thin = bs.thin_file(Xq[:, 7]) > 0.5
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, k, title in zip(axes, ("W_true", "W_proj"), ("Oracle", "Projection estimate")):
        ax.scatter(u[~thin], r[k][~thin, 0], s=2, alpha=0.4, color="#216a8b", label="oldest trade $\\geq$ 48 months")
        ax.scatter(u[thin], r[k][thin, 0], s=2, alpha=0.5, color="#df8b36", label="oldest trade < 48 months")
        ax.set_xlabel("revolving utilization")
        ax.set_title(title, fontsize=9)
        ax.axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("revolving-block evidence (logit)")
    axes[0].legend(fontsize=7, frameon=False, markerscale=4)
    fig.savefig(FIG / "revolving_evidence.pdf")
    plt.close(fig)


def main():
    runs = load_sim()
    S, R = sim_report(runs)
    S["libraries"] = lib_report()
    H = heloc_report()
    fig_kernel(S)
    fig_attribution(S)
    fig_revolving(runs)
    (RES / "summary.json").write_text(json.dumps(dict(sim=S, heloc=H), indent=1, default=float))
    print(json.dumps(dict(sim={k: S[k] for k in ["auc", "rmse", "h3", "additivity", "selection", "aa"]}, heloc=H),
                     indent=1, default=lambda v: np.round(v, 3).tolist() if hasattr(v, "shape") else float(v))[:6000])


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------ tables
def write_tables():
    d = json.load(open(RES / "summary.json"))
    S, H = d["sim"], d["heloc"]
    hdr = " & ".join(["Revolving", "Delinq.", "File age", "Seeking", "Install.", "Public rec."])
    f2 = lambda v: " & ".join(f"{x:.2f}" for x in v)
    f2n = lambda v: " & ".join("--" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.2f}" for x in v)
    runs = [dict(np.load(RES / f"sim_seed{s}.npz")) for s in range(1, 6)]
    zero_pt = np.mean([np.sqrt(np.mean(r["W_true"] ** 2, axis=0)) for r in runs], axis=0)
    zero_bin = np.mean([np.sqrt(np.nanmean(r["Wb_true"] ** 2, axis=0)) for r in runs], axis=0)

    # evidence accuracy
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{Error of conditional evidence estimates against the oracle in the bureau-style simulation: root mean squared error (logit units), mean over five replicates, for about 3{,}350 combined evaluation applicants per replicate. Direct and projection estimates are compared with the pointwise evidence $W_j$ and the global and leaf-local estimates with the class-level evidence $W^B_j$; errors in different panels refer to different targets and are not compared with each other. The first two rows give the standard deviation of the oracle across applicants. Zero-evidence baselines assign zero to every applicant for each target. Replicate standard deviations of the RMSE are at most 0.02 for direct and projection estimates and at most 0.07 for leaf-local estimates.}",
         r"\label{tab:accuracy}",
         r"\resizebox{\linewidth}{!}{\begin{tabular}{@{}l" + "r" * J + r"@{}}\toprule",
         r"& " + hdr + r"\\\midrule",
         r"Oracle sd, pointwise $W_j$ & " + f2(S["sd_true_pt"]) + r"\\",
         r"Oracle sd, class-level $W^B_j$ & " + f2(S["sd_true_bin"]) + r"\\\midrule",
         r"Zero evidence, pointwise target & " + f2(zero_pt) + r"\\",
         r"Direct & " + f2(S["rmse"]["Direct"]) + r"\\",
         r"Projection & " + f2(S["rmse"]["Projection"]) + r"\\\midrule",
         r"Zero evidence, class-level target & " + f2(zero_bin) + r"\\",
         r"Global WoE of class & " + f2(S["rmse"]["Global WoE"]) + r"\\"]
    for lab in ["Leaf Y, $\\kappa=1$", "Leaf Y, selected", "Leaf X, $\\kappa=1$", "Leaf X, selected"]:
        nice = lab.replace("Leaf Y", "Leaf-local, default geometry").replace("Leaf X", "Leaf-local, class geometry").replace("selected", "selected $\\kappa$")
        L.append(nice + " & " + f2(S["rmse"][lab]) + r"\\")
    L += [r"\midrule", r"Sign agreement$^{a}$, projection & " + f2n(S["sign"]["Projection"]) + r"\\",
          r"Sign agreement, global WoE & " + f2n(S["sign"]["Global WoE"]) + r"\\",
          r"Sign agreement, leaf-local (class, selected) & " + f2n(S["sign"]["Leaf X, selected"]) + r"\\",
          r"Share with $|W_j|>0.25$ & " + f2(S["material_share"]) + r"\\",
          r"\bottomrule\end{tabular}}",
          r"\par\smallskip\raggedright\footnotesize $^{a}$Share of applicants with $|$oracle$|>0.25$ whose estimate has the oracle's sign; -- indicates 20 or fewer such applicants. The qualifying records can differ between the two targets.",
          r"\end{table}"]
    (TAB / "accuracy.tex").write_text("\n".join(L) + "\n")

    # coalition evidence and selection
    se = S["selection"]
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{Coalition evidence of the 400 declined applicants per replicate and the effect of the earlier approval policy, means over five replicates (logit units). The coalition excess is $H_j=E_j-W_j$. The absolute booked-minus-through-the-door difference $|W^b_j-W_j|$ is computed from the oracle for the test applicants; earlier-policy declines are applicants with booking probability $s(x)<0.5$ and current-model declines are applicants with $f(x)>c$. The last three rows give the RMSE of direct estimates fitted on booked accounts only, against the through-the-door and booked oracles, and of direct estimates fitted on all applicants against the through-the-door oracle.}",
         r"\label{tab:coalition}",
         r"\resizebox{\linewidth}{!}{\begin{tabular}{@{}l" + "r" * J + r"@{}}\toprule",
         r"& " + hdr + r"\\\midrule",
         r"Coalition evidence $E_j$, oracle mean & " + f2(S["E_mean"]) + r"\\",
         r"Leave-one-block-out $W_j$, oracle mean & " + f2(S["Wloo_mean_decl"]) + r"\\",
         r"Coalition excess $H_j$, oracle mean & " + f2(S["redund_mean"]) + r"\\",
         r"RMSE of projection estimate of $E_j$ & " + f2(S["E_rmse"]) + r"\\",
         r"Oracle sd of $E_j$ & " + f2(S["E_sd"]) + r"\\\midrule",
         r"$|W^b_j-W_j|$, all test applicants & " + f2(se["all"]) + r"\\",
         r"$|W^b_j-W_j|$, earlier-policy declines & " + f2(se["legacy_decl"]) + r"\\",
         r"$|W^b_j-W_j|$, current-model declines & " + f2(se["current_decl"]) + r"\\",
         r"Direct, booked fit vs.\ through-the-door oracle & " + f2(se["dir_vs_ttd"]) + r"\\",
         r"Direct, booked fit vs.\ booked oracle & " + f2(se["dir_vs_booked"]) + r"\\",
         r"Direct, all-applicant fit vs.\ through-the-door oracle & " + f2(se["dirttd_vs_ttd"]) + r"\\",
         r"\bottomrule\end{tabular}}", r"\end{table}"]
    (TAB / "coalition.tex").write_text("\n".join(L) + "\n")

    # adverse action
    aa = S["aa"]
    m = lambda k: np.mean(aa[k], 0)
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{Grouped baseline-Shapley attribution of the score gap for 400 declined applicants per replicate, means over five replicates. Attributions are in estimated default log odds; the mean total gap to the references is " + f"{np.mean(aa['mean_gap']):.2f}" + r". Leading-reason shares are the fractions of applicants whose largest attribution falls on each block.}",
         r"\label{tab:aarc}",
         r"\resizebox{\linewidth}{!}{\begin{tabular}{@{}l" + "r" * J + r"@{}}\toprule",
         r"& " + hdr + r"\\\midrule",
         r"Mean attribution, production model, score band & " + f2(m("C_mean")) + r"\\",
         r"Mean attribution, true log odds, score band & " + f2(m("C_true_mean")) + r"\\",
         r"Mean attribution, production model, proximity & " + f2(m("C_prox_mean")) + r"\\\midrule",
         r"Leading-reason share, production model & " + f2(m("dist_prod")) + r"\\",
         r"Leading-reason share, true log odds & " + f2(m("dist_true")) + r"\\",
         r"Leading-reason share, proximity references & " + f2(m("dist_prox")) + r"\\",
         r"\bottomrule\end{tabular}}", r"\end{table}"]
    (TAB / "aarc.tex").write_text("\n".join(L) + "\n")

    # HELOC
    hn = H["names"]
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{FICO HELOC data, ten random splits. Correlations are across test applicants within a split, averaged over splits. Means of $E_j$, $W_j$ and the attribution $C^f_j$ are over declined applicants. Estimated evidence uses the projection estimator.}",
         r"\label{tab:heloc}",
         r"\resizebox{\linewidth}{!}{\begin{tabular}{@{}l" + "r" * len(hn) + r"@{}}\toprule",
         r"& " + " & ".join(["Delinq.", "File age", "Trades", "Inquiries", "Revolving", "Install."]) + r"\\\midrule",
         r"Corr(direct, projection) & " + f2(H["corr_dir_proj"]) + r"\\",
         r"Corr(projection, leaf-local default geometry) & " + f2(H["corr_proj_leafY"]) + r"\\",
         r"Corr(leaf-local default geometry, global WoE) & " + f2(H["corr_leafY_glob"]) + r"\\\midrule",
         r"Mean coalition evidence $E_j$ & " + f2(np.mean(H["E_mean"], 0)) + r"\\",
         r"Mean leave-one-block-out $W_j$ & " + f2(np.mean(H["Wl_mean"], 0)) + r"\\",
         r"Mean attribution $C^f_j$ & " + f2(np.mean(H["C_mean"], 0)) + r"\\",
         r"Leading-reason share & " + f2(H["top_dist"]) + r"\\",
         r"\bottomrule\end{tabular}}", r"\end{table}"]
    (TAB / "heloc.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    write_tables()
