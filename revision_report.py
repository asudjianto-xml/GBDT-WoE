"""Analyses added in revision: H3 with uncertainty, selection subgroups, case and diagnostics.

Writes results/revision_summary.json and tables/h3.tex, tables/case.tex.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import bureau_sim as bs

ROOT = Path(__file__).resolve().parent
RES, TAB = ROOT / "results", ROOT / "tables"
J = bs.J
N_BOOT_CI = 2000


def oof(X, y, groups, kind, n_splits):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        if kind == "logistic":
            m = LogisticRegression(max_iter=3000).fit((X[tr] - mu) / sd, y[tr])
            p[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
        else:
            m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0)
            p[te] = m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def paired_ci(y, p0, p1, clusters, rng):
    """Cluster bootstrap of AUC(p1) - AUC(p0) over out-of-fold predictions."""
    uc, inv = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uc))]
    d = []
    for _ in range(N_BOOT_CI):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(uc), len(uc))])
        if y[idx].min() == y[idx].max():
            continue
        d.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p0[idx]))
    return np.quantile(d, [0.025, 0.975]).tolist()


def h3(rows, feats, outcome, groups, clusters, n_splits, rng):
    y = rows[outcome].astype(int)
    out = dict(rate=float(y.mean()))
    for kind in ("logistic", "gbdt"):
        P = {name: oof(np.column_stack([rows[f] for f in fs]), y, groups, kind, n_splits) for name, fs in feats.items()}
        res = {name: float(roc_auc_score(y, p)) for name, p in P.items()}
        for name in feats:
            if name != "margin":
                res[f"diff_{name}"] = res[name] - res["margin"]
                res[f"ci_{name}"] = paired_ci(y, P["margin"], P[name], clusters, rng)
        out[kind] = res
    return out


def sim_rows():
    rows = {k: [] for k in ["agree", "margin", "rel_margin", "unique", "diffE", "uniq_true", "diffE_true",
                            "wrong", "seed", "uid"]}
    for f in sorted(glob.glob(str(RES / "sim_seed[0-9].npz"))):
        r = np.load(f)
        seed = int(Path(f).stem[-1])
        C = r["score_prod"]
        top = C.argmax(1)
        tb = [r[f"score_boot{b}"].argmax(1) for b in range(10)]
        extra = RES / f"revision_refits_seed{seed}.npz"
        if extra.exists():
            tb += list(np.load(extra)["C"].argmax(2))
        tb = np.stack(tb, 1)
        srt = np.sort(C, 1)
        idx = np.arange(len(top))
        Wp = r["W_proj"][r["a_pos"]]
        rows["agree"].append((tb == top[:, None]).mean(1))
        rows["margin"].append(srt[:, -1] - srt[:, -2])
        rows["rel_margin"].append((srt[:, -1] - srt[:, -2]) / r["score_prod_gap"])
        rows["unique"].append(Wp[idx, top])
        rows["diffE"].append(r["E_hat"][idx, top] - r["W_loo_hat_a"][idx, top])
        rows["uniq_true"].append(r["W_loo_true_a"][idx, top])
        rows["diffE_true"].append(r["E_true"][idx, top] - r["W_loo_true_a"][idx, top])
        rows["wrong"].append(top != r["score_true"].argmax(1))
        rows["seed"].append(np.full(len(top), seed))
        rows["uid"].append(seed * 100000 + r["dsel"])
        n_refits = tb.shape[1]
    R = {k: np.concatenate(v) for k, v in rows.items()}
    return R, n_refits


def heloc_rows():
    import run_heloc as rh
    Jh = len(rh.NAMES)
    allm = (1 << Jh) - 1
    rows = {k: [] for k in ["agree", "margin", "rel_margin", "unique", "diffE", "rep", "uid"]}
    diag = {k: [] for k in ["ny1", "ess1", "absdev", "lam_kappa"]}
    for rep in range(10):
        r = np.load(RES / f"heloc_rep{rep}.npz")
        m = json.load(open(RES / f"heloc_rep{rep}.json"))
        C = r["score_prod"]
        top = C.argmax(1)
        tb = np.stack([r[f"score_boot{b}"].argmax(1) for b in range(10)], 1)
        srt = np.sort(C, 1)
        idx = np.arange(len(top))
        Wl = r["Vh"][:, [allm]] - r["Vh"][:, [allm & ~(1 << j) for j in range(Jh)]]
        rows["agree"].append((tb == top[:, None]).mean(1))
        rows["margin"].append(srt[:, -1] - srt[:, -2])
        rows["rel_margin"].append((srt[:, -1] - srt[:, -2]) / r["score_prod_gap"])
        rows["unique"].append(Wl[idx, top])
        rows["diffE"].append(r["E_hat"][idx, top] - Wl[idx, top])
        rows["rep"].append(np.full(len(top), rep))
        rows["uid"].append(r["ids_decl"])
        diag["ny1"].append(np.median(r["ny_Y"][:, :, 1], 0))
        diag["ess1"].append(np.median(r["ess_Y"][:, :, 1], 0))
        diag["absdev"].append(np.mean(np.abs(r["W_leafY"] - r["W_glob"]), 0))
        diag["lam_kappa"].append([m["kernel_selected"][f"Y{j}"] for j in range(Jh)])
    R = {k: np.concatenate(v) for k, v in rows.items()}
    D = {k: (np.mean(v, 0).tolist() if k != "lam_kappa" else v) for k, v in diag.items()}
    D["sd_glob"] = np.mean([np.load(RES / f"heloc_rep{rep}.npz")["W_glob"].std(0) for rep in range(10)], 0).tolist()
    D["n_unique_declined"] = int(len(np.unique(R["uid"])))
    D["n_declined_rows"] = int(len(R["uid"]))
    return R, D


def selection_subgroups():
    out = {k: [] for k in ["abs_current_decl", "signed_current_decl", "abs_legacy_decl", "signed_legacy_decl",
                           "abs_all", "signed_all", "share_current_decl", "share_legacy_decl", "overlap"]}
    for f in sorted(glob.glob(str(RES / "sim_seed[0-9].npz"))):
        r = np.load(f)
        d = r["W_booked"] - r["W_true"]
        cur = r["lF"] > r["cut"]
        leg = r["legacy_q"] > bs.LEGACY_T          # earlier-policy approval probability below 0.5
        out["abs_current_decl"].append(np.abs(d[cur]).mean(0))
        out["signed_current_decl"].append(d[cur].mean(0))
        out["abs_legacy_decl"].append(np.abs(d[leg]).mean(0))
        out["signed_legacy_decl"].append(d[leg].mean(0))
        out["abs_all"].append(np.abs(d).mean(0))
        out["signed_all"].append(d.mean(0))
        out["share_current_decl"].append(cur.mean())
        out["share_legacy_decl"].append(leg.mean())
        out["overlap"].append((cur & leg).sum() / cur.sum())
    return {k: np.mean(v, 0).tolist() for k, v in out.items()}


def main():
    rng = np.random.default_rng(0)
    S = {}
    R, nref = sim_rows()
    S["n_refits"] = int(nref)
    S["n_applicants"] = int(len(R["agree"]))
    feats = {"margin": ["margin", "rel_margin"], "estimated": ["margin", "rel_margin", "unique", "diffE"],
             "oracle": ["margin", "rel_margin", "uniq_true", "diffE_true"]}
    S["h3_sim"] = {}
    for thr in (0.7, 0.8, 0.9):
        R[f"unstable{thr}"] = R["agree"] < thr
        S["h3_sim"][f"unstable{thr}"] = h3(R, feats, f"unstable{thr}", R["seed"], R["uid"], 5, rng)
    S["h3_sim"]["wrong"] = h3(R, feats, "wrong", R["seed"], R["uid"], 5, rng)
    S["agree_mean"] = float(R["agree"].mean())
    q = np.quantile(R["rel_margin"], [1 / 3, 2 / 3])
    g = np.digitize(R["rel_margin"], q)
    S["agree_by_margin_tercile"] = [float(R["agree"][g == t].mean()) for t in range(3)]
    q = np.quantile(R["diffE"], [1 / 3, 2 / 3])
    g = np.digitize(R["diffE"], q)
    S["agree_by_diffE_tercile"] = [float(R["agree"][g == t].mean()) for t in range(3)]

    Hr, Hd = heloc_rows()
    Hr["unstable0.8"] = Hr["agree"] < 0.8
    feats_h = {"margin": ["margin", "rel_margin"], "estimated": ["margin", "rel_margin", "unique", "diffE"]}
    S["h3_heloc"] = h3(Hr, feats_h, "unstable0.8", Hr["uid"], Hr["uid"], 5, rng)
    S["heloc_diag"] = Hd
    S["selection"] = selection_subgroups()
    for name in ("oracle", "install", "logodds", "samplesize", "case"):
        p = RES / f"revision_{name}.json"
        if p.exists():
            S[name] = json.load(open(p))
    (RES / "revision_summary.json").write_text(json.dumps(S, indent=1, default=float))
    write_tables(S)
    write_extra_tables(S)
    print(json.dumps({k: S[k] for k in ("n_refits", "h3_sim", "h3_heloc", "selection")}, indent=1, default=float)[:5000])


def write_tables(S):
    f3 = lambda v: f"{v:.3f}"
    ci = lambda c: f"[{c[0]:+.3f}, {c[1]:+.3f}]"
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{Prediction of an unstable leading reason from the attribution margin alone and with the leading block's evidence added. Out-of-fold AUC; the simulation uses leave-one-replicate-out folds and " + str(S["n_refits"]) + r" bootstrap refits per replicate; HELOC uses folds grouped by applicant and 10 refits. Intervals are 95 percent cluster-bootstrap intervals (2{,}000 resamples of applicants) for the paired difference in out-of-fold AUC; they do not include refitting of the classifiers.}",
         r"\label{tab:h3}",
         r"\resizebox{\linewidth}{!}{\begin{tabular}{@{}llrrrrr@{}}\toprule",
         r"Data and outcome & Classifier & Rate & Margin & + estimated evidence & Difference (95\% interval) & + oracle evidence\\\midrule"]
    rows = [("Simulation, agreement $<0.7$", S["h3_sim"]["unstable0.7"]),
            ("Simulation, agreement $<0.8$", S["h3_sim"]["unstable0.8"]),
            ("Simulation, agreement $<0.9$", S["h3_sim"]["unstable0.9"]),
            ("Simulation, differs from true model", S["h3_sim"]["wrong"]),
            ("HELOC, agreement $<0.8$", S["h3_heloc"])]
    for lab, d in rows:
        for kind in ("logistic", "gbdt"):
            x = d[kind]
            orc = f3(x["oracle"]) if "oracle" in x else "--"
            L.append(f"{lab if kind == 'logistic' else ''} & {'logistic' if kind == 'logistic' else 'GBDT'} & "
                     f"{d['rate']:.3f} & {f3(x['margin'])} & {f3(x['estimated'])} & {x['diff_estimated']:+.3f} {ci(x['ci_estimated'])} & {orc}\\\\")
    L += [r"\bottomrule\end{tabular}}", r"\end{table}"]
    (TAB / "h3.tex").write_text("\n".join(L) + "\n")




def write_extra_tables(S):
    names = ["Revolving", "Delinq.", "File age", "Seeking", "Install.", "Public rec."]
    ss = S["samplesize"]
    sizes = [2500, 10000, 30000, 100000]
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{Leaf-local estimation with evidence samples of different sizes, replicate 1. RMSE against the class-level oracle for the class geometry at the selected $(\kappa,\lambda)$, and for global WoE computed from the same subsample. The last row gives the median default-class effective sample size at 100{,}000 records.}",
         r"\label{tab:samplesize}",
         r"\resizebox{\linewidth}{!}{\begin{tabular}{@{}l" + "r" * 6 + r"@{}}\toprule",
         r"Evidence records & " + " & ".join(names) + r"\\\midrule"]
    for n in sizes:
        L.append(f"{n:,}".replace(",", "{,}") + " & " + " & ".join(f"{ss[f'X{j}_{n}']['rmse_sel']:.2f}" for j in range(6)) + r"\\")
    L.append(r"Global WoE, 100{,}000 & " + " & ".join(f"{ss[f'X{j}_100000']['rmse_glob']:.2f}" for j in range(6)) + r"\\")
    L.append(r"Global WoE, 2{,}500 & " + " & ".join(f"{ss[f'X{j}_2500']['rmse_glob']:.2f}" for j in range(6)) + r"\\")
    L.append(r"Default-class ESS, 100{,}000 & " + " & ".join(f"{ss[f'X{j}_100000']['ess1_median']:.0f}" for j in range(6)) + r"\\")
    L += [r"\bottomrule\end{tabular}}", r"\end{table}"]
    (TAB / "samplesize.tex").write_text("\n".join(L) + "\n")

    c = S["case"]["case"]
    rec, rs, rp = c["record"], c["refs"]["score"], c["refs"]["proximity"]
    rows = [("Revolving utilization", "revolving_utilization", "{:.2f}"), ("Revolving limit (\\$)", "revolving_limit", "{:,.0f}"),
            ("Revolving balance (\\$)", "revolving_balance", "{:,.0f}"), ("Accounts 30+ days past due", "trades_30plus_dpd", "{:.2f}"),
            ("Accounts 60+ days past due", "trades_60plus_dpd", "{:.2f}"), ("Accounts 90+ days past due", "trades_90plus_dpd", "{:.2f}"),
            ("Oldest trade (months)", "oldest_trade_months", "{:.0f}"), ("Inquiries in 6 months", "inquiries_6m", "{:.2f}"),
            ("Installment fraction outstanding", "installment_fraction", "{:.2f}"), ("Public record", "public_record_flag", "{:.2f}")]
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{A declined applicant from replicate 1 whose leading block changes with the reference policy. Reference columns give means over the 50 references of each policy; the share of references with any account 30+ days past due is " +
         f"{rs['share_any_30dpd']:.2f} under the score-band policy and {rp['share_any_30dpd']:.2f} under the proximity policy" +
         r". The lower panel gives block attributions $C^f_j$ of the production model and the applicant's oracle evidence $W_j$ and $E_j$ (logit units).}",
         r"\label{tab:case}",
         r"\begin{tabular}{@{}lrrr@{}}\toprule",
         r"& Applicant & Score-band references & Proximity references\\\midrule"]
    for lab, k, fmt in rows:
        L.append(f"{lab} & {fmt.format(rec[k])} & {fmt.format(rs[k])} & {fmt.format(rp[k])}".replace(",", "{,}") + r"\\")
    L.append(r"\midrule & Attribution, score band & Attribution, proximity & Oracle $W_j$ / $E_j$\\\midrule")
    full = ["Revolving", "Delinquency", "File age", "Credit seeking", "Installment", "Public record"]
    for j in range(6):
        sg = lambda v: f"{v:+.2f}".replace("-0.00", "0.00").replace("+0.00", "0.00")
        L.append(f"{full[j]} & {sg(rs['C'][j])} & {sg(rp['C'][j])} & {sg(c['W_true'][j])} / {sg(c['E_true'][j])}" + r"\\")
    L.append(f"Total gap & {rs['gap']:.2f} & {rp['gap']:.2f} & \\\\")
    L += [r"\bottomrule\end{tabular}", r"\end{table}"]
    (TAB / "case.tex").write_text("\n".join(L) + "\n")




def typeset_minus():
    """Typeset negative numbers in generated tables with a minus sign."""
    import re
    for p in TAB.glob("*.tex"):
        s = p.read_text()
        s2 = re.sub(r"(?<=[\s\[(/&])-(?=\d)", r"$-$", s)
        s2 = s2.replace("$-$0.000 ", "0.000 ").replace("+0.000 ", "0.000 ").replace("+0.000]", "0.000]")
        if s2 != s:
            p.write_text(s2)




def write_calibration_table():
    C = json.load(open(RES / "calibration.json"))
    sims = [C[f"sim{s}"]["full"] for s in range(1, 6)]
    hel = [C[f"heloc{k}"] for k in range(10)]
    rng_ = lambda v, d=2: (f"{min(v):.{d}f} to {max(v):.{d}f}").replace("-", "$-$")
    rows = [("Platt slope $b$", lambda f: f["platt_b"], 2),
            ("Calibration slope before Platt", lambda f: f["raw_outcome"]["cal_slope"], 2),
            ("Calibration slope", lambda f: f["outcome"]["cal_slope"], 2),
            ("Calibration intercept", lambda f: f["outcome"]["cal_intercept"], 2),
            ("Expected calibration error (deciles)", lambda f: f["outcome"]["ece"], 3),
            ("Mean predicted probability", lambda f: f["outcome"]["mean_pred"], 3),
            ("Observed default rate", lambda f: f["outcome"]["mean_obs"], 3)]
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{Calibration of the production model on the test sample: ranges over five simulation replicates and ten HELOC splits. The calibration slope and intercept are from a logistic regression of the outcome on the model's log odds (intercept with slope fixed at one). The lower panel compares the simulation model with the true log odds, for all test applicants and for applicants above the cutoff.}",
         r"\label{tab:calibration}",
         r"\begin{tabular}{@{}lrr@{}}\toprule",
         r"& Simulation & HELOC\\\midrule"]
    for lab, fn, d in rows:
        L.append(f"{lab} & {rng_([fn(f) for f in sims], d)} & {rng_([fn(h) for h in hel], d)}" + r"\\")
    L.append(r"\midrule")
    tr = [("Bias against true log odds, all", lambda f: f["truth"]["bias_logit"]),
          ("Slope of true on fitted log odds, all", lambda f: f["truth"]["slope_true_on_fit"]),
          ("Bias against true log odds, declined", lambda f: f["declined"]["truth"]["bias_logit"]),
          ("Slope of true on fitted log odds, declined", lambda f: f["declined"]["truth"]["slope_true_on_fit"]),
          ("Mean predicted probability, declined", lambda f: f["declined"]["mean_pred"]),
          ("Mean true probability, declined", lambda f: f["declined"]["mean_true"])]
    for lab, fn in tr:
        L.append(f"{lab} & {rng_([fn(f) for f in sims], 3)} & --" + r"\\")
    L += [r"\bottomrule\end{tabular}", r"\end{table}"]
    (TAB / "calibration.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
    write_calibration_table()
    typeset_minus()
