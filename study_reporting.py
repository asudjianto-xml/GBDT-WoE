"""Report-only interpretation checks for the conditional-WoE manuscript.

Reads archived simulations; never fits models or overwrites study results.
usage: python study_reporting.py --report-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parent
RES, TAB, SEC = ROOT / "results", ROOT / "tables", ROOT / "sections"
NAMES = ["Revolving", "Delinquency", "File age", "Credit seeking", "Installment", "Public record"]


def interpretation_checks():
    runs = [dict(np.load(RES / f"sim_seed{s}.npz")) for s in range(1, 6)]
    out = {"zero_pointwise": [], "zero_class": [], "decline_share": [], "methods": {}}
    for r in runs:
        out["zero_pointwise"].append(np.sqrt(np.mean(r["W_true"] ** 2, axis=0)))
        out["zero_class"].append(np.sqrt(np.nanmean(r["Wb_true"] ** 2, axis=0)))
        out["decline_share"].append(float((r["lF"] > r["cut"]).mean()))
    for method, key in [("Direct", "W_dir"), ("Projection", "W_proj")]:
        rows = []
        for r in runs:
            ef = r["lF"] - r["eta_q"]
            ew = r[key] - r["W_true"]
            er = ef[:, None] - ew
            full_mse = np.repeat(np.mean(ef ** 2), 6)
            reduced_mse = np.mean(er ** 2, axis=0)
            twice_cross = 2 * np.mean(ef[:, None] * er, axis=0)
            evidence_mse = np.mean(ew ** 2, axis=0)
            np.testing.assert_allclose(full_mse + reduced_mse - twice_cross, evidence_mse, atol=1e-12)
            rows.append(dict(full_mse=full_mse, reduced_mse=reduced_mse, twice_cross=twice_cross,
                             evidence_mse=evidence_mse, reduced_rmse=np.sqrt(reduced_mse),
                             evidence_rmse=np.sqrt(evidence_mse),
                             correlation=np.array([np.corrcoef(ef, er[:, j])[0, 1] for j in range(6)]),
                             decline_rmse=np.sqrt(np.mean(ew[r["a_pos"]] ** 2, axis=0))))
        out["methods"][method] = {k: np.mean([r[k] for r in rows], axis=0).tolist() for k in rows[0]}
    for key in ["zero_pointwise", "zero_class"]:
        out[key] = np.mean(out[key], axis=0).tolist()
    out["decline_share"] = float(np.mean(out["decline_share"]))

    # The selected case's true file-age contribution has a closed form.
    # Reconstruct only its observed reference records, with no model fitting.
    import bureau_sim as bs
    from run_sim import SPLITS
    c = json.loads((RES / "revision_case.json").read_text())["case"]
    seed, i = c["seed"], c["index"]
    r = runs[seed - 1]
    X, _, _, _ = bs.generate(sum(SPLITS.values()), np.random.default_rng(seed))
    offsets = np.cumsum([0] + list(SPLITS.values()))
    ref_index = list(SPLITS).index("ref")
    refs = X[offsets[ref_index]:offsets[ref_index + 1]]
    x = r["Xa"][i]
    tx = expit((48 - x[7]) / 8)
    out["case"] = {"seed": seed, "index": i}
    for policy in ["score", "proximity"]:
        rr = refs[r[f"refs_{policy}"][i]]
        tr = expit((48 - rr[:, 7]) / 8)
        contribution = float(np.mean((tx - tr) * (0.6 - 1.7 * (x[2] + rr[:, 2]))))
        np.testing.assert_allclose(contribution, r[f"{policy}_true"][i, 2], atol=1e-12)
        out["case"][policy] = contribution
    return out


def write_interpretation(d):
    direct, proj = d["methods"]["Direct"], d["methods"]["Projection"]
    text = (
        "Projection improves both components of the evidence comparison. "
        f"For file age, reduced-posterior RMSE falls from {direct['reduced_rmse'][2]:.3f} to {proj['reduced_rmse'][2]:.3f}, "
        f"while correlation of full and reduced errors rises from {direct['correlation'][2]:.3f} to {proj['correlation'][2]:.3f}. "
        f"For installment debt, reduced-posterior RMSE changes only from {direct['reduced_rmse'][4]:.3f} to {proj['reduced_rmse'][4]:.3f}, "
        f"but the error correlation rises from {direct['correlation'][4]:.3f} to {proj['correlation'][4]:.3f}. "
        "The cross-product decomposition in Appendix~\\ref{app:errors} confirms that greater cancellation contributes to the smaller evidence error. "
        "An accurate difference therefore does not imply equally accurate full and reduced risk estimates.\n\n"
        f"The combined evaluation sample contains {100*d['decline_share']:.1f} percent current-model declines, "
        "rather than the 20 percent targeted by the policy. "
        "Table~\\ref{tab:decline_accuracy} reports errors separately for the 400 sampled declines per replicate. "
        "Errors are larger among declines for every block, although projection retains lower error than direct refitting.\n"
    )
    (SEC / "error_findings.tex").write_text(text)
    (SEC / "case_mechanism.tex").write_text(
        "The applicant's oldest trade is 294 months, yet file age contributes "
        "$+0.40$ against score-band references and approximately zero against proximity references. "
        "In this deliberately constructed risk model, younger files receive lower risk at high utilization. "
        "For a reference $r$, the true-model file-age allocation is "
        "$\\{\\tau(x)-\\tau(r)\\}\\{0.6-1.7(u(x)+u(r))\\}$, obtained by averaging whether file age enters before or after revolving credit. "
        f"Averaging over the saved references gives {d['case']['score']:.3f} and {d['case']['proximity']:.3f} logit units, respectively, "
        "matching the exact true-model allocations. The positive fitted contribution against the score band therefore follows the constructed interaction; "
        "it is not evidence that a long credit history is generally adverse.\n"
    )
    (SEC / "diagnostic_details.tex").write_text(
        "The zero-evidence baseline has RMSE $\\sqrt{\\E[W_j^2]}$ for the pointwise target and "
        "$\\sqrt{\\E[(W_j^B)^2]}$ for the class-level target. It measures the error of assigning no conditional evidence to any applicant. "
        "Table~\\ref{tab:error_decomposition} evaluates Eq.~\\eqref{eq:error_difference} on the same combined evaluation records as the main accuracy table. "
        "Entries are means of replicate mean squared errors and cross-products, so the identity holds across each row before rounding. "
        "Positive cross-products represent cancellation in the evidence difference; correlation alone does not quantify that contribution.\n"
    )
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{Error in the evidence difference, averaged over five replicates on the combined evaluation sample. The first two terms are squared full and reduced log-odds errors; subtracting twice their cross-product gives the evidence MSE. Values are in squared log-odds units.}",
             r"\label{tab:error_decomposition}",
             r"\begin{tabular}{@{}llrrrr@{}}\toprule",
             r"Block & Estimator & $\E[e_F^2]$ & $\E[e_{-j}^2]$ & $2\E[e_Fe_{-j}]$ & Evidence MSE\\\midrule"]
    for j, name in enumerate(NAMES):
        for method in ["Direct", "Projection"]:
            m = d["methods"][method]
            lines.append(f"{name if method == 'Direct' else ''} & {method} & " + " & ".join(
                f"{m[k][j]:.4f}" for k in ["full_mse", "reduced_mse", "twice_cross", "evidence_mse"]) + r"\\")
    lines += [r"\bottomrule\end{tabular}", r"\end{table}"]
    (TAB / "error_decomposition.tex").write_text("\n".join(lines) + "\n")
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{Pointwise conditional-evidence RMSE among the 400 sampled declines per replicate, compared with the combined evaluation sample. Entries are means of replicate RMSE in logit units. The decline sample is selected by the original production model and cutoff.}",
             r"\label{tab:decline_accuracy}", r"\resizebox{\linewidth}{!}{\begin{tabular}{@{}lrrrrrr@{}}\toprule",
             r"& Revolving & Delinq. & File age & Seeking & Install. & Public rec.\\\midrule"]
    for method in ["Direct", "Projection"]:
        for label, key in [("combined", "evidence_rmse"), ("sampled declines", "decline_rmse")]:
            lines.append(f"{method}, {label} & " + " & ".join(f"{v:.3f}" for v in d["methods"][method][key]) + r"\\")
    lines += [r"\bottomrule\end{tabular}}", r"\end{table}"]
    (TAB / "decline_accuracy.tex").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true", required=True)
    parser.parse_args()
    d = interpretation_checks()
    (ROOT / "build").mkdir(exist_ok=True)
    (ROOT / "build/interpretation_checks.json").write_text(json.dumps(d, indent=2) + "\n")
    write_interpretation(d)
    import report
    import revision_report
    report.write_tables()
    s = json.loads((RES / "summary.json").read_text())["sim"]
    report.fig_kernel(s)
    report.fig_attribution(s)
    revision_report.typeset_minus()
    print("Regenerated interpretation prose, tables and comparison figures from archived results.")


if __name__ == "__main__":
    main()
