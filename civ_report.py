"""Conditional information value (Section 4.4) from the archived results.

usage: python civ_report.py
Reads results/sim_seed{1..5}.npz and results/heloc_rep{0..9}.npz; refits nothing.
Writes results/civ_summary.json and, when the coalition arrays are present, tables/civ.tex.

The information value of a quantity g is E[g | Y=1] - E[g | Y=0].

Simulation. Marginal, conditional, coalition and joint values use the N_CIV random test
applicants saved by run_sim.py (Vt_civ, Vh_civ), with class means weighted by the true default
probability. The *_q entries repeat the conditional and joint values on the query set (3,000
random test records plus 400 sampled declines): non-declined query records all come from the
random draw, so it held 3000 - n_nondeclined declines, and the two strata are reweighted to
that decline rate.

HELOC. Class means over all test records with observed outcomes; coalition values from Vq.
"""

from __future__ import annotations

import json
from math import factorial
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent
RES, TAB = ROOT / "results", ROOT / "tables"
N_RANDOM = 3000
SIM_COLS = ["Revolving", "Delinq.", "File age", "Seeking", "Install.", "Public rec."]
HELOC_COLS = ["Delinquency", "File age", "Trade vol.", "Inquiries", "Revolving", "Installment"]

Array = NDArray[np.float64]
Summary = dict[str, dict[str, Any]]


def class_gap(g: NDArray[Any], w1: Array, w0: Array) -> Array:
    """Class-mean gap E[g | Y=1] - E[g | Y=0] for each column of g.

    Args:
        g: Values, one row per record (a vector is treated as one column).
        w1: Weight of each record in the default class.
        w0: Weight of each record in the non-default class.

    Returns:
        One gap per column of g.
    """
    G = np.asarray(g, dtype=np.float64).reshape(len(g), -1)
    return (w1 @ G) / w1.sum() - (w0 @ G) / w0.sum()


def shapley_weights(J: int) -> Array:
    """Matrix S such that V @ S.T gives the Shapley values of a game V stored by bitmask.

    Args:
        J: Number of blocks.

    Returns:
        Array of shape (J, 2**J).
    """
    S = np.zeros((J, 1 << J))
    for j in range(J):
        for mask in range(1 << J):
            if mask >> j & 1:
                continue
            k = mask.bit_count()
            w = factorial(k) * factorial(J - k - 1) / factorial(J)
            S[j, mask | 1 << j] += w
            S[j, mask] -= w
    return S


def game_values(V: Array, w1: Array, w0: Array) -> dict[str, Any]:
    """Marginal, leave-one-block-out, coalition and joint information values of a game.

    Args:
        V: Coalition evidence V(S) by bitmask, shape (n, 2**J), with V[:, 0] = 0.
        w1: Default-class weights.
        w0: Non-default-class weights.

    Returns:
        Dictionary of per-block arrays and the scalar joint value.
    """
    J = int(np.log2(V.shape[1]))
    full = (1 << J) - 1
    return {
        "marginal": class_gap(V[:, [1 << j for j in range(J)]], w1, w0),
        "conditional": class_gap(V[:, [full]] - V[:, [full & ~(1 << j) for j in range(J)]], w1, w0),
        "coalition": class_gap(V @ shapley_weights(J).T, w1, w0),
        "joint": float(class_gap(V[:, full], w1, w0)[0]),
    }


def sim_seed(path: Path) -> dict[str, Any]:
    """Information values for one simulation replicate."""
    d = np.load(path, allow_pickle=False)
    dec = d["lF"] > d["cut"]
    n_nd = int((~dec).sum())
    rate = (N_RANDOM - n_nd) / N_RANDOM
    w = np.where(dec, rate / dec.sum(), (1 - rate) / n_nd)
    w1, w0 = w * d["ptrue_q"], w * (1 - d["ptrue_q"])
    out: dict[str, Any] = {
        "decline_rate": rate,
        "conditional_oracle_q": class_gap(d["W_true"], w1, w0),
        "conditional_projection_q": class_gap(d["W_proj"], w1, w0),
        "conditional_direct_q": class_gap(d["W_dir"], w1, w0),
        "joint_oracle_q": float(class_gap(d["eta_q"], w1, w0)[0]),
        "joint_fitted_q": float(class_gap(d["lF"], w1, w0)[0]),
    }
    if "Vt_civ" in d.files:
        p = d["ptrue_civ"]
        for name in ("oracle", "projection"):
            V = d["Vt_civ" if name == "oracle" else "Vh_civ"]
            out.update({f"{k}_{name}": v for k, v in game_values(V, p, 1 - p).items()})
    return out


def heloc_rep(path: Path) -> dict[str, Any]:
    """Information values for one HELOC split."""
    d = np.load(path, allow_pickle=False)
    y = d["yq"].astype(np.float64)
    out: dict[str, Any] = {
        "conditional_projection_q": class_gap(d["W_proj"], y, 1 - y),
        "conditional_direct_q": class_gap(d["W_dir"], y, 1 - y),
        "joint_fitted_q": float(class_gap(d["lF"], y, 1 - y)[0]),
    }
    if "Vq" in d.files:
        out.update({f"{k}_projection": v for k, v in game_values(d["Vq"], y, 1 - y).items()})
    return out


def summarize(runs: list[dict[str, Any]]) -> Summary:
    """Mean and standard deviation across replicates of every entry."""
    out: Summary = {}
    for k in runs[0]:
        v = np.array([r[k] for r in runs], dtype=np.float64)
        out[k] = {"mean": v.mean(0).tolist(), "sd": v.std(0, ddof=1).tolist()}
    return out


def table(sim: Summary, hel: Summary, n_sim: int, n_hel: int) -> str:
    """LaTeX source of the conditional information value table."""

    def m(s: Summary, k: str) -> Array:
        return np.atleast_1d(np.asarray(s[k]["mean"], dtype=np.float64))

    def row(label: str, v: Array) -> str:
        return label + " & " + " & ".join(f"{x:.2f}" for x in v) + r"\\"

    def span(label: str, text: str) -> str:
        return label + r" & \multicolumn{6}{c}{" + text + r"}\\"

    def sums(s: Summary, sfx: str) -> str:
        return " / ".join(f"{m(s, f'{k}_{sfx}').sum():.2f}" for k in ("marginal", "conditional", "coalition"))

    sd_max = max(max(sim[k]["sd"]) for k in ("conditional_oracle", "coalition_oracle"))
    caption = (
        r"\caption{Conditional information value (natural units). Simulation: oracle and projection values on "
        r"1{,}000 random test applicants per replicate, class means weighted by the true default probability; "
        f"means over {n_sim} replicates, replicate standard deviations at most {sd_max:.2f} for the oracle values. "
        f"HELOC: projection estimates over the test records, means over {n_hel} splits. Marginal values use the "
        r"evidence of the block alone, $V_x(\{j\})$, or its projection estimate.}"
    )
    lines = [
        r"\begin{table}[t]\centering\small\color{dbblue}",
        caption,
        r"\label{tab:civ}",
        r"\resizebox{\linewidth}{!}{\begin{tabular}{@{}lrrrrrr@{}}\toprule",
        r"\multicolumn{7}{@{}l}{\emph{Simulation}}\\",
        " & ".join(["", *SIM_COLS]) + r"\\\midrule",
        row(r"Marginal IV, oracle", m(sim, "marginal_oracle")),
        row(r"Conditional IV $\mathrm{IV}_j$, oracle", m(sim, "conditional_oracle")),
        row(r"Coalition IV $\mathrm{IV}^E_j$, oracle", m(sim, "coalition_oracle")),
        row(r"Conditional IV $\mathrm{IV}_j$, projection", m(sim, "conditional_projection")),
        row(r"Coalition IV $\mathrm{IV}^E_j$, projection", m(sim, "coalition_projection")),
        r"\midrule",
        span(
            r"Joint IV $\E[f^\ast(X)\mid Y=1]-\E[f^\ast(X)\mid Y=0]$",
            f"{m(sim, 'joint_oracle')[0]:.2f} (projection {m(sim, 'joint_projection')[0]:.2f})",
        ),
        span(r"Sum of marginal / conditional / coalition IVs, oracle", sums(sim, "oracle")),
        r"\midrule",
        r"\multicolumn{7}{@{}l}{\emph{FICO HELOC}}\\",
        " & ".join(["", *HELOC_COLS]) + r"\\\midrule",
        row(r"Marginal IV, projection", m(hel, "marginal_projection")),
        row(r"Conditional IV $\mathrm{IV}_j$, projection", m(hel, "conditional_projection")),
        row(r"Coalition IV $\mathrm{IV}^E_j$, projection", m(hel, "coalition_projection")),
        r"\midrule",
        span(r"Joint IV, projection", f"{m(hel, 'joint_projection')[0]:.2f}"),
        span(r"Sum of marginal / conditional / coalition IVs", sums(hel, "projection")),
        r"\bottomrule\end{tabular}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Summarize the archived runs and write the JSON summary and the LaTeX table."""
    sim_files = sorted(RES.glob("sim_seed[0-9].npz"))
    hel_files = sorted(RES.glob("heloc_rep[0-9].npz"))
    sim = summarize([sim_seed(p) for p in sim_files])
    hel = summarize([heloc_rep(p) for p in hel_files])
    (RES / "civ_summary.json").write_text(json.dumps({"sim": sim, "heloc": hel}, indent=1) + "\n")
    for title, s in (("simulation", sim), ("HELOC", hel)):
        print(f"\n{title}")
        for k, v in s.items():
            print(f"  {k:<28}", np.round(v["mean"], 3), "sum", round(float(np.sum(v["mean"])), 3))
    if "coalition_oracle" in sim and "coalition_projection" in hel:
        (TAB / "civ.tex").write_text(table(sim, hel, len(sim_files), len(hel_files)))
        print("\nwritten tables/civ.tex")
    else:
        print("\ncoalition arrays not found; rerun run_sim.py and run_heloc.py to fill the marginal and coalition rows")


if __name__ == "__main__":
    main()
