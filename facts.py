"""Print the key numbers from results/summary.json in readable form."""
import json, numpy as np
from pathlib import Path
d = json.load(open(Path(__file__).resolve().parent / "results/summary.json"))
S, H = d["sim"], d["heloc"]
f = lambda v, n=3: np.round(np.asarray(v, float), n).tolist()
print("seeds", S["seeds"], "auc", f(S["auc"]), "auc_true", f(S["auc_true"]), "ll", f(S["logloss"]), "ll_true", f(S["logloss_true"]))
print("band", S["band_size"], "identity", S["identity_maxabs"], "pi", S["pi_true"])
print("sd pt", f(S["sd_true_pt"]), "sd bin", f(S["sd_true_bin"]), "coarsening", f(S["coarsening_rmse"]))
for k in S["rmse"]: print(f"{k:22s} rmse {f(S['rmse'][k])}  sd {f(S['rmse_sd'][k])}  corr {f(S['corr'][k],2)} sign {f(S['sign'][k],2)}")
print("material share", f(S["material_share"]))
print("selected Y", S["selected_Y"]); print("selected X", S["selected_X"])
print("ess Y sel", f(S["ess_Y_sel"],0)); print("ess Y k1", f(S["ess_Y_k1"],0)); print("ess X sel", f(S["ess_X_sel"],0))
print("leafY rmse by ess tercile (low,mid,high)", f(S["leafY_rmse_by_ess_tercile"]))
g = np.array(S["grid_Y"]); print("gridY best over lambda by kappa", f(g.min(2))); g = np.array(S["grid_X"]); print("gridX", f(g.min(2)))
print("additivity", {k: f(v) for k, v in S["additivity"].items()})
print("E rmse", f(S["E_rmse"]), "E sd", f(S["E_sd"]), "E mean", f(S["E_mean"]), "Wloo mean", f(S["Wloo_mean_decl"]), "shared", f(S["redund_mean"]))
print("selection", {k: f(v) for k, v in S["selection"].items()})
aa = S["aa"]
for k in aa: print("aa", k, f(np.mean(aa[k], 0)), "| per seed", f(aa[k]) if np.ndim(aa[k]) == 1 else "")
print("h3", json.dumps(S["h3"], default=float))
print("stability by shared tercile", f(S["stability_by_shared_tercile"]), "edges", f(S["shared_tercile_edges"]), "by margin", f(S["stability_by_margin_tercile"]))
print("subgroups", json.dumps(S["subgroups"], indent=0))
for k, v in S["libraries"].items():
    print("lib", k, "auc", f(v["auc"]), "rmse", f(np.mean(v["rmse"], 0)), "agree_xgb", f(v["agree_xgb"]), "agree_true", f(v["agree_true"]), "corrW", f(np.mean(v["corr_W_xgb"], 0)), "refs_pass", f(v["refs_passing"]))
if H:
    for k, v in H.items(): print("heloc", k, f(v) if not isinstance(v, (dict, list)) or k not in ("names",) and not isinstance(v, dict) else v)
