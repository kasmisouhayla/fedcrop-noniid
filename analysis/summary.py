"""Print the tables and statistical tests reported in the paper from the result files."""
import json, os
import numpy as np
from scipy import stats
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L = lambda f: json.load(open(os.path.join(ROOT, "results", f)))
M, Y, X = L("results_main.json"), L("results_yield.json"), L("results_extra.json")
E1, E3, E4, E5, E7, E8, E9 = M["E1"], M["E3"], M["E4"], M["E5"], Y["E7"], X["E8"], X["E9"]

def hit(c, t):
    for r, a in c:
        if a >= t: return r
    return None

print("Table 1 - crop recommendation (global accuracy %, mean ± sd over seeds)")
for m in ["local", "fedavg", "fedprox", "scaffold", "fedavg_bn", "fedavg_ln", "fedbn", "central"]:
    row = [f"{100*np.mean([r['g_acc'] for r in E1[f'{a}|{m}']]):5.1f}" for a in ["100", "0.5", "0.1", "0.05"]]
    rs = E1[f"0.05|{m}"]
    l = 100 * np.mean([r["l_acc"] for r in rs]); w = 100 * np.mean([min(r["farm"]) for r in rs])
    h = [hit(r["curve"], .95) for r in rs] if "curve" in rs[0] else []
    r95 = f"{np.mean(h):.0f}" if h and all(h) else "-"
    print(f"  {m:<10}", *row, f"| L {l:5.1f} worst {w:5.1f} R95 {r95}")
for a in ["100", "0.5", "0.1", "0.05"]:
    print(f"  ft (a={a}) G {100*np.mean([r['g_acc'] for r in E8[a]]):.1f} L {100*np.mean([r['l_acc'] for r in E8[a]]):.1f}")

print("\nTable 2 - yield prediction (R2 on unseen countries G / future years L)")
for m in ["local", "fedavg", "scaffold", "fedavg_bn", "fedbn", "ft", "central"]:
    print(f"  {m:<10} G {np.mean([r['g_acc'] for r in E7[m]]):6.2f}  L {np.mean([r['l_acc'] for r in E7[m]]):5.2f}")

print("\nWilcoxon signed-rank tests (paired by seed)")
w = lambda x, y: stats.wilcoxon(x, y).pvalue
g = lambda d, k: [r["g_acc"] for r in d[k]]
print("  SCAFFOLD vs FedAvg, alpha=0.05:", w(g(E1, "0.05|scaffold"), g(E1, "0.05|fedavg")))
print("  FedAvg vs FedAvg+BN, alpha=0.1:", w(g(E1, "0.1|fedavg"), g(E1, "0.1|fedavg_bn")))
print("  FedBN vs FedAvg, sensor shift:", w(g(E4, "100|fedbn"), g(E4, "100|fedavg")))
print("  SCAFFOLD vs FedAvg, K=50:", w(g(E5, "0.05|scaffold"), g(E5, "0.05|fedavg")))
print("  Yield: FedAvg vs FedAvg+BN:", w(g(E7, "fedavg"), g(E7, "fedavg_bn")))
print("  DP (z=2, alpha=1): SCAFFOLD vs FedAvg:", w(g(E9, "1.0|2.0"), g(E3, "1.0|2.0")))
