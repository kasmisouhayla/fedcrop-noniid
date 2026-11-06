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

print("\nWilcoxon signed-rank tests (two-sided, paired by seed), Holm-corrected")
w = lambda x, y: stats.wilcoxon(x, y).pvalue
g = lambda d, k: [r["g_acc"] for r in d[k]]
r95 = lambda d, k: [hit(r["curve"], .95) for r in d[k]]
# The Holm family is every test reported in the paper plus the two supplementary tests (yield, DP).
tests = [
    ("SCAFFOLD vs FedAvg, G, alpha=0.05", w(g(E1, "0.05|scaffold"), g(E1, "0.05|fedavg"))),
    ("SCAFFOLD vs FedAvg, R95, alpha=0.05", w(r95(E1, "0.05|scaffold"), r95(E1, "0.05|fedavg"))),
    ("SCAFFOLD vs FedAvg, G, alpha=0.1", w(g(E1, "0.1|scaffold"), g(E1, "0.1|fedavg"))),
    ("FedAvg vs FedAvg+BN, alpha=0.1", w(g(E1, "0.1|fedavg"), g(E1, "0.1|fedavg_bn"))),
    ("FedAvg+LN vs FedAvg, alpha=0.05", w(g(E1, "0.05|fedavg_ln"), g(E1, "0.05|fedavg"))),
    ("FedBN vs FedAvg, sensor shift", w(g(E4, "100|fedbn"), g(E4, "100|fedavg"))),
    ("SCAFFOLD vs FedAvg, K=50", w(g(E5, "0.05|scaffold"), g(E5, "0.05|fedavg"))),
    ("Yield: FedAvg vs FedAvg+BN", w(g(E7, "fedavg"), g(E7, "fedavg_bn"))),
    ("DP (z=2, alpha=1): SCAFFOLD vs FedAvg", w(g(E9, "1.0|2.0"), g(E3, "1.0|2.0"))),
]


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values (same order as the input)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


p_adj = holm(np.array([p for _, p in tests]))
for (name, p), pa in zip(tests, p_adj):
    print(f"  {name:40s} raw p = {p:.4f}   Holm p = {pa:.4f}   {'significant' if pa < 0.05 else 'n.s.'} at 5%")
