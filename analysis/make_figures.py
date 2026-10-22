import os
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(ROOT, "results", "results_main.json"))); OUT = os.path.join(ROOT, "figures") + os.sep
E1, E2, E3 = R["E1"], R["E2"], R["E3"]
AL = ["100", "1.0", "0.5", "0.1", "0.05"]; XL = ["IID", "1.0", "0.5", "0.1", "0.05"]
COL = dict(fedavg="#2a78d6", scaffold="#1baf7a", fedavg_bn="#eb6834", fedbn="#6250d6", local="#8a8984")
MK = dict(fedavg="o", scaffold="^", fedavg_bn="s", fedbn="D", local="x")
LS = dict(fedavg="-", scaffold="-.", fedavg_bn="--", fedbn=":", local="-")
LAB = dict(local="Local only", fedavg="FedAvg", fedprox="FedProx", scaffold="SCAFFOLD", fedavg_bn="FedAvg + BN",
           fedbn="FedBN", central="Centralized")
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "grid.color": "#e4e3df", "grid.linewidth": 0.5})
def hit(c, t):
    for r, a in c:
        if a >= t: return r
    return None
def stat(a, m, key):
    v = [100 * (min(r["farm"]) if key == "worst" else r[key]) for r in E1[f"{a}|{m}"]]
    return np.mean(v), np.std(v)
def r95(runs):
    h = [hit(r["curve"], 0.95) for r in runs]; hh = [x for x in h if x]
    return (np.mean(hh), len(hh)) if hh else (None, 0)
L = []
L.append("TABLE 1 — mean ± sd over 5 seeds. G = accuracy on the balanced global test set (22 crops); "
         "L = accuracy on the farms' local test sets; Worst = lowest farm's L; R95 = rounds to reach 95 % G (runs reaching it).")
for a, xl in zip(["100", "0.5", "0.1", "0.05"], ["IID", "α=0.5", "α=0.1", "α=0.05"]):
    L.append(f"\n{xl}")
    for m in ["local", "fedavg", "fedprox", "scaffold", "fedavg_bn", "fedbn", "central"]:
        g, l, w = stat(a, m, "g_acc"), stat(a, m, "l_acc"), stat(a, m, "worst")
        s = f"  {LAB[m]:<12} G {g[0]:5.1f} ± {g[1]:.1f} | L {l[0]:5.1f} ± {l[1]:.1f} | Worst {w[0]:5.1f}"
        if "curve" in E1[f"{a}|{m}"][0]:
            v, n = r95(E1[f"{a}|{m}"]); s += f" | R95 {('%.0f' % v) if v else '—'} ({n}/5)"
        L.append(s)
L.append("\nTABLE 2 — rounds to 95 % global accuracy vs local epochs E (no BN, 5 seeds)")
for a in ["1.0", "0.1"]:
    for E in ["1", "2", "5"]:
        fa, sc = r95(E2[f"{a}|{E}|fedavg"]), r95(E2[f"{a}|{E}|scaffold"])
        L.append(f"  α={a} E={E}: FedAvg {fa[0]:.0f} ({fa[1]}/5) | SCAFFOLD {sc[0]:.0f} ({sc[1]}/5) | speed-up ×{fa[0]/sc[0]:.1f}")
L.append("\nTABLE 3 — record-level DP-SGD at each farm (FedAvg, 100 rounds, E=1, B=16, clip C=3, δ=1e-5; ε of the worst farm)")
for a in ["1.0", "0.1"]:
    nd = [100 * r["g_acc"] for r in E3[f"{a}|nodp"]]
    L.append(f"  α={a} no DP: {np.mean(nd):.1f} ± {np.std(nd):.1f}")
    for z in ["0.0", "0.5", "1.0", "1.5", "2.0", "3.0", "5.0"]:
        rs = E3[f"{a}|{z}"]; g = [100 * r["g_acc"] for r in rs]
        e = np.mean([r["eps_max"] for r in rs])
        L.append(f"  α={a} z={z}: {np.mean(g):.1f} ± {np.std(g):.1f}   ε_worst {'∞ (clipping only)' if z=='0.0' else '%.1f' % e}")
open(OUT + "results_summary.txt", "w").write("\n".join(L)); print("\n".join(L))

# Fig 1 — generalisation vs personalisation
fig, axs = plt.subplots(1, 2, figsize=(4.6, 1.85), sharey=True); x = np.arange(len(AL))
for ax, key, t in [(axs[0], "g_acc", "(a) Global accuracy (22 crops)"), (axs[1], "l_acc", "(b) Local accuracy (own crops)")]:
    for m in ["fedavg", "scaffold", "fedavg_bn", "fedbn", "local"]:
        mu = [stat(a, m, key)[0] for a in AL]; sd = [stat(a, m, key)[1] for a in AL]
        ax.errorbar(x, mu, yerr=sd, color=COL[m], marker=MK[m], ls=LS[m], lw=1.2, ms=3.5, capsize=2, mec="white", mew=0.5, label=LAB[m])
    cen = np.mean([stat(a, "central", key)[0] for a in AL]); ax.axhline(cen, color="#52514e", lw=0.8, ls=(0, (4, 3)))
    ax.text(-0.1, cen + 1.2, "Centralized", fontsize=6.5, color="#52514e")
    ax.set_xticks(x, XL); ax.set_xlabel("Dirichlet α (heterogeneity →)"); ax.set_title(t, fontsize=6.5, loc="left"); ax.grid(axis="y")
axs[0].set_ylabel("Accuracy (%)"); axs[0].set_ylim(20, 103)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, columnspacing=0.8, handlelength=1.8, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.05))
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(OUT + "fig1_generalization_personalization.png", dpi=800, bbox_inches="tight")
fig.savefig(OUT + "fig1_generalization_personalization.pdf", bbox_inches="tight")

# Fig 2 — convergence at alpha=0.05 and privacy-utility
fig, axs = plt.subplots(1, 2, figsize=(4.6, 1.85)); ax = axs[0]
for m in ["fedavg", "scaffold", "fedavg_bn", "fedbn"]:
    cs = np.array([[c[1] for c in r["curve"]] for r in E1[f"0.05|{m}"]]) * 100; rr = [c[0] for c in E1[f"0.05|{m}"][0]["curve"]]
    ax.plot(rr, cs.mean(0), color=COL[m], ls=LS[m], lw=1.2, label=LAB[m])
    ax.fill_between(rr, cs.mean(0) - cs.std(0), cs.mean(0) + cs.std(0), color=COL[m], alpha=0.12, lw=0)
ax.axhline(95, color="#8a8984", lw=0.6, ls=(0, (2, 2))); ax.set_ylim(0, 100)
ax.set_xlabel("Communication round"); ax.set_ylabel("Global accuracy (%)"); ax.grid(axis="y")
ax.set_title("(a) Convergence at α=0.05", fontsize=6.5, loc="left"); ax.legend(frameon=False, fontsize=6.5, loc="lower right")
ax = axs[1]
for a, col, mk in [("1.0", "#2a78d6", "o"), ("0.1", "#eb6834", "s")]:
    zs = ["5.0", "3.0", "2.0", "1.5", "1.0", "0.5"]
    ex = [np.mean([r["eps_max"] for r in E3[f"{a}|{z}"]]) for z in zs]
    ac = [100 * np.mean([r["g_acc"] for r in E3[f"{a}|{z}"]]) for z in zs]; sd = [100 * np.std([r["g_acc"] for r in E3[f"{a}|{z}"]]) for z in zs]
    ax.errorbar(ex, ac, yerr=sd, color=col, marker=mk, lw=1.2, ms=3.5, capsize=2, mec="white", mew=0.5, label=f"α={a}")
    ax.axhline(100 * np.mean([r["g_acc"] for r in E3[f"{a}|nodp"]]), color=col, lw=0.8, ls=(0, (4, 3)))
ax.set_xscale("log"); ax.set_xlabel("ε (worst farm, δ=10⁻⁵)"); ax.set_ylabel("Global accuracy (%)"); ax.grid(axis="y")
ax.set_title("(b) Privacy–utility trade-off", fontsize=6.5, loc="left"); ax.legend(frameon=False, fontsize=6.5, title="dashed: no DP", title_fontsize=6)
fig.tight_layout(); fig.savefig(OUT + "fig2_convergence_privacy.png", dpi=800, bbox_inches="tight"); fig.savefig(OUT + "fig2_convergence_privacy.pdf", bbox_inches="tight")
print("ok")
