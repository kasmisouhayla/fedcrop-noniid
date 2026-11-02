"""Pre-training heterogeneity indicator: size-weighted Jensen-Shannon divergence
between each client's crop mix and the pooled crop mix (needs only per-client
crop counts, which can be shared through secure aggregation)."""
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np, fl, fl_reg as F
from scipy import stats

def js(p, q):
    m = (p + q) / 2
    kl = lambda a, b: np.sum(np.where(a > 0, a * np.log2(np.where(a > 0, a, 1) / b), 0))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def indicator(counts):
    counts = np.asarray(counts, float); n = counts.sum(1); pooled = counts.sum(0) / counts.sum()
    return float(np.sum(n / n.sum() * np.array([js(c / c.sum(), pooled) for c in counts])))

M = json.load(open(os.path.join(ROOT, 'results', 'results_main.json'))); E1 = M['E1']
def hit(c, t=.95):
    for r, a in c:
        if a >= t: return r
X, y, _ = fl.load(); rows = []
for a in ['100', '1.0', '0.5', '0.1', '0.05']:
    for s in range(10):
        farms, G = fl.make_farms(X, y, 10, float(a), s)
        H = indicator([np.bincount(f['ytr'], minlength=22) for f in farms])
        fa, sc, ce = E1[f'{a}|fedavg'][s], E1[f'{a}|scaffold'][s], E1[f'{a}|central'][s]
        rows.append(dict(alpha=a, seed=s, H=H, drop=100 * (ce['g_acc'] - fa['g_acc']),
                         r95_fa=hit(fa['curve']), r95_sc=hit(sc['curve']),
                         gainR=hit(fa['curve']) - hit(sc['curve']),
                         gainW=100 * (min(sc['farm']) - min(fa['farm']))))
H = np.array([r['H'] for r in rows])
print('Indicator H by alpha (mean [min, max]):')
for a in ['100', '1.0', '0.5', '0.1', '0.05']:
    h = [r['H'] for r in rows if r['alpha'] == a]; print(f'  alpha={a:>4}: {np.mean(h):.3f} [{min(h):.3f}, {max(h):.3f}]')
for k, lab in [('drop', 'FedAvg accuracy loss vs centralized (points)'), ('r95_fa', 'FedAvg rounds to 95%'),
               ('gainR', 'SCAFFOLD saving in rounds'), ('gainW', 'SCAFFOLD gain on worst farm (points)')]:
    v = np.array([r[k] for r in rows]); rho, p = stats.spearmanr(H, v)
    lo = [r for r in rows if r['alpha'] in ('0.1', '0.05')]
    rho2, p2 = stats.spearmanr([r['H'] for r in lo], [r[k] for r in lo])
    print(f'{lab:45s} all: rho={rho:.2f} (p={p:.1e}) | within alpha<=0.1: rho={rho2:.2f} (p={p2:.3f})')
# simple decision rule
for T in [0.1, 0.2, 0.3, 0.4, 0.5]:
    big = [r for r in rows if r['H'] >= T]; small = [r for r in rows if r['H'] < T]
    f = lambda L, k: np.mean([r[k] for r in L]) if L else float('nan')
    print(f'H>={T}: n={len(big):2d} SCAFFOLD saves {f(big,"gainR"):5.1f} rounds | H<{T}: n={len(small):2d} saves {f(small,"gainR"):5.1f}')
# natural partition: yield countries, crop mix of each country
Xr, yr, area, year = F.load_yield(); Hy = []
nI = 10
for s in range(10):
    farms, G = F.make_countries(Xr, yr, area, year, s)
    Hy.append(indicator([np.bincount(f['Xtr'][:, :nI].argmax(1), minlength=nI) for f in farms]))  # first nI columns = one-hot crop
print(f'Yield (78 countries, crop mix): H = {np.mean(Hy):.3f} [{min(Hy):.3f}, {max(Hy):.3f}]')
json.dump(dict(rows=rows, yield_H=Hy), open(os.path.join(ROOT, 'results', 'heterogeneity.json'), 'w'))
