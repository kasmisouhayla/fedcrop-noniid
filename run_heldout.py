"""Held-out validation on fresh seeds 10-19: FedAvg, SCAFFOLD, FedAvg+FT, SCAFFOLD+FT
(same hyper-parameters as the main study) and the heterogeneity indicator."""
import sys, os, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'analysis'))
import numpy as np, fl
part = sys.argv[1]; ALPHAS = ['100', '1.0', '0.5'] if part == 'A' else ['0.1', '0.05']
X, y, _ = fl.load(); out = {}; t0 = time.time()
def ft(r, farms, G, s):
    rng = np.random.default_rng(5000 + s)
    m = [fl.local_train(r['model'], f, 22, rng, 2, 0.1, 32)[0] for f in farms]; e = fl.evaluate(m, farms, G, 22)
    return dict(g_acc=e['g_acc'], l_acc=e['l_acc'], farm=e['farm'].tolist())
for a in ALPHAS:
    for s in range(10, 20):
        farms, G = fl.make_farms(X, y, 10, float(a), s)
        counts = [np.bincount(f['ytr'], minlength=22).tolist() for f in farms]
        rec = dict(counts=counts)
        for algo in ['fedavg', 'scaffold']:
            r = fl.run_fl(farms, G, 22, algo, s, rounds=200, lr=0.1, mu=0.01, eval_every=1)
            rec[algo] = dict(g_acc=r['g_acc'], l_acc=r['l_acc'], farm=r['farm'].tolist(), curve=r['curve'])
            rec[algo + '_ft'] = ft(r, farms, G, s)
        c = fl.run_central(farms, G, 22, s, lr=0.1); rec['central'] = dict(g_acc=c['g_acc'])
        out[f'{a}|{s}'] = rec
    print('alpha', a, f'{time.time()-t0:.0f}s', flush=True)
json.dump(out, open(os.path.join(ROOT, 'results', f'raw_heldout_{part}.json'), 'w'))
print('done', part, flush=True)
