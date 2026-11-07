"""Sensitivity of FedProx to mu: mu in {0.001, 0.01, 0.1} against FedAvg on crop recommendation,
same protocol as the main study (K = 10, T = 200, E = 2, B = 32, lr = 0.1), on the two tuning
seeds (100, 101) and on the ten main seeds (0-9). Supports the statement in Sect. 3.4 that the
three values of mu perform alike."""
import sys, os, json, logging
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
logging.disable(logging.WARNING)
import numpy as np, fl

X, y, _ = fl.load(); C, K, R, LR = 22, 10, 200, 0.1
RUNS = [('fedavg', 'fedavg', 0.01), ('fedprox_mu0.001', 'fedprox', 0.001),
        ('fedprox_mu0.01', 'fedprox', 0.01), ('fedprox_mu0.1', 'fedprox', 0.1)]
out = {}
for tag, seeds in [('tuning', [100, 101]), ('main', list(range(int(os.environ.get('SEEDS', 10)))))]:
    print(f'\n=== {tag} seeds {seeds[0]}-{seeds[-1]} ===')
    for a in [100, 1.0, 0.5, 0.1, 0.05]:
        res = {}
        for s in seeds:
            farms, G = fl.make_farms(X, y, K, a, s)
            for name, algo, mu in RUNS:
                r = fl.run_fl(farms, G, C, algo, s, rounds=R, lr=LR, mu=mu, eval_every=R)
                res.setdefault(name, []).append(dict(seed=s, g_acc=r['g_acc'], l_acc=r['l_acc']))
        out[f'{tag}|{a}'] = res
        g = {k: 100 * np.mean([d['g_acc'] for d in v]) for k, v in res.items()}
        spread = max(g[k] for k in g if k != 'fedavg') - min(g[k] for k in g if k != 'fedavg')
        print(f'alpha={a:>5}: ' + '  '.join(f'{k} {v:5.1f}' for k, v in g.items()) + f'  | spread over mu: {spread:.2f} pts')
json.dump(out, open(os.path.join(ROOT, 'results', 'raw_mu.json'), 'w'))
