"""Combined recipe: federated training (FedAvg or SCAFFOLD, optional layer norm)
followed by short local fine-tuning. Same seeds, splits and hyper-parameters as
the FedAvg + FT baseline (no extra tuning)."""
import sys, os, json, time, logging, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
import numpy as np, fl, fl_reg as F
part = sys.argv[1]; SEEDS = range(int(os.environ.get('SEEDS', 10))); t0 = time.time(); out = {}

def run_ftx(M, farms, G, C, seed, algo, use_ln, ft_epochs=2, bs=32, **kw):
    r = M.run_fl(farms, G, C, algo, seed, bs=bs, use_ln=use_ln, **kw)
    rng = np.random.default_rng(5000 + seed)
    models = [M.local_train(r['model'], f, C, rng, ft_epochs, kw['lr'], bs)[0] for f in farms]
    ev = M.evaluate(models, farms, G, C)
    return dict(g_acc=ev['g_acc'], l_acc=ev['l_acc'], farm=np.asarray(ev['farm']).tolist(),
                g_glob=r['g_acc'], l_glob=r['l_acc'])

VAR = [('scaffold_ft', 'scaffold', False), ('scaffold_ln_ft', 'scaffold', True), ('fedavg_ln_ft', 'fedavg', True)]
if part == 'crop':
    X, y, _ = fl.load(); R = {}
    for tag, a, sh in [(str(a), a, 0.0) for a in [100, 1.0, 0.5, 0.1, 0.05]] + [('shift|100', 100, 0.5), ('shift|0.1', 0.1, 0.5)]:
        for s in SEEDS:
            farms, G = fl.make_farms(X, y, 10, a, s, shift=sh)
            for name, algo, ln in VAR:
                R.setdefault(f'{tag}|{name}', []).append(run_ftx(fl, farms, G, 22, s, algo, ln, rounds=200, lr=0.1, eval_every=200))
        print('crop', tag, f'{time.time()-t0:.0f}s', flush=True)
    out['crop'] = R
else:
    Xr, yr, area, year = F.load_yield(); R = {}
    for s in SEEDS:
        farms, G = F.make_countries(Xr, yr, area, year, s)
        for name, algo, ln in VAR:
            R.setdefault(name, []).append(run_ftx(F, farms, G, 1, s, algo, ln, rounds=300, frac=0.13, lr=.01, eval_every=300))
        print('yield seed', s, f'{time.time()-t0:.0f}s', flush=True)
    out['yield'] = R
json.dump(out, open(os.path.join(ROOT, 'results', f'raw_recipe_{part}.json'), 'w'))
print('done', part, f'{time.time()-t0:.0f}s', flush=True)
