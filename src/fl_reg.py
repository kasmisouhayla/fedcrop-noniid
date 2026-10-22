# -*- coding: utf-8 -*-
"""
Federated learning on the Crop Recommendation dataset (pure numpy).

Model : MLP 7 -> H -> 22 with an optional batch-normalisation layer after the
        hidden linear layer (needed by FedBN).
Algos : Local-only, Centralised, FedAvg, FedProx, SCAFFOLD, FedBN,
        and DP-FedAvg with record-level DP-SGD at each farm (RDP accountant).
Split : Dirichlet label partition over K farms (Li et al., ICDE 2022);
        each farm keeps 25 % of its records as local test data, so the union of
        local test sets is the global test set.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(ROOT, "data", "Crop_recommendation.csv")


# ------------------------------------------------------------------ data
def load_yield(path=os.path.join(ROOT, "data", "yield_df.csv"), min_records=40):
    d = pd.read_csv(path, index_col=0)
    d = d.groupby(["Area", "Item", "Year"], as_index=False).mean(numeric_only=True)
    sizes = d.groupby("Area").size(); d = d[d.Area.isin(sizes[sizes >= min_records].index)].reset_index(drop=True)
    items = sorted(d.Item.unique())
    X = np.column_stack([(d.Item.to_numpy(dtype=object)[:, None] == np.array(items, dtype=object)[None, :]).astype(float),
                         d.Year.values, np.log1p(d["average_rain_fall_mm_per_year"].values),
                         np.log1p(d["pesticides_tonnes"].values), d["avg_temp"].values])
    y = np.log(d["hg/ha_yield"].values)
    return X, y, d.Area.to_numpy(dtype=object), d.Year.to_numpy()


def make_countries(X, y, area, year, seed, holdout_frac=0.2, test_from=2010):
    """Clients = countries. 20% of countries held out entirely (global test: unseen countries);
    remaining countries train on years < test_from and test locally on years >= test_from."""
    rng = np.random.default_rng(seed)
    countries = np.array(sorted(set(area)))
    held = set(rng.choice(countries, int(round(holdout_frac * len(countries))), replace=False))
    tr_mask = ~np.isin(area, list(held)) & (year < test_from)
    mu, sd = X[tr_mask].mean(0), X[tr_mask].std(0) + 1e-8
    ym, ys = y[tr_mask].mean(), y[tr_mask].std()
    Xs, yz = (X - mu) / sd, (y - ym) / ys
    farms = []
    for c in countries:
        if c in held: continue
        tr = np.where((area == c) & (year < test_from))[0]; te = np.where((area == c) & (year >= test_from))[0]
        if len(tr) < 10 or len(te) < 2: continue
        farms.append(dict(name=c, Xtr=Xs[tr], ytr=yz[tr], Xte=Xs[te], yte=yz[te]))
    gm = np.isin(area, list(held))
    return farms, dict(X=Xs[gm], y=yz[gm])


def load():
    d = pd.read_csv(DATA)
    X = d.drop(columns="label").values.astype(np.float64)
    classes = sorted(d.label.unique())
    y = d.label.map({c: i for i, c in enumerate(classes)}).values
    return X, y, classes


def dirichlet_split(y, K, alpha, rng, min_size=20):
    C = y.max() + 1
    while True:
        idx = [[] for _ in range(K)]
        for c in range(C):
            ids = rng.permutation(np.where(y == c)[0])
            p = rng.dirichlet(alpha * np.ones(K))
            cuts = (np.cumsum(p) * len(ids)).astype(int)[:-1]
            for k, part in enumerate(np.split(ids, cuts)):
                idx[k] += part.tolist()
        if min(len(i) for i in idx) >= min_size:
            return [np.array(i) for i in idx]


def make_farms(X, y, K, alpha, seed, global_test=0.25, local_test=0.2, min_size=20, shift=0.0):
    """Stratified global test set (all 22 crops, balanced), Dirichlet partition of
    the remaining records over K farms, and a local test split inside each farm."""
    rng = np.random.default_rng(seed)
    gte = np.concatenate([rng.permutation(np.where(y == c)[0])[:int(round(global_test * (y == c).sum()))]
                          for c in range(y.max() + 1)])
    rest = np.setdiff1d(np.arange(len(y)), gte)
    parts = dirichlet_split(y[rest], K, alpha, rng, min_size=min_size)
    farms = []
    for ids in parts:
        ids = rng.permutation(rest[ids])
        nt = max(1, int(round(local_test * len(ids))))
        farms.append(dict(te=ids[:nt], tr=ids[nt:]))
    tr_all = np.concatenate([f["tr"] for f in farms])
    # standardisation with global statistics (a non-sensitive summary that can be
    # obtained by secure aggregation of per-farm sums and sums of squares)
    mu, sd = X[tr_all].mean(0), X[tr_all].std(0) + 1e-8
    Xs = (X - mu) / sd
    for f in farms:
        f["Xtr"], f["ytr"] = Xs[f["tr"]], y[f["tr"]]
        f["Xte"], f["yte"] = Xs[f["te"]], y[f["te"]]
        if shift > 0:   # per-farm sensor calibration error: gain and offset per attribute
            d = X.shape[1]
            f["A"] = rng.uniform(1 - shift, 1 + shift, d); f["B"] = rng.normal(0, shift, d)
            f["Xtr"] = f["Xtr"] * f["A"] + f["B"]; f["Xte"] = f["Xte"] * f["A"] + f["B"]
    G = dict(X=Xs[gte], y=y[gte])
    return farms, G


# ------------------------------------------------------------------ model
SHARED = ["W1", "b1", "W2", "b2"]
BNKEYS = ["g", "be"]          # trainable BN params
BNSTAT = ["rm", "rv"]         # running statistics


def init(d, H, C, rng, bn=True, ln=False):
    p = dict(W1=rng.normal(0, np.sqrt(2 / d), (d, H)), b1=np.zeros(H),
             W2=rng.normal(0, np.sqrt(1 / H), (H, C)), b2=np.zeros(C))
    if bn:
        p.update(g=np.ones(H), be=np.zeros(H), rm=np.zeros(H), rv=np.ones(H))
    if ln:
        p.update(lg=np.ones(H), lb=np.zeros(H))
    return p


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def forward(p, X, train):
    h = X @ p["W1"] + p["b1"]
    cache = {"X": X}
    if "g" in p:
        if train and len(X) > 1:
            m, v = h.mean(0), h.var(0)
            p["rm"] = 0.9 * p["rm"] + 0.1 * m
            p["rv"] = 0.9 * p["rv"] + 0.1 * v
        else:
            m, v = p["rm"], p["rv"]
        inv = 1 / np.sqrt(v + 1e-5)
        hn = (h - m) * inv
        cache.update(hn=hn, inv=inv)
        h = p["g"] * hn + p["be"]
    if "lg" in p:
        lm, lv = h.mean(1, keepdims=True), h.var(1, keepdims=True)
        linv = 1 / np.sqrt(lv + 1e-5)
        lhn = (h - lm) * linv
        cache.update(lhn=lhn, linv=linv)
        h = p["lg"] * lhn + p["lb"]
    a = np.maximum(h, 0)
    cache.update(pre=h, a=a)
    return a @ p["W2"] + p["b2"], cache


def grads(p, X, y, C):
    """Mean cross-entropy gradient (batch-norm aware)."""
    z, c = forward(p, X, True)
    n = len(X)
    dz = (z - y.reshape(-1, 1)) / n
    g = dict(W2=c["a"].T @ dz, b2=dz.sum(0))
    dh = (dz @ p["W2"].T) * (c["pre"] > 0)
    if "lg" in p:
        g["lg"] = (dh * c["lhn"]).sum(0)
        g["lb"] = dh.sum(0)
        dhn = dh * p["lg"]; Hh = dh.shape[1]
        dh = c["linv"] / Hh * (Hh * dhn - dhn.sum(1, keepdims=True) - c["lhn"] * (dhn * c["lhn"]).sum(1, keepdims=True))
    if "g" in p:
        g["g"] = (dh * c["hn"]).sum(0)
        g["be"] = dh.sum(0)
        dhn = dh * p["g"]
        dh = c["inv"] / n * (n * dhn - dhn.sum(0) - c["hn"] * (dhn * c["hn"]).sum(0))
    g["W1"] = X.T @ dh
    g["b1"] = dh.sum(0)
    return g


def dp_grads(p, X, y, clip, sigma, rng):
    """Per-example clipped + Gaussian-noised gradient sum (no BN), DP-SGD."""
    z, c = forward(p, X, True)
    n = len(X)
    dz = softmax(z)
    dz[np.arange(n), y] -= 1
    a = c["a"]
    dh = (dz @ p["W2"].T) * (c["pre"] > 0)
    # squared per-example norms of each layer's gradient (outer products)
    sq = ((a ** 2).sum(1) * (dz ** 2).sum(1) + (dz ** 2).sum(1)
          + (X ** 2).sum(1) * (dh ** 2).sum(1) + (dh ** 2).sum(1))
    f = np.minimum(1, clip / (np.sqrt(sq) + 1e-12))
    dzf, dhf = dz * f[:, None], dh * f[:, None]
    g = dict(W2=a.T @ dzf, b2=dzf.sum(0), W1=X.T @ dhf, b1=dhf.sum(0))
    for k in g:
        g[k] = g[k] + rng.normal(0, sigma * clip, g[k].shape)
    return g


def predict(p, X):
    return forward(p, X, False)[0][:, 0]


# ------------------------------------------------------------------ local training
def local_train(p, f, C, rng, epochs, lr, bs, mode="avg", gp=None, mu=0.0,
                c_glob=None, c_loc=None, dp=None):
    p = {k: v.copy() for k, v in p.items()}
    X, y = f["Xtr"], f["ytr"]
    n = len(X)
    steps = 0
    for _ in range(epochs):
        if dp is None:
            order = rng.permutation(n)
            batches = [order[i:i + bs] for i in range(0, n, bs)]
        else:  # Poisson sampling, rate q = bs / n, ceil(n / bs) steps per epoch
            q = min(1.0, bs / n)
            batches = [np.where(rng.random(n) < q)[0] for _ in range(int(np.ceil(n / bs)))]
        for b in batches:
            if dp is None:
                if len(b) < 2:
                    continue
                g = grads(p, X[b], y[b], C)
            else:
                g = dp_grads(p, X[b], y[b], dp["clip"], dp["sigma"], rng)
                g = {k: v / bs for k, v in g.items()}       # expected batch size
            for k in g:
                if mode == "prox":
                    g[k] = g[k] + mu * (p[k] - gp[k])
                if mode == "scaffold":
                    g[k] = g[k] - c_loc[k] + c_glob[k]
                p[k] -= lr * g[k]
            steps += 1
    return p, steps


def weighted_avg(models, w, keys):
    w = np.asarray(w, float) / np.sum(w)
    return {k: sum(wi * m[k] for wi, m in zip(w, models)) for k in keys}


# ------------------------------------------------------------------ evaluation
def macro_f1(yt, yp, C):
    f1 = []
    for c in range(C):
        tp = ((yp == c) & (yt == c)).sum()
        fp = ((yp == c) & (yt != c)).sum()
        fn = ((yp != c) & (yt == c)).sum()
        if tp + fp + fn > 0:
            f1.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1))


def r2(yt, yp):
    return 1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2)


def evaluate(models, farms, G, C):
    """g_acc = R2 on unseen countries (global generalization), averaged over deployed models;
    l_acc = pooled R2 on each country's future years with its own deployed model;
    farm  = per-country RMSE on its future years."""
    g, yt, yp, farm = [], [], [], []
    for p, f in zip(models, farms):
        g.append(r2(G["y"], predict(p, G["X"])))
        pl = predict(p, f["Xte"])
        yt.append(f["yte"]); yp.append(pl); farm.append(float(np.sqrt(np.mean((pl - f["yte"]) ** 2))))
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return dict(g_acc=float(np.mean(g)), g_f1=0.0, l_acc=float(r2(yt, yp)), farm=np.array(farm))


# ------------------------------------------------------------------ runners
def run_fl(farms, G, C, algo, seed, rounds=100, frac=0.5, epochs=2, lr=0.05, bs=32,
           H=64, mu=0.01, dp=None, eval_every=1, target=None, use_bn=False, use_ln=False):
    rng = np.random.default_rng(1000 + seed)
    K, d = len(farms), farms[0]["Xtr"].shape[1]
    bn = algo == "fedbn" or (use_bn and dp is None)
    ln = use_ln and dp is None
    gp = init(d, H, C, rng, bn=bn, ln=ln)
    lnk = ["lg", "lb"] if ln else []
    train_keys = SHARED + (BNKEYS if bn else []) + lnk
    avg_keys = SHARED + lnk + ([] if algo == "fedbn" else (BNKEYS + BNSTAT if bn else []))
    local_bn = [{k: gp[k].copy() for k in BNKEYS + BNSTAT} for _ in range(K)] if algo == "fedbn" else None
    zeros = lambda: {k: np.zeros_like(gp[k]) for k in train_keys}
    c_glob, c_loc = (zeros(), [zeros() for _ in range(K)]) if algo == "scaffold" else (None, None)
    sizes = np.array([len(f["Xtr"]) for f in farms])
    m = max(1, int(round(frac * K)))
    curve, hit = [], None
    part = np.zeros(K, int)

    def deployed():
        if algo == "fedbn":
            return [{**gp, **local_bn[k]} for k in range(K)]
        return [gp] * K

    for r in range(1, rounds + 1):
        S = rng.choice(K, m, replace=False)
        part[S] += 1
        outs, dcs = [], []
        for k in S:
            start = {**gp, **local_bn[k]} if algo == "fedbn" else gp
            mode = {"fedprox": "prox", "scaffold": "scaffold"}.get(algo, "avg")
            p, steps = local_train(start, farms[k], C, rng, epochs, lr, bs, mode=mode,
                                   gp=gp, mu=mu, c_glob=c_glob,
                                   c_loc=c_loc[k] if c_loc else None, dp=dp)
            if algo == "fedbn":
                local_bn[k] = {kk: p[kk] for kk in BNKEYS + BNSTAT}
            if algo == "scaffold":
                new_c = {kk: c_loc[k][kk] - c_glob[kk] + (gp[kk] - p[kk]) / (steps * lr)
                         for kk in train_keys}
                dcs.append({kk: new_c[kk] - c_loc[k][kk] for kk in train_keys})
                c_loc[k] = new_c
            outs.append(p)
        new = weighted_avg(outs, sizes[S], avg_keys)
        gp = {**gp, **new}
        if algo == "scaffold":
            for kk in train_keys:
                c_glob[kk] = c_glob[kk] + sum(dc[kk] for dc in dcs) / K
        if r % eval_every == 0 or r == rounds:
            ev = evaluate(deployed(), farms, G, C)
            curve.append((r, ev["g_acc"]))
            if target is not None and hit is None and ev["g_acc"] >= target:
                hit = r
    ev = evaluate(deployed(), farms, G, C)
    ev["curve"] = curve
    ev["hit"] = hit
    ev["part"] = part
    ev["n"] = sizes
    ev["model"] = gp
    ev["local_bn"] = local_bn
    return ev


def run_ft(farms, G, C, seed, ft_epochs=5, ft_lr=None, bs=32, **kw):
    """FedAvg followed by local fine-tuning: a standard personalized-FL baseline."""
    r = run_fl(farms, G, C, "fedavg", seed, bs=bs, **kw)
    rng = np.random.default_rng(5000 + seed)
    lr = ft_lr if ft_lr is not None else kw.get("lr", 0.05)
    models = [local_train(r["model"], f, C, rng, ft_epochs, lr, bs)[0] for f in farms]
    ev = evaluate(models, farms, G, C)
    ev["global"] = r
    return ev


def run_central(farms, G, C, seed, epochs=200, lr=0.05, bs=32, H=64, use_bn=False):
    rng = np.random.default_rng(1000 + seed)
    pool = dict(Xtr=np.vstack([f["Xtr"] for f in farms]),
                ytr=np.concatenate([f["ytr"] for f in farms]))
    p = init(pool["Xtr"].shape[1], H, C, rng, bn=use_bn)
    p, _ = local_train(p, pool, C, rng, epochs, lr, bs)
    return evaluate([p] * len(farms), farms, G, C)


def run_local(farms, G, C, seed, epochs=200, lr=0.05, bs=32, H=64, use_bn=False):
    rng = np.random.default_rng(1000 + seed)
    models = []
    for f in farms:
        p = init(f["Xtr"].shape[1], H, C, rng, bn=use_bn)
        p, _ = local_train(p, f, C, rng, epochs, lr, bs)
        models.append(p)
    return evaluate(models, farms, G, C)
