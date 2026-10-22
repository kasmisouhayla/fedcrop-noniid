import sys, json, time, logging; import os; ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.path.join(ROOT,'src')); logging.disable(logging.WARNING)
import numpy as np, fl
import dp_accounting as dpa
part=sys.argv[1]
X,y,cl=fl.load(); C=22; SEEDS=range(int(os.environ.get('SEEDS', 10))); K=10
R, LR, MU = 200, 0.1, 0.01
out={'config':dict(K=K,rounds=R,lr=LR,mu=MU,frac=0.5,epochs=2,bs=32,hidden=64,seeds=list(SEEDS),tuning_seeds=[100,101])}
t0=time.time()
def summ(r): return dict(g_acc=r['g_acc'],g_f1=r['g_f1'],l_acc=r['l_acc'],farm=r['farm'].tolist())
VAR=[('fedavg','fedavg',{}),('fedprox','fedprox',{}),('scaffold','scaffold',{}),('fedavg_bn','fedavg',dict(use_bn=True)),
     ('fedbn','fedbn',{}),('fedavg_ln','fedavg',dict(use_ln=True))]
if part=='A':
    E1={}
    for a in [100,1.0,0.5,0.1,0.05]:
        for s in SEEDS:
            farms,G=fl.make_farms(X,y,K,a,s)
            E1.setdefault(f'{a}|central',[]).append(summ(fl.run_central(farms,G,C,s,lr=LR)))
            E1.setdefault(f'{a}|local',[]).append(summ(fl.run_local(farms,G,C,s,lr=LR)))
            for name,algo,kw in VAR:
                r=fl.run_fl(farms,G,C,algo,s,rounds=R,lr=LR,mu=MU,eval_every=1,**kw)
                d=summ(r); d['curve']=r['curve']; E1.setdefault(f'{a}|{name}',[]).append(d)
        print('E1 alpha',a,f'{time.time()-t0:.0f}s',flush=True)
    out['E1']=E1
else:
    E4={}
    for a in [100,0.1]:
        for s in SEEDS:
            farms,G=fl.make_farms(X,y,K,a,s,shift=0.5)
            E4.setdefault(f'{a}|central',[]).append(summ(fl.run_central(farms,G,C,s,lr=LR)))
            E4.setdefault(f'{a}|local',[]).append(summ(fl.run_local(farms,G,C,s,lr=LR)))
            for name,algo,kw in VAR:
                if name=='fedprox': continue
                r=fl.run_fl(farms,G,C,algo,s,rounds=R,lr=LR,mu=MU,eval_every=R,**kw); E4.setdefault(f'{a}|{name}',[]).append(summ(r))
        print('E4 alpha',a,f'{time.time()-t0:.0f}s',flush=True)
    out['E4']=E4
    E5={}
    for a in [0.1,0.05]:
        for s in SEEDS:
            farms,G=fl.make_farms(X,y,50,a,s,min_size=5)
            for algo in ['fedavg','scaffold']:
                r=fl.run_fl(farms,G,C,algo,s,rounds=400,frac=0.1,lr=LR,eval_every=2)
                d=summ(r); d['curve']=r['curve']; E5.setdefault(f'{a}|{algo}',[]).append(d)
        print('E5 alpha',a,f'{time.time()-t0:.0f}s',flush=True)
    out['E5']=E5
    E2={}
    for a in [1.0,0.1]:
        for E in [1,2,5]:
            for algo in ['fedavg','scaffold']:
                for s in SEEDS:
                    farms,G=fl.make_farms(X,y,K,a,s)
                    r=fl.run_fl(farms,G,C,algo,s,rounds=R,lr=LR,mu=MU,epochs=E,eval_every=1)
                    E2.setdefault(f'{a}|{E}|{algo}',[]).append(dict(curve=r['curve'],g_acc=r['g_acc']))
        print('E2 alpha',a,f'{time.time()-t0:.0f}s',flush=True)
    out['E2']=E2
    def eps(z,q,steps,delta=1e-5):
        if z==0: return float('inf')
        acc=dpa.rdp.RdpAccountant(); acc.compose(dpa.PoissonSampledDpEvent(q,dpa.GaussianDpEvent(z)),int(steps))
        return float(acc.get_epsilon(delta))
    E3={}; RD,ED,BD,CLIP=100,1,16,3.0
    out['config_dp']=dict(rounds=RD,epochs=ED,bs=BD,clip=CLIP,delta=1e-5,lr=LR)
    for a in [1.0,0.1]:
        for s in SEEDS:
            farms,G=fl.make_farms(X,y,K,a,s)
            r=fl.run_fl(farms,G,C,'fedavg',s,rounds=RD,lr=LR,epochs=ED,bs=BD,eval_every=RD)
            E3.setdefault(f'{a}|nodp',[]).append(dict(g_acc=r['g_acc'],l_acc=r['l_acc']))
            for z in [0.0,0.5,1.0,1.5,2.0,3.0,5.0]:
                r=fl.run_fl(farms,G,C,'fedavg',s,rounds=RD,lr=LR,epochs=ED,bs=BD,eval_every=RD,dp=dict(clip=CLIP,sigma=z))
                es=[eps(z,min(1,BD/n),p*ED*int(np.ceil(n/BD))) for n,p in zip(r['n'],r['part'])]
                E3.setdefault(f'{a}|{z}',[]).append(dict(g_acc=r['g_acc'],l_acc=r['l_acc'],eps_max=max(es),eps_med=float(np.median(es))))
        print('E3 alpha',a,f'{time.time()-t0:.0f}s',flush=True)
    out['E3']=E3
json.dump(out,open(os.path.join(ROOT,'results',f'raw_main_{part}.json'),'w'))
print('done',part,f'{time.time()-t0:.0f}s',flush=True)
