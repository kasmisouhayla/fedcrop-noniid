import sys, json, time, warnings, logging; import os; ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.path.join(ROOT,'src')); warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
import numpy as np, fl, fl_reg as F
import dp_accounting as dpa
part=sys.argv[1]; SEEDS=range(int(os.environ.get('SEEDS', 10))); t0=time.time(); out={}
def summ(r): return dict(g_acc=r['g_acc'],l_acc=r['l_acc'],farm=r['farm'].tolist())
if part=='A':
    Xr,yr,area,year=F.load_yield(); E7={}
    for s in SEEDS:
        farms,G=F.make_countries(Xr,yr,area,year,s)
        E7.setdefault('central',[]).append(summ(F.run_central(farms,G,1,s,epochs=10,lr=0.05)))
        E7.setdefault('local',[]).append(summ(F.run_local(farms,G,1,s,epochs=100,lr=0.05)))
        for name,algo,lr,kw in [('fedavg','fedavg',.01,{}),('fedprox','fedprox',.01,{}),('scaffold','scaffold',.01,{}),
                                ('fedavg_bn','fedavg',.1,dict(use_bn=True)),('fedbn','fedbn',.1,{}),('fedavg_ln','fedavg',.01,dict(use_ln=True))]:
            E7.setdefault(name,[]).append(summ(F.run_fl(farms,G,1,algo,s,rounds=300,frac=0.13,lr=lr,eval_every=300,**kw)))
        E7.setdefault('ft',[]).append(summ(F.run_ft(farms,G,1,s,ft_epochs=2,rounds=300,frac=0.13,lr=.01,eval_every=300)))
        print('E7 seed',s,f'{time.time()-t0:.0f}s',flush=True)
    out['E7']=E7
    out['E7_info']=dict(clients=len(farms),train=int(sum(len(f['ytr']) for f in farms)),global_test=int(len(G['y'])))
else:
    X,y,_=fl.load(); E8={}
    for a in [100,1.0,0.5,0.1,0.05]:
        for s in SEEDS:
            farms,G=fl.make_farms(X,y,10,a,s)
            E8.setdefault(f'{a}',[]).append(summ(fl.run_ft(farms,G,22,s,ft_epochs=2,rounds=200,lr=0.1,eval_every=200)))
    for a in [100,0.1]:
        for s in SEEDS:
            farms,G=fl.make_farms(X,y,10,a,s,shift=0.5)
            E8.setdefault(f'shift|{a}',[]).append(summ(fl.run_ft(farms,G,22,s,ft_epochs=2,rounds=200,lr=0.1,eval_every=200)))
    print('E8',f'{time.time()-t0:.0f}s',flush=True); out['E8']=E8
    def eps(z,q,steps,delta=1e-5):
        acc=dpa.rdp.RdpAccountant(); acc.compose(dpa.PoissonSampledDpEvent(q,dpa.GaussianDpEvent(z)),int(steps)); return float(acc.get_epsilon(delta))
    E9={}
    for a in [1.0,0.1]:
        for s in SEEDS:
            farms,G=fl.make_farms(X,y,10,a,s)
            for z in [1.0,2.0,3.0]:
                r=fl.run_fl(farms,G,22,'scaffold',s,rounds=100,lr=0.1,epochs=1,bs=16,eval_every=100,dp=dict(clip=3.0,sigma=z))
                es=[eps(z,min(1,16/n),p*int(np.ceil(n/16))) for n,p in zip(r['n'],r['part'])]
                E9.setdefault(f'{a}|{z}',[]).append(dict(g_acc=r['g_acc'],eps_max=max(es)))
    print('E9',f'{time.time()-t0:.0f}s',flush=True); out['E9']=E9
json.dump(out,open(os.path.join(ROOT,'results',f'raw_extra_{part}.json'),'w'),default=float)
print('done',part,f'{time.time()-t0:.0f}s',flush=True)
