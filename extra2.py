import json, numpy as np
from multiprocessing import Pool
import extra as Ex, experiments as X
if __name__=='__main__':
    out={}
    with Pool(2) as pool:
        for eta in (1600.,):
            for sc in ('E2','E4'):
                rs=pool.map(Ex.one,[(sc,0,sd,dict(eta=eta),{}) for sd in X.SEEDS[sc][:10]])
                out[f'{sc}|eta{int(eta)}']=Ex.agg(rs)
    E=json.load(open('out/extra.json')); E['eta1600']=out; json.dump(E,open('out/extra.json','w'),indent=1)
    for k,v in out.items(): print(k,{q:[round(x,3) for x in v[q]] for q in ('eff','vrmse','ci','eRs','numax','smax','etr','dvmax')})
