"""Admissible initial conditions: grid of x(0) at STC with the poor initial estimate (E1 start-up)."""
import numpy as np, json, pickle, pvsim as S, experiments as X
p = X.params(); prof = S.profile(0.6, X.H)
res = []
for v0 in (20., 24., 28., 32., 35.):
    for i0 in (0., 4., 8., 12.):
        for vo0 in (40., 60., 80., 110.):
            o = S.run(0, prof, [1.0, 1 / 200.], p, x0=(v0, i0, vo0), stride=1)
            t = o[:, 0]; k = t > 0.5
            res.append(dict(v0=v0, i0=i0, vo0=vo0, vomin=float(o[:, 3].min()), tsat=float(1e3 * o[:, 24].sum() * X.H),
                            tlast_sat=float(1e3 * t[o[:, 24] > 0][-1]) if (o[:, 24] > 0).any() else 0.0,
                            eff_end=float(100 * o[k, 6].sum() / o[k, 7].sum()), e1_end=float(np.abs(o[k, 14]).max()),
                            vmin=float(o[:, 1].min()), vmax=float(o[:, 1].max())))
ok = [r for r in res if r['eff_end'] > 99.9]
S_ = dict(n=len(res), n_ok=len(ok), vomin=min(r['vomin'] for r in res), tsat_max=max(r['tsat'] for r in res),
          tlast_sat=max(r['tlast_sat'] for r in res), n_sat=sum(r['tsat'] > 0 for r in res),
          eff_end_min=min(r['eff_end'] for r in res), vmin=min(r['vmin'] for r in res), vmax=max(r['vmax'] for r in res))
json.dump(dict(summary=S_, runs=res), open('out/ic_grid.json', 'w'), indent=1); print(S_)
worst = sorted(res, key=lambda r: r['vomin'])[:3]; print(worst)
# trajectory-wise matched-disturbance estimate in E2/E4: d_u + |fv_hat/C1 + k1| |nu|
for sc in ('E1', 'E2', 'E3', 'E4'):
    L = pickle.load(open(f'out/{sc}.pkl', 'rb')); mx = 0; rmx = 0; frac = []
    for o in L[0]:
        t = o[:, 0]; m = t > 0.05; dl = np.zeros(t.size); rr = np.zeros(t.size)
        for j in np.where(m)[0][::3]:
            Iph, I0, a = S.pv_consts(o[j, 18], o[j, 19])
            _, fv, _, _ = S.sdm_all(o[j, 23], o[j, 10], o[j, 11], Iph, I0, a)
            rr[j] = abs(fv / S.C1 + 800.); dl[j] = 0.1 * o[j, 2] / S.Lind + rr[j] * abs(o[j, 22] - o[j, 17])
        mx = max(mx, dl.max()); rmx = max(rmx, rr.max()); frac.append(np.mean(dl[m][::3] > 1200 - 219))
    print(sc, 'max rho along traj', round(rmx), 'max (d_u + rho|nu|)', round(mx), 'frac time > eta-219', np.mean(frac))
