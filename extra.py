"""extra.py -- supplementary runs for the reviewer responses (all numbers written to out/extra.json):
transient metrics, saturation statistics, P&O 5x5 grid, switching-level PWM with delay,
sub-step convergence, sampled-data (ZOH) T_s sweep, Euler step refinement, chatter mitigation."""
import numpy as np, pickle, json, sys
from multiprocessing import Pool
import pvsim as S, experiments as X

O = 'out/'
TH = {'E1': [1.0, 1 / 200.], 'E2': [1.0, 1 / 200.]}
pso = json.load(open(O + 'pso.json')); TH['E3'] = TH['E4'] = [pso['Rs'], 1 / pso['Rsh']]
NZ = {'E1': X.NONOISE, 'E2': X.NOISE, 'E3': X.NOISE, 'E4': X.NOISE}
KEYS = ['eff', 'vrmse', 'ci', 'eRs', 'etr', 'dvmax', 'dpmax', 'ts', 'ts_step', 'dmin', 'dmax', 'vomin', 'tsat', 'numax']


def one(a):
    sc, mode, seed, kw, rkw = a
    p = X.params(**kw); h = rkw.pop('h', X.H)
    prof = X.scen(sc) if h == X.H else None
    if prof is None:
        H0 = X.H; X.H = h; prof = X.scen(sc); X.H = H0
    stride = max(1, int(round(1e-4 / h)))
    o = S.run(mode, prof, TH[sc], p, h=h, noise_rms=NZ[sc], seed=seed, stride=stride, **rkw)
    m = X.metrics(o, steps=X.STEPS[sc])
    m['smax'] = float(np.abs(o[o[:, 0] > 0.5, 15]).max()); m['ripi'] = float(o[o[:, 0] > 0.5, 26][:2000].mean())
    m['ripv'] = float(o[o[:, 0] > 0.5, 27][:2000].mean())
    return {k: float(v) for k, v in m.items()}


def agg(rs, keys=KEYS + ['smax', 'ripi', 'ripv']):
    return {k: [float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))] for k in keys}


if __name__ == '__main__':
    E = {}
    with Pool(2) as pool:
        # ---- P&O 5x5 grid, every scenario
        grid = {}
        for T in (1e-3, 2e-3, 5e-3, 10e-3, 20e-3):
            for dd in (0.001, 0.002, 0.003, 0.005, 0.010):
                for sc in ('E1', 'E2', 'E3', 'E4'):
                    rs = pool.map(one, [(sc, 4, sd, dict(po_T=T, po_step=dd), {}) for sd in X.SEEDS[sc]])
                    grid[f'{T*1e3:g}ms|{dd:g}|{sc}'] = agg(rs, ['eff', 'vrmse'])
            print('grid', T, flush=True)
        E['po_grid'] = grid
        # ---- switching-level PWM (exact switching instants, centred PWM), transport delay Nd
        sw = {}
        for Nd in (1, 2, 3, 4):
            sw[f'E1|Nd{Nd}|n10'] = agg(pool.map(one, [('E1', 0, 1, {}, dict(sw=1, Nd=Nd, nsub=10))]))
        for ns in (5, 20, 40):
            sw[f'E1|Nd1|n{ns}'] = agg(pool.map(one, [('E1', 0, 1, {}, dict(sw=1, Nd=1, nsub=ns))]))
        for sc in ('E2', 'E4'):
            for Nd in (1, 2):
                sw[f'{sc}|Nd{Nd}|n10'] = agg(pool.map(one, [(sc, 0, sd, {}, dict(sw=1, Nd=Nd, nsub=10)) for sd in X.SEEDS[sc][:10]]))
        sw['E1|avg|Nd1'] = agg(pool.map(one, [('E1', 0, 1, {}, dict(Nd=1))]))
        sw['E1|avg|Nd2'] = agg(pool.map(one, [('E1', 0, 1, {}, dict(Nd=2))]))
        sw['E1|avg|Nd3'] = agg(pool.map(one, [('E1', 0, 1, {}, dict(Nd=3))]))
        E['switching'] = sw; print('sw', flush=True)
        # ---- sampled-data check: plant at 1 us, controller (ZOH) at T_s
        zoh = {}
        for Ts in (10, 20, 30, 40, 50, 60, 70, 80):
            zoh[f'{Ts}us'] = agg(pool.map(one, [('E1', 0, 1, {}, dict(h=Ts * 1e-6, nsub=Ts))]))
        E['zoh'] = zoh; print('zoh', flush=True)
        # ---- Euler step refinement of the averaged model (controller at 20 us)
        ref = {}
        for ns in (1, 2, 4, 20):
            for sc in ('E1', 'E2'):
                ref[f'{sc}|n{ns}'] = agg(pool.map(one, [(sc, 0, sd, {}, dict(nsub=ns)) for sd in X.SEEDS[sc][:5]]))
        E['refine'] = ref
        # ---- chatter mitigation (phi_b = 0.20 A, tau_S = 0.25 ms), E2
        E['mitig'] = agg(pool.map(one, [('E2', 0, sd, dict(phib=0.20, tauS=2.5e-4), {}) for sd in X.SEEDS['E2']]))
        # ---- reduced in-layer gain (phi_b = 1.0 A, k2 = 1000 1/s), E2
        E['mitig2'] = agg(pool.map(one, [('E2', 0, sd, dict(phib=1.0, k2=1000.), {}) for sd in X.SEEDS['E2']]))
    # ---- discrete sliding loop with delay: s_{k+1} = s_k - a s_{k-Nd}, a = T_s L_fast
    a = 2e-5 * 28000; rho = {}
    for Nd in range(0, 5):
        c = np.zeros(Nd + 2); c[0] = 1; c[1] = -1; c[-1] += a
        rho[str(Nd)] = float(np.abs(np.roots(c)).max())
    E['rho_delay'] = rho
    json.dump(E, open(O + 'extra.json', 'w'), indent=1)
