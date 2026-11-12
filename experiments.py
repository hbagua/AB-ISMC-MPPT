import numpy as np, pickle, os, json, sys
import pvsim as S

OUT = 'out'; os.makedirs(OUT, exist_ok=True)
H = 2e-5
NOISE = (0.10, 0.10, 0.05, 0.20, 2.0, 0.05)   # filtered rms of (v, iL, i_pv, vo, G, T) sensors
NONOISE = (0, 0, 0, 0, 0, 0)
STEPS = {'E1': [0.0, 1.0, 2.0], 'E2': [0.0, 1.0, 2.0], 'E3': [0.0, 3.0, 6.0],
         'E4': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]}
SEEDS = {'E1': [1], 'E2': list(range(1, 21)), 'E3': list(range(1, 11)), 'E4': list(range(1, 11))}
PI_GAINS = dict(pi_Kp=0.02, pi_Ki=5.0)      # retuned: the earlier (0.08, 120) pair is unstable on this plant
PHI_R2 = 2.07 ** 2; PHI_G2 = 28.06 ** 2    # regressor energies at the STC MPP (Sec. 3.3)
RHO = 0.003                                # relative leakage
NAMES = {0: 'AB-ISMC (proposed)', 5: 'AB-ISMC + dither', 1: 'Fixed-parameter B-ISMC',
         2: 'Oracle B-ISMC ($\\theta$ known)', 3: 'Conventional SMC', 4: 'P\\&O',
         6: 'Cascaded P\\&O + PI'}


def params(**kw):
    base = dict(gR=3.0, gG=1e-3, sigl_R=RHO * PHI_R2, sigl_G=RHO * PHI_G2, tauG=1e-3,
                Rsmax=1.2, gmax=0.01, eta=1200., ad=0.0, wr=500., tauT=1e-3,
                lam_c=1000., kc_s=4000., **PI_GAINS)
    base.update(kw)
    return S.default_params(**base)


def ramp(t, t0, t1, a, b):
    return np.where(t < t0, a, np.where(t > t1, b, a + (b - a) * (t - t0) / (t1 - t0)))


def pso_offline(seed=3, n=30, iters=100):
    rng = np.random.default_rng(seed)
    Iph, I0, a = S.pv_consts(1000, 25)
    V = np.linspace(0, 36.0, 41)
    It = np.array([S.sdm_I(v, 0.35, 1 / 320, Iph, I0, a) for v in V])
    Im = It + rng.normal(size=V.size) * 0.005 * S.Isc_ref
    lb = np.array([0.05, 1 / 1000.]); ub = np.array([1.5, 1 / 60.])
    def cost(th):
        return sum((Im[j] - S.sdm_I(V[j], th[0], th[1], Iph, I0, a)) ** 2 for j in range(V.size))
    X = lb + rng.random((n, 2)) * (ub - lb); Vv = np.zeros((n, 2))
    pb = X.copy(); pc = np.array([cost(x) for x in X]); gb = pb[pc.argmin()].copy(); gc = pc.min()
    for it in range(iters):
        r1, r2 = rng.random((n, 2)), rng.random((n, 2))
        Vv = 0.7 * Vv + 1.5 * r1 * (pb - X) + 1.5 * r2 * (gb - X)
        X = np.clip(X + Vv, lb, ub)
        c = np.array([cost(x) for x in X]); m = c < pc
        pb[m] = X[m]; pc[m] = c[m]
        if pc.min() < gc: gc = pc.min(); gb = pb[pc.argmin()].copy()
    return gb, gc


def chatter_index(o, t0=0.2, k=20):
    """CI = 100 x rms of (d - its 2 ms centred moving average), t > t0.
    The moving average is taken only where the full window exists (no zero padding):
    the first and last k samples of the record are excluded."""
    t = o[:, 0]; d = o[:, 4]
    ma = np.convolve(d, np.ones(k) / k, mode='same')
    m = t > t0; m[:k] = False; m[-k:] = False
    return 100 * np.sqrt(np.mean((d - ma)[m] ** 2))


def settle_all(o, steps, tol=0.01, win=0.45):
    """worst 1 % power-band settling time (s) over all steps (last instant outside the band)."""
    t = o[:, 0]; ts = 0.0
    for t0 in steps:
        m = (t >= t0) & (t <= t0 + win)
        err = np.abs(o[m, 6] - o[m, 7]) / o[m, 7]
        bad = np.where(err > tol)[0]
        if bad.size: ts = max(ts, t[m][bad[-1]] - t0)
    return ts


def metrics(o, t0=0.2, tid=0.5, steps=(0.0,)):
    t = o[:, 0]; P = o[:, 6]; Ps = o[:, 7]; v = o[:, 1]; Vs = o[:, 8]
    m = t > t0; mi = t > tid
    eff = 100 * np.trapezoid(P, t) / np.trapezoid(Ps, t)
    vr = 100 * np.sqrt(np.mean((v[m] - Vs[m]) ** 2)) / np.mean(Vs[m])
    ci = chatter_index(o, t0)
    Rs, g, thR, thg = o[:, 12], o[:, 13], o[:, 10], o[:, 11]
    eRs = 100 * np.sqrt(np.mean(((thR - Rs) / Rs)[mi] ** 2))
    eRh = 100 * np.sqrt(np.mean(((1 / thg - 1 / g) / (1 / g))[mi] ** 2))
    eref = np.sqrt(np.mean((o[:, 9] - Vs)[mi] ** 2))
    # transient windows: 50 ms after t = 0 and after every step
    tw = np.zeros(t.size, bool)
    for s0 in steps: tw |= (t >= s0) & (t < s0 + 0.05)
    etr = 100 * np.sum(P[tw]) / np.sum(Ps[tw])
    # step windows only (start-up excluded)
    sw_ = np.zeros(t.size, bool)
    for s0 in steps[1:]: sw_ |= (t >= s0) & (t < s0 + 0.05)
    if not sw_.any(): sw_ = tw
    dvmax = np.abs(v - Vs)[sw_].max(); dpmax = (Ps - P)[sw_].max()
    return dict(eff=eff, vrmse=vr, ci=ci, eRs=eRs, eRsh=eRh, eref=eref, etr=etr, dvmax=dvmax,
                dpmax=dpmax, ts=1e3 * settle_all(o, steps), ts_step=1e3 * settle_all(o, steps[1:]) if len(steps) > 1 else 0.0,
                dmin=o[:, 4].min(), dmax=o[:, 4].max(), vomin=o[:, 3].min(),
                tsat=1e3 * o[:, 24].sum() * (t[1] - t[0]), vmin=v[t > 0.05].min(), vmax=v[t > 0.05].max(),
                vrmin=o[:, 9].min(), numax=np.abs(o[:, 22] - o[:, 17])[t > 0.01].max())


def settle(o, t_step, t_end, tol=0.02):
    t = o[:, 0]; m = (t >= t_step) & (t <= t_end)
    err = np.abs(o[m, 6] - o[m, 7]) / o[m, 7]
    bad = np.where(err > tol)[0]
    return (t[m][bad[-1]] - t_step) if bad.size else 0.0


def run_mode(mode, prof, th0, p, noise, seed, **kw):
    if mode == 5:
        pp = (dict(p[0], ad=0.5), p[1].copy()); pp[1][6] = 0.5
        return S.run(0, prof, th0, pp, h=H, noise_rms=noise, seed=seed, **kw)
    return S.run(mode, prof, th0, p, h=H, noise_rms=noise, seed=seed, **kw)


def scen(name):
    if name == 'E1' or name == 'E2':
        T = 4.0
        G = lambda t: ramp(t, 1.0, 1.02, 1000, 550) + ramp(t, 2.0, 2.02, 0, 350)
        Tc = lambda t: 25 + 15 * t / T
        return S.profile(T, H, G=G, Tc=Tc)
    if name == 'E3':
        T = 8.0
        G = lambda t: ramp(t, 3.0, 3.02, 1000, 800) + ramp(t, 6.0, 6.02, 0, 200)
        Tc = lambda t: 25 + 20 * t / T
        Rs = lambda t: 0.35 + 0.45 * t / T
        g = lambda t: 1 / 320. + (1 / 150. - 1 / 320.) * t / T
        return S.profile(T, H, G=G, Tc=Tc, Rs=Rs, g=g)
    if name == 'E4':
        T = 3.5
        G = lambda t: (ramp(t, 0.5, 0.51, 1000, 300) + ramp(t, 1.0, 1.01, 0, 500)
                       + ramp(t, 1.5, 1.51, 0, -300))
        Tc = lambda t: 25 + 20 * t / T
        RL = lambda t: ramp(t, 2.0, 2.002, 40, 25) + ramp(t, 2.5, 2.502, 0, 30)
        return S.profile(T, H, G=G, Tc=Tc, RL=RL)


def run_scenario(name, modes, seeds, th0, noise):
    fn = f'{OUT}/{name}.pkl'
    if os.path.exists(fn):
        return pickle.load(open(fn, 'rb'))
    p = params(); prof = scen(name)
    res = {}
    for mode in modes:
        res[mode] = []
        for sd in seeds:
            o = run_mode(mode, prof, th0, p, noise, sd)
            res[mode].append(o)
        print(name, mode, 'done', flush=True)
    pickle.dump(res, open(fn, 'wb'))
    return res


if __name__ == '__main__':
    which = sys.argv[1]
    th_pso, c = pso_offline()
    print('PSO theta', th_pso, 'Rsh', 1 / th_pso[1], 'cost', c)
    json.dump(dict(Rs=float(th_pso[0]), Rsh=float(1 / th_pso[1]), cost=float(c)), open(f'{OUT}/pso.json', 'w'))
    bad = [1.0, 1 / 200.]
    allm = [0, 5, 1, 2, 3, 4, 6]
    if which == 'E1': run_scenario('E1', allm, SEEDS['E1'], bad, NONOISE)
    if which == 'E2': run_scenario('E2', allm, SEEDS['E2'], bad, NOISE)
    if which == 'E3': run_scenario('E3', allm, SEEDS['E3'], list(th_pso), NOISE)
    if which == 'E4': run_scenario('E4', allm, SEEDS['E4'], list(th_pso), NOISE)
