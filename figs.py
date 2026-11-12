import numpy as np, pickle, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import root
import pvsim as S, experiments as X, consts

plt.rcParams.update({'font.size': 8, 'axes.grid': True, 'grid.alpha': .3, 'figure.dpi': 150,
                     'lines.linewidth': 1.0, 'legend.fontsize': 6.5})
O = 'out/'
L = {k: pickle.load(open(f'{O}{k}.pkl', 'rb')) for k in ['E1', 'E2', 'E3', 'E4']}
STY = {0: ('C3', '-'), 1: ('C0', '--'), 2: ('k', ':'), 3: ('C2', '-.'), 4: ('C1', '-'), 5: ('C4', '-')}
LAB = {0: 'AB-ISMC (proposed)', 1: 'Fixed-parameter B-ISMC', 2: 'Oracle', 3: 'Conventional SMC',
       4: 'P&O', 5: 'AB-ISMC + dither'}
R = {}
p, pa = X.params()
pd = p
k1, k2, lam, eta, phib = pd['k1'], pd['k2'], pd['lam'], pd['eta'], pd['phib']
C1 = S.C1
beta = 4 / (k1 * k2 * C1 ** 2); mu = beta * k2 * lam; w = 4 / (k1 * C1 ** 2)
Gam = np.array([pd['gR'], pd['gG']]); Sig = np.array([pd['sigl_R'], pd['sigl_G']])


def save(name):
    plt.tight_layout(); plt.savefig(f'{O}{name}.pdf'); plt.close()


# ---------- Fig: E1 tracking
o1 = L['E1']
fig, ax = plt.subplots(2, 2, figsize=(7, 4.2))
for m in [0, 1, 3, 4]:
    o = o1[m][0]; c, ls = STY[m]
    ax[0, 0].plot(o[:, 0], o[:, 6], c=c, ls=ls, label=LAB[m])
    ax[0, 1].plot(o[:, 0], 100 * (o[:, 7] - o[:, 6]) / o[:, 7], c=c, ls=ls)
o = o1[0][0]
ax[0, 0].plot(o[:, 0], o[:, 7], 'k:', lw=.8, label='$P_{mpp}$ (true)')
ax[0, 0].set_ylabel('$P_{pv}$ (W)'); ax[0, 0].legend(ncol=2); ax[0, 0].set_xlabel('t (s)')
ax[0, 1].set_ylim(-1, 12); ax[0, 1].set_ylabel('power deficit (%)'); ax[0, 1].set_xlabel('t (s)')
ax[1, 0].plot(o[:, 0], o[:, 1], 'C3', label='$v_{pv}$'); ax[1, 0].plot(o[:, 0], o[:, 8], 'k:', label='$V_{mpp}$ (true)')
ax[1, 0].plot(o[:, 0], o[:, 9], 'C0--', lw=.7, label='$V_{mpp}(\\hat\\theta)$'); ax[1, 0].legend()
ax[1, 0].set_ylabel('voltage (V)'); ax[1, 0].set_xlabel('t (s)')
ax[1, 1].plot(o[:, 0], o[:, 4], 'C3', lw=.6); ax[1, 1].set_ylabel('duty $d$'); ax[1, 1].set_xlabel('t (s)')
for a, s_ in zip(ax.ravel(), 'abcd'): a.set_title(f'({s_})', loc='left', fontsize=7)
save('fig_e1')

# ---------- Fig: identification E1 / E2 / E3
fig, ax = plt.subplots(2, 3, figsize=(7.2, 3.9))
o = o1[0][0]; t = o[:, 0]
ax[0, 0].plot(t, o[:, 10], 'C3', label='$\\hat R_s$'); ax[0, 0].plot(t, o[:, 12], 'k:', label='$R_s$'); ax[0, 0].set_ylabel('$R_s$ ($\\Omega$)'); ax[0, 0].legend()
ax[1, 0].plot(t, 1 / o[:, 11], 'C3'); ax[1, 0].plot(t, 1 / o[:, 13], 'k:'); ax[1, 0].set_ylabel('$R_{sh}$ ($\\Omega$)')
ax[0, 0].set_title('(a) E1: bad initialisation', fontsize=7)
o = L['E2'][0][0]; t = o[:, 0]
ax[0, 1].plot(t, o[:, 10], 'C3'); ax[0, 1].plot(t, o[:, 12], 'k:'); ax[0, 1].set_title('(b) E2: noisy sensors', fontsize=7)
ax[1, 1].plot(t, 1 / o[:, 11], 'C3'); ax[1, 1].plot(t, 1 / o[:, 13], 'k:')
o = L['E3'][0][0]; t = o[:, 0]
ax[0, 2].plot(t, o[:, 10], 'C3'); ax[0, 2].plot(t, o[:, 12], 'k:'); ax[0, 2].set_title('(c) E3: ageing drift', fontsize=7)
ax[1, 2].plot(t, 1 / o[:, 11], 'C3'); ax[1, 2].plot(t, 1 / o[:, 13], 'k:')
for a in ax[1]: a.set_xlabel('t (s)')
save('fig_id')

# ---------- Lyapunov / sliding (E1, noise-free) -------------
o = o1[0][0]; t = o[:, 0]
th_t = np.stack([o[:, 12], o[:, 13]], 1); th_h = np.stack([o[:, 10], o[:, 11]], 1); tt = th_t - th_h
Vc = 0.5 * o[:, 14] ** 2 + 0.5 * beta * o[:, 15] ** 2 + 0.5 * mu * o[:, 16] ** 2
Vth = 0.5 * w * np.sum(tt ** 2 / Gam, 1)
V = Vc + Vth
kk = 10; Vs = np.convolve(V, np.ones(kk) / kk, 'same'); dV = np.gradient(Vs, t)
fig, ax = plt.subplots(2, 2, figsize=(7, 4))
ax[0, 0].semilogy(t, V, 'k', label='$V$'); ax[0, 0].semilogy(t, Vc + 1e-12, 'C0', lw=.6, label='$V_c$'); ax[0, 0].semilogy(t, Vth, 'C3', lw=.6, label='$V_\\theta$')
ax[0, 0].legend(); ax[0, 0].set_ylim(1e-8, None); ax[0, 0].set_title('(a) composite Lyapunov function', fontsize=7)
ax[0, 1].plot(t, dV / V[0], 'k', lw=.6); ax[0, 1].set_ylim(-60, 20); ax[0, 1].set_title('(b) $\\dot V/V(0)$ (1 ms smoothing)', fontsize=7)
ax[1, 0].plot(t, o[:, 15], 'C2', lw=.6); ax[1, 0].set_title('(c) sliding variable $s$ (A)', fontsize=7); ax[1, 0].set_xlabel('t (s)')
ax[1, 1].plot(t, o[:, 14], 'C0', lw=.6); ax[1, 1].plot(t, o[:, 17], 'C1', lw=.6); ax[1, 1].set_title('(d) $e_1$ (V, blue) and $\\varepsilon$ (A, orange)', fontsize=7); ax[1, 1].set_xlabel('t (s)')
ax[1, 1].set_ylim(-3, 3); ax[1, 0].set_ylim(-0.5, 0.5)
save('fig_lyap')
post = t > 0.5
R['lyap'] = dict(V0=float(V[0]), Vmax_post=float(V[post].max()), Vmed_post=float(np.median(V[post])),
                 frac_dV_neg_above=float(np.mean(dV[(V > 10 * np.median(V[post]))] < 0)),
                 s_rms=float(np.sqrt(np.mean(o[post, 15] ** 2))), s_abs_max=float(np.abs(o[post, 15]).max()),
                 e1_rms=float(np.sqrt(np.mean(o[post, 14] ** 2))), eps_rms=float(np.sqrt(np.mean(o[post, 17] ** 2))),
                 e1_max=float(np.abs(o[post, 14]).max()), eps_max=float(np.abs(o[post, 17]).max()),
                 Vc_max_post=float(Vc[post].max()))
# ISS gain check: e1 <= sqrt(2 V_c,inf), V_c,inf = (eps^2/(k1 C1^2) + beta eta phib)/kappa_c
kc_ = min(k1, k2, lam / 2)
R['iss'] = dict(gain=float(np.sqrt(2 / (k1 * C1 ** 2 * kc_))), kappa_c=kc_, beta=beta, mu=mu, w=w,
                floor=float(np.sqrt(2 * beta * eta * phib / kc_ / 1.0)))

# ---------- chatter zoom
fig, ax = plt.subplots(1, 2, figsize=(7, 2.3))
for m in [0, 3, 4]:
    o = o1[m][0]; k = (o[:, 0] > 0.60) & (o[:, 0] < 0.64); ax[0].plot(o[k, 0], o[k, 4], c=STY[m][0], ls=STY[m][1], label=LAB[m])
ax[0].legend(); ax[0].set_xlabel('t (s)'); ax[0].set_ylabel('duty $d$'); ax[0].set_title('(a) noise-free', fontsize=7)
for m in [0, 3]:
    o = L['E2'][m][0]; k = (o[:, 0] > 0.60) & (o[:, 0] < 0.64); ax[1].plot(o[k, 0], o[k, 4], c=STY[m][0], ls=STY[m][1], lw=.6)
ax[1].set_xlabel('t (s)'); ax[1].set_title('(b) noisy sensors', fontsize=7)
save('fig_duty')

# ---------- E3 drift
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
for m in [0, 1, 2]:
    o = L['E3'][m][0]; t = o[:, 0]
    ax[0].plot(t, o[:, 9] - o[:, 8], c=STY[m][0], ls=STY[m][1], label=LAB[m], lw=.8)
    ax[1].plot(t, o[:, 1] - o[:, 8], c=STY[m][0], ls=STY[m][1], lw=.5)
    ax[2].plot(t, 100 * (o[:, 7] - o[:, 6]) / o[:, 7], c=STY[m][0], ls=STY[m][1], lw=.5)
ax[0].legend(); ax[0].set_title('(a) $V_{mpp}(\\hat\\theta)-V_{mpp}$ (V)', fontsize=7); ax[1].set_title('(b) $v_{pv}-V_{mpp}$ (V)', fontsize=7)
ax[2].set_title('(c) power deficit (%)', fontsize=7); ax[2].set_ylim(-1, 3); ax[1].set_ylim(-3, 3)
for a in ax: a.set_xlabel('t (s)')
save('fig_drift')

# ---------- E4
fig, ax = plt.subplots(1, 2, figsize=(7, 2.5))
for m in [0, 3, 4, 1]:
    o = L['E4'][m][0]; ax[0].plot(o[:, 0], o[:, 6], c=STY[m][0], ls=STY[m][1], lw=.7, label=LAB[m])
o = L['E4'][0][0]; ax[0].plot(o[:, 0], o[:, 7], 'k:', lw=.8, label='$P_{mpp}$')
ax[0].legend(ncol=2); ax[0].set_xlabel('t (s)'); ax[0].set_ylabel('$P_{pv}$ (W)')
ax[1].plot(o[:, 0], o[:, 18] / 10, label='$G/10$ (W/m$^2$)'); ax[1].plot(o[:, 0], o[:, 20], label='$R_L$ ($\\Omega$)'); ax[1].plot(o[:, 0], o[:, 19], label='$T$ ($^\\circ$C)')
ax[1].legend(); ax[1].set_xlabel('t (s)')
save('fig_e4')

# ---------- sensitivity / identifiability
Iph, I0, a = S.pv_consts(1000, 25); vm = S.vmpp(0.35, 1 / 320, Iph, I0, a, 29.)
Pm = vm * S.sdm_I(vm, .35, 1 / 320, Iph, I0, a)
Rse = np.linspace(0.1, 1.2, 111); lossR = []; Vref = []
for r_ in Rse:
    v2 = S.vmpp(r_, 1 / 320, Iph, I0, a, 29.); Vref.append(v2); lossR.append(100 * (1 - v2 * S.sdm_I(v2, .35, 1 / 320, Iph, I0, a) / Pm))
Rhe = np.linspace(80, 1000, 100); lossH = []
for r_ in Rhe:
    v2 = S.vmpp(.35, 1 / r_, Iph, I0, a, 29.); lossH.append(100 * (1 - v2 * S.sdm_I(v2, .35, 1 / 320, Iph, I0, a) / Pm))
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
ax[0].plot(Rse, lossR); ax[0].set_xlabel('$\\hat R_s$ ($\\Omega$)'); ax[0].set_ylabel('power loss (%)'); ax[0].axvline(.35, c='k', ls=':')
ax[0].set_title('(a) loss vs. $R_s$ error', fontsize=7)
ax[1].semilogy(Rhe, np.array(lossH) + 1e-6); ax[1].set_xlabel('$\\hat R_{sh}$ ($\\Omega$)'); ax[1].set_title('(b) loss vs. $R_{sh}$ error', fontsize=7)
# identifiability: normalised regressors along STC I-V neighbourhood
vv = np.linspace(26, 31, 50); PH = []
for v in vv:
    _, fv, fR, fg = S.sdm_all(v, .35, 1 / 320, Iph, I0, a); PH.append([fR * .35, fg * (1 / 320)])
PH = np.array(PH); M = PH.T @ PH / len(vv); ev = np.linalg.eigvalsh(M)
ax[2].plot(vv, PH[:, 0], label='$\\partial f/\\partial \\ln R_s$'); ax[2].plot(vv, PH[:, 1], label='$\\partial f/\\partial \\ln g_{sh}$')
ax[2].legend(); ax[2].set_xlabel('$v_{pv}$ (V)'); ax[2].set_title('(c) regressors (A)', fontsize=7)
save('fig_sens')
iR = np.argmin(np.abs(Rse - 0.63)); iR2 = np.argmin(np.abs(Rse - 0.8))
R['sens'] = dict(Vmpp=float(vm), Pmpp=float(Pm), loss_Rs063=float(lossR[iR]), loss_Rs080=float(lossR[iR2]), Vref_Rs063=float(Vref[iR]),
                 loss_Rsh100=float(np.interp(100, Rhe, lossH)), loss_Rsh1000=float(lossH[-1]), eig_ratio=float(ev[1] / ev[0]), ev=[float(e) for e in ev],
                 dV_dRs=float((S.vmpp(.36, 1 / 320, Iph, I0, a, vm) - S.vmpp(.34, 1 / 320, Iph, I0, a, vm)) / .02),
                 dV_dRsh=float((S.vmpp(.35, 1 / 321, Iph, I0, a, vm) - S.vmpp(.35, 1 / 319, Iph, I0, a, vm)) / 2),
                 fR=float(S.sdm_all(vm, .35, 1 / 320, Iph, I0, a)[2]), fg=float(S.sdm_all(vm, .35, 1 / 320, Iph, I0, a)[3]))

# ---------- local eigenvalue analysis (7 states, estimator included)
def rhs(x, G, Tc, Rst, gt, RL, th0, dl=1e-2):
    v, iL, vo, sig, thR, thg, Vd = x
    Iph, I0, a = S.pv_consts(G, Tc)
    ipv = S.sdm_I(v, Rst, gt, Iph, I0, a)
    fh, fv, fR, fg = S.sdm_all(v, thR, thg, Iph, I0, a); eps = ipv - fh
    tR = pd['gR'] * (fR * eps - pd['sigl_R'] * (thR - th0[0])); tg = pd['gG'] * (fg * eps - pd['sigl_G'] * (thg - th0[1]))
    Vr = S.vmpp(thR, thg, Iph, I0, a, 28.); Vr2 = S.vmpp(thR + dl * tR, thg + dl * tg, Iph, I0, a, Vr)
    Vrd = (Vr2 - Vr) / dl; Vdd_ = -pd['wr'] * (Vd - Vr); Vddd = -pd['wr'] * (Vdd_ - Vrd)
    e1 = v - Vd; al = fh + C1 * (k1 * e1 - Vdd_); e2 = iL - al; s = e2 + lam * sig
    fh2 = S.sdm_I(v, thR + dl * tR, thg + dl * tg, Iph, I0, a); fdo = (fh2 - fh) / dl
    a1 = fv * (fh - iL) / C1 + fdo + C1 * k1 * (-k1 * e1 - e2 / C1) - C1 * Vddd + (fv / C1 + k1) * eps
    kc = k1 * k2 * C1 / 4
    u = v - S.Lind * (a1 - lam * e2 - k2 * s - eta * np.clip(s / phib, -1, 1) + kc * e1); d = 1 - u / vo
    return np.array([(ipv - iL) / C1, (v - pd['rL'] * iL - (1 - d) * vo) / S.Lind, ((1 - d) * iL - vo / RL) / S.C2, e2, tR, tg, Vdd_])

eig_rows = []
th0 = [1.0, 1 / 200.]
for (G, Tc, Rs_, Rsh_, RL) in [(1000, 25, .35, 320, 40), (550, 25, .35, 320, 40), (900, 40, .35, 320, 40), (300, 25, .35, 320, 40), (1000, 45, .8, 150, 40), (800, 35, .6, 200, 25)]:
    prof = S.profile(3.0, 2e-5, G=G, Tc=Tc, Rs=Rs_, g=1 / Rsh_, RL=RL)
    o = S.run(0, prof, th0, (pd, pa), h=2e-5)
    x0 = np.array([o[-1, 1], o[-1, 2], o[-1, 3], o[-1, 16], o[-1, 10], o[-1, 11], o[-1, 9]])
    f = lambda x: rhs(x, G, Tc, Rs_, 1 / Rsh_, RL, th0)
    sol = root(f, x0, method='hybr', tol=1e-12); xs = sol.x
    st = np.array([1e-5, 1e-5, 1e-4, 1e-7, 1e-6, 1e-8, 1e-5]); A = np.zeros((7, 7))
    for j in range(7):
        e = np.zeros(7); e[j] = st[j]; A[:, j] = (f(xs + e) - f(xs - e)) / (2 * st[j])
    ev_ = np.linalg.eigvals(A)
    eig_rows.append(dict(G=G, T=Tc, Rs=Rs_, Rsh=Rsh_, RL=RL, resid=float(np.abs(f(xs)).max()), maxRe=float(ev_.real.max()),
                         minRe=float(ev_.real.min()), eigs=[[float(e.real), float(e.imag)] for e in ev_]))
R['eig'] = eig_rows
fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
for r_ in eig_rows:
    e = np.array(r_['eigs']); ax[0].plot(e[:, 0], e[:, 1], 'x', ms=4)
    ax[1].plot(e[:, 0], e[:, 1], 'x', ms=4, label=f"G={r_['G']}, T={r_['T']}")
ax[0].set_xscale('symlog', linthresh=1e-2); ax[0].set_xlabel('Re($\\lambda$)'); ax[0].set_ylabel('Im($\\lambda$)')
ax[1].set_xlim(-0.3, 0.02); ax[1].set_ylim(-1, 1); ax[1].set_xlabel('Re($\\lambda$), zoom near origin'); ax[1].legend(ncol=2)
save('fig_eig')

# ---------- Monte-Carlo summary figure (E2)
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2)); ms = [0, 1, 3, 4]
for a_, key, ttl in zip(ax, ['eff', 'vrmse', 'ci'], ['avg. efficiency (%)', 'RMS $v_{pv}-V_{mpp}$ (%)', 'chatter index (% duty)']):
    vals = [[X.metrics(o, steps=X.STEPS['E2'])[key] for o in L['E2'][m]] for m in ms]
    a_.boxplot(vals, tick_labels=['prop.', 'fixed', 'conv.', 'P&O'], showfliers=False); a_.set_title(ttl, fontsize=7)
save('fig_mc')

# ---------- tables
def fmt(ms_, key, d=2): return f'{np.mean(ms_[key]):.{d}f}' + (f'$\\pm${np.std(ms_[key]):.{d}f}' if len(ms_[key]) > 1 else '')
def table(sc, cols, modes, ts=None):
    rows = []
    for m in modes:
        Ms = [X.metrics(o, steps=X.STEPS[sc]) for o in L[sc][m]]; D = {k: [x[k] for x in Ms] for k in Ms[0]}
        cells = [fmt(D, k, 2 if k not in ('eref',) else 2) for k in cols]
        rows.append(f"{X.NAMES[m]} & " + ' & '.join(cells) + r' \\')
    return '\n'.join(rows)
TAB = {}
for sc in ['E1', 'E2', 'E3', 'E4']:
    TAB[sc] = table(sc, ['eff', 'vrmse', 'ci', 'eRs', 'eRsh', 'eref'], [0, 5, 1, 2, 3, 4, 6])
json.dump(TAB, open(O + 'tables.json', 'w'))
# settling after step (E1 at t=1 s, 1% band) & P&O
R['settle'] = {str(m): float(X.settle(L['E1'][m][0], 1.0, 1.95, 0.01)) for m in [0, 1, 3, 4]}
R['pso'] = json.load(open(O + 'pso.json'))
R['const'] = consts.compute_constants(pd, v_lo=19.9, v_hi=31.2)
R['gains'] = dict(beta=beta, mu=mu, w=w, kappa=min(k1, k2, lam / 2, float(np.min(Sig * Gam))), kappa_theta=float(np.min(Sig * Gam)))
R['metrics'] = {}
for sc in ['E1', 'E2', 'E3', 'E4']:
    R['metrics'][sc] = {}
    for m in [0, 5, 1, 2, 3, 4, 6]:
        Ms = [X.metrics(o, steps=X.STEPS[sc]) for o in L[sc][m]]
        R['metrics'][sc][str(m)] = {k: [float(np.mean([x[k] for x in Ms])), float(np.std([x[k] for x in Ms])),
                                        float(np.min([x[k] for x in Ms])), float(np.max([x[k] for x in Ms]))] for k in Ms[0]}
json.dump(R, open(O + 'results.json', 'w'), indent=1)
print(json.dumps({k: R[k] for k in ['lyap', 'iss', 'settle', 'const', 'gains']}, indent=1))
print([(r_['G'], r_['T'], round(r_['resid'], 6), round(r_['maxRe'], 5), round(r_['minRe'], 1)) for r_ in eig_rows])
