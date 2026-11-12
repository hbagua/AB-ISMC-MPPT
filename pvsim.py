"""
pvsim.py -- single-diode PV module + averaged boost converter + controllers.
Reproduces every number/figure of the revised manuscript.
Unknown parameter vector: theta = [Rs, g_sh],  g_sh = 1/Rsh  (S).
Controllers (mode):
 0 proposed AB-ISMC with on-line adaptation law (leakage + projection + dither)
 1 fixed-parameter backstepping ISMC (theta frozen at offline estimate)
 2 oracle backstepping ISMC (true theta(t) known)
 3 conventional current-mode integral SMC (fixed offline theta)
 4 perturb & observe on duty cycle
 6 cascaded P&O (outer, V_ref) + PI voltage loop (inner, every T_s)
(mode 5 = mode 0 with reference dither, see experiments.run_mode)
Implementation matches the manuscript: Eq. (10) with component-wise clamp, conditional
anti-windup, omega_r = 500 rad/s, x(0) = (28 V, 7.5 A, 80 V), six filtered/noisy channels,
MT19937 seeds, optional switching-level PWM plant with transport delay.
"""
import numpy as np
from numba import njit
from math import exp, log, sqrt, sin, cos, pi, fabs

q = 1.602176634e-19; kB = 1.380649e-23
Ns = 60; nid = 1.3; Isc_ref = 8.35; Voc_ref = 36.3
Ki = 4.2e-3; Tref = 298.15; Eg = 1.12
C1 = 470e-6; C2 = 470e-6; Lind = 3e-3


@njit(cache=True)
def lambertw_log(y):
    """principal branch W0(exp(y)), y real"""
    x = exp(y)
    if y > 1.0:
        w = y - log(y)
    else:
        w = x / (1.0 + x)
    for _ in range(10):
        ew = exp(w)
        f = w * ew - x
        w = w - f / (ew * (w + 1.0) - (w + 2.0) * f / (2.0 * w + 2.0))
    return w


@njit(cache=True)
def pv_consts(G, Tc):
    Tk = Tc + 273.15
    a = nid * Ns * kB * Tk / q
    Iph = (Isc_ref + Ki * (Tk - Tref)) * G / 1000.0
    aref = nid * Ns * kB * Tref / q
    I0ref = Isc_ref / (exp(Voc_ref / aref) - 1.0)
    I0 = I0ref * (Tk / Tref) ** 3 * exp(Eg * q / (nid * kB) * (1.0 / Tref - 1.0 / Tk))
    return Iph, I0, a


@njit(cache=True)
def sdm_I(v, Rs, g, Iph, I0, a):
    den = 1.0 + Rs * g
    y = log(Rs * I0 / (a * den)) + (Rs * (Iph + I0) + v) / (a * den)
    return ((Iph + I0) - v * g) / den - (a / Rs) * lambertw_log(y)


@njit(cache=True)
def sdm_all(v, Rs, g, Iph, I0, a):
    I = sdm_I(v, Rs, g, Iph, I0, a)
    E = exp((v + I * Rs) / a)
    k = I0 * E / a + g
    D = 1.0 + Rs * k
    return I, -k / D, -I * k / D, -(v + I * Rs) / D   # I, f_v, f_Rs, f_g


@njit(cache=True)
def dPdV(v, Rs, g, Iph, I0, a):
    I, fv, _, _ = sdm_all(v, Rs, g, Iph, I0, a)
    return I + v * fv


@njit(cache=True)
def vmpp(Rs, g, Iph, I0, a, v0):
    v = v0
    for _ in range(15):
        d = 1e-3
        p0 = dPdV(v, Rs, g, Iph, I0, a)
        dd = (dPdV(v + d, Rs, g, Iph, I0, a) - dPdV(v - d, Rs, g, Iph, I0, a)) / (2 * d)
        if dd > -1e-3:
            dd = -1e-3
        st = -p0 / dd
        if st > 3.0: st = 3.0
        if st < -3.0: st = -3.0
        v += st
        if fabs(st) < 1e-9:
            break
    return v


@njit(cache=True)
def sat(x):
    return max(-1.0, min(1.0, x))


NC = 28  # log columns


@njit(cache=True)
def simulate(mode, h, N, G, Tc, Rst, gt, RLt, noise, P, th0, x0, stride, nsub, Nd, sw):
    """h = controller sampling period T_s (zero-order hold on d).
    sw = 0: averaged boost model, plant integrated with nsub Euler sub-steps of h/nsub.
    sw = 1: switching-level model, centred (symmetric) PWM with period h; the switching
            instants are placed exactly (no duty quantisation) and each of the three
            sub-intervals (off/on/off) is integrated with nsub Euler sub-steps.
    Nd   : transport delay in whole sampling periods between the computed and applied duty."""
    k1 = P[0]; k2 = P[1]; lam = P[2]; eta = P[3]; phib = P[4]
    wr = P[5]; ad = P[6]; wd = 2 * pi * P[7]
    gR = P[8]; gG = P[9]; sigl_R = P[10]; sigl_G = P[11]
    tauG = P[12]; tauS = P[13]; rL = P[14]; dmin = P[15]; dmax = P[16]
    Rsmin = P[17]; Rsmax = P[18]; gmin = P[19]; gmax = P[20]
    lam_c = P[21]; eta_c = P[22]; po_T = P[23]; po_step = P[24]
    kc_s = P[25]; tauT = P[26]; pi_Kp = P[27]; pi_Ki = P[28]; pi_T = P[29]; pi_dV = P[30]
    Rds = P[31]; VD = P[32]
    kc = k1 * k2 * C1 / 4.0
    alS = h / (tauS + h)

    v = x0[0]; iL = x0[1]; vo = x0[2]
    thR = th0[0]; thg = th0[1]
    sig = 0.0
    Gf = G[0]; Tf = Tc[0]
    vm = v; iLm = iL; vom = vo
    Iph, I0, a = pv_consts(G[0], Tc[0])
    ipvm = sdm_I(v, Rst[0], gt[0], Iph, I0, a)
    Iphf, I0f, af = pv_consts(Gf, Tf)
    Vr = vmpp(thR, thg, Iphf, I0f, af, 29.0)
    Vd = Vr
    vstar = vmpp(Rst[0], gt[0], Iph, I0, a, 29.0)
    d = 0.68
    Iref_prev = sdm_I(Vr, th0[0], th0[1], Iphf, I0f, af)
    po_dir = 1.0; po_Pprev = 0.0; po_cnt = 0
    Vref_po = v; pi_int = 0.68 / pi_Ki
    dbuf = np.full(8, d)
    nlog = N // stride
    out = np.zeros((nlog, NC))
    jl = 0
    for i in range(N):
        t = i * h
        Gt = G[i]; Tt = Tc[i]; Rs_t = Rst[i]; g_t = gt[i]; RL = RLt[i]
        Iph, I0, a = pv_consts(Gt, Tt)
        ipv = sdm_I(v, Rs_t, g_t, Iph, I0, a)
        # ---- sensors (noise + first-order conditioning filter), all six channels
        vm += alS * (v + noise[i, 0] - vm)
        iLm += alS * (iL + noise[i, 1] - iLm)
        ipvm += alS * (ipv + noise[i, 2] - ipvm)
        vom += alS * (vo + noise[i, 3] - vom)
        Gf_n = Gf + h / tauG * (Gt + noise[i, 4] - Gf)
        Tf_n = Tf + h / tauT * (Tt + noise[i, 5] - Tf)
        e1 = 0.0; s = 0.0; eps_m = 0.0; vref = 0.0
        dcmd = d
        if mode <= 2:
            Iphf, I0f, af = pv_consts(Gf, Tf)
            fh, fv, fRs, fg = sdm_all(vm, thR, thg, Iphf, I0f, af)
            eps_m = ipvm - fh
            thR_n = thR; thg_n = thg
            if mode == 0:
                # Eq. (10): forward Euler + component-wise clamp Pi_Theta
                tR = gR * (fRs * eps_m - sigl_R * (thR - th0[0]))
                tg = gG * (fg * eps_m - sigl_G * (thg - th0[1]))
                thR_n = min(Rsmax, max(Rsmin, thR + h * tR))
                thg_n = min(gmax, max(gmin, thg + h * tg))
            elif mode == 2:
                j = min(i + 1, N - 1)
                thR_n = Rst[j]; thg_n = gt[j]
            Iphn, I0n, an = pv_consts(Gf_n, Tf_n)
            Vr_next = vmpp(thR_n, thg_n, Iphn, I0n, an, Vr)
            Vrdot = (Vr_next - Vr) / h
            Vd_dot = -wr * (Vd - Vr)
            Vd_dd = -wr * (Vd_dot - Vrdot)
            dith = ad if mode == 0 else 0.0
            vref = Vd + dith * sin(wd * t)
            vrd = Vd_dot + dith * wd * cos(wd * t)
            vrdd = Vd_dd - dith * wd * wd * sin(wd * t)
            e1 = vm - vref
            alpha1 = fh + C1 * (k1 * e1 - vrd)
            e2 = iLm - alpha1
            s = e2 + lam * sig
            fh_n = sdm_I(vm, thR_n, thg_n, Iphn, I0n, an)
            fdot_o = (fh_n - fh) / h
            a1dot = (fv * (fh - iLm) / C1 + fdot_o + C1 * k1 * (-k1 * e1 - e2 / C1)
                     - C1 * vrdd + (fv / C1 + k1) * eps_m)
            u = vm - Lind * (a1dot - lam * e2 - k2 * s - eta * sat(s / phib) + kc * e1)
            dcmd = 1.0 - u / vom
            if dcmd <= dmax and dcmd >= dmin:      # conditional anti-windup
                sig += h * e2
            Vd += h * Vd_dot
            thR = thR_n; thg = thg_n; Vr = Vr_next
        elif mode == 3:
            Iphf, I0f, af = pv_consts(Gf, Tf)
            Vr = vmpp(th0[0], th0[1], Iphf, I0f, af, Vr)
            Iref = sdm_I(Vr, th0[0], th0[1], Iphf, I0f, af)
            Irefdot = (Iref - Iref_prev) / h; Iref_prev = Iref
            e1 = iLm - Iref
            s = e1 + lam_c * sig
            u = vm - Lind * (Irefdot - lam_c * e1 - kc_s * s - eta_c * sat(s / phib))
            dcmd = 1.0 - u / vom
            if dcmd <= dmax and dcmd >= dmin:
                sig += h * e1
        elif mode == 4:  # direct duty-ratio P&O
            po_cnt += 1
            if po_cnt >= int(round(po_T / h)):
                po_cnt = 0
                Pm = vm * ipvm
                if Pm < po_Pprev:
                    po_dir = -po_dir
                po_Pprev = Pm
                d += po_dir * po_step
            dcmd = d
        else:  # mode 6: cascaded P&O (outer, on V_ref) + PI voltage loop (inner, every T_s)
            po_cnt += 1
            if po_cnt >= int(round(pi_T / h)):
                po_cnt = 0
                Pm = vm * ipvm
                if Pm < po_Pprev:
                    po_dir = -po_dir
                po_Pprev = Pm
                Vref_po += po_dir * pi_dV
            e1 = vm - Vref_po
            dcmd = pi_Kp * e1 + pi_Ki * (pi_int + h * e1)
            if dcmd <= dmax and dcmd >= dmin:
                pi_int += h * e1
            vref = Vref_po
        Gf = Gf_n; Tf = Tf_n
        d = dcmd
        if d > dmax: d = dmax
        if d < dmin: d = dmin
        # ---- transport delay (Nd whole periods)
        for jb in range(7, 0, -1):
            dbuf[jb] = dbuf[jb - 1]
        dbuf[0] = d
        da = dbuf[Nd]
        # ---- plant; inductor ESR rL (and Rds, VD) are NOT known to the controller
        v_s = v; iL_s = iL; vo_s = vo
        iLmin = iL; iLmax = iL; vmin = v; vmax = v
        if sw == 0:
            hs = h / nsub
            for _ in range(nsub):
                ip = sdm_I(v, Rs_t, g_t, Iph, I0, a)
                dv = (ip - iL) / C1
                diL = (v - rL * iL - (1.0 - da) * vo) / Lind
                dvo = ((1.0 - da) * iL - vo / RL) / C2
                v += hs * dv; iL += hs * diL; vo += hs * dvo
                if iL < 0.0: iL = 0.0
        else:
            for seg in range(3):
                if seg == 1:
                    q_on = 1.0; hs = da * h / nsub
                else:
                    q_on = 0.0; hs = 0.5 * (1.0 - da) * h / nsub
                for _ in range(nsub):
                    ip = sdm_I(v, Rs_t, g_t, Iph, I0, a)
                    dv = (ip - iL) / C1
                    diL = (v - (1.0 - q_on) * (vo + VD) - (rL + q_on * Rds) * iL) / Lind
                    dvo = ((1.0 - q_on) * iL - vo / RL) / C2
                    v += hs * dv; iL += hs * diL; vo += hs * dvo
                    if iL < 0.0: iL = 0.0
                    if iL < iLmin: iLmin = iL
                    if iL > iLmax: iLmax = iL
                    if v < vmin: vmin = v
                    if v > vmax: vmax = v
        # ---- logging (quantities at the sampling instant t_k)
        if i % stride == 0 and jl < nlog:
            vstar = vmpp(Rs_t, g_t, Iph, I0, a, vstar)
            Pst = vstar * sdm_I(vstar, Rs_t, g_t, Iph, I0, a)
            fhat_true = sdm_I(v_s, thR, thg, Iph, I0, a)
            out[jl, 0] = t; out[jl, 1] = v_s; out[jl, 2] = iL_s; out[jl, 3] = vo_s
            out[jl, 4] = d; out[jl, 5] = ipv; out[jl, 6] = v_s * ipv
            out[jl, 7] = Pst; out[jl, 8] = vstar; out[jl, 9] = Vr
            out[jl, 10] = thR; out[jl, 11] = thg; out[jl, 12] = Rs_t; out[jl, 13] = g_t
            out[jl, 14] = e1; out[jl, 15] = s; out[jl, 16] = sig
            out[jl, 17] = ipv - fhat_true; out[jl, 18] = Gt; out[jl, 19] = Tt
            out[jl, 20] = RL; out[jl, 21] = vref; out[jl, 22] = eps_m; out[jl, 23] = vm
            out[jl, 24] = 1.0 if (dcmd > dmax or dcmd < dmin) else 0.0
            out[jl, 25] = dcmd; out[jl, 26] = iLmax - iLmin; out[jl, 27] = vmax - vmin
            jl += 1
    return out


def default_params(**kw):
    p = dict(k1=800., k2=4000., lam=1000., eta=1200., phib=0.05, wr=500.,
             ad=0.5, fd=40., gR=0.15, gG=0.01, sigl_R=0.5, sigl_G=0.5,
             tauG=5e-3, tauS=1e-4, rL=0.10, dmin=0.05, dmax=0.95,
             Rsmin=0.10, Rsmax=1.5, gmin=1/1000., gmax=1/60.,
             lam_c=1000., eta_c=1200., po_T=0.01, po_step=0.003,
             kc_s=4000., tauT=1e-3, pi_Kp=0.08, pi_Ki=120., pi_T=2e-3, pi_dV=0.15,
             Rds=0.040, VD=0.40)
    p.update(kw)
    order = ['k1','k2','lam','eta','phib','wr','ad','fd','gR','gG','sigl_R','sigl_G',
             'tauG','tauS','rL','dmin','dmax','Rsmin','Rsmax','gmin','gmax',
             'lam_c','eta_c','po_T','po_step','kc_s','tauT','pi_Kp','pi_Ki','pi_T','pi_dV',
             'Rds','VD']
    return p, np.array([p[k] for k in order], dtype=np.float64)


def profile(T, h, G=None, Tc=None, Rs=None, g=None, RL=None):
    N = int(round(T / h)); t = np.arange(N) * h
    f = lambda x, d: (x(t) if callable(x) else np.full(N, d if x is None else x))
    return t, f(G, 1000.), f(Tc, 25.), f(Rs, 0.35), f(g, 1/320.), f(RL, 40.)


X0 = (28.0, 7.5, 80.0)   # v(0), i_L(0), v_o(0)  (Section 8.1)


def run(mode, prof, th0, p, h=2e-5, noise_rms=(0, 0, 0, 0, 0, 0), seed=1, x0=X0, stride=5,
        nsub=1, Nd=0, sw=0):
    """noise_rms: post-filter rms of (v, iL, i_pv, vo, G, T). RNG: MT19937(seed)."""
    t, G, Tc, Rs, g, RL = prof
    N = len(t)
    rng = np.random.Generator(np.random.MT19937(seed))
    pd, pa = p
    nr = np.zeros(6); nr[:len(noise_rms)] = noise_rms
    # raw white-noise std so that the *filtered* rms equals noise_rms
    tau = np.array([pd['tauS']] * 4 + [pd['tauG'], pd['tauT']])
    al = h / (tau + h)
    al[4] = h / pd['tauG']; al[5] = h / pd['tauT']
    fac = 1.0 / np.sqrt(al / (2 - al))
    noise = rng.normal(size=(N, 6)) * (nr * fac)
    return simulate(mode, h, N, G, Tc, Rs, g, RL, noise, pa, np.array(th0, float),
                    np.array(x0, float), stride, nsub, Nd, sw)
