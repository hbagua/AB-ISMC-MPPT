import numpy as np, pvsim as S
def compute_constants(p, nu_bar=0.3, v_lo=24., v_hi=31., Gs=(300,550,1000), Ts=(25,45), seed=1):
    """Constants of Theorem 1 / Lemma 1 by sampling D x Theta (v in [v_lo, v_hi])."""
    rng=np.random.default_rng(seed)
    Rs_lo,Rs_hi,g_lo,g_hi=p['Rsmin'],p['Rsmax'],p['gmin'],p['gmax']
    D=np.hypot(Rs_hi-Rs_lo,g_hi-g_lo)
    rho_eps=0.; varrho=0.; Ltheta=0.; LV=0.; fvmax=0.; psi=0.; fRmax=0.; fgmax=0.
    Vr_lo=1e9; Vr_hi=-1e9
    for G in Gs:
      for T in Ts:
        Iph,I0,a=S.pv_consts(G,T)
        for _ in range(4000):
            v=rng.uniform(v_lo,v_hi)
            Rs=rng.uniform(Rs_lo,Rs_hi); g=rng.uniform(g_lo,g_hi)
            Rh=rng.uniform(Rs_lo,Rs_hi); gh=rng.uniform(g_lo,g_hi)
            I,fv,fR,fg=S.sdm_all(v,Rh,gh,Iph,I0,a)
            It=S.sdm_I(v,Rs,g,Iph,I0,a)
            r=It-I-(fR*(Rs-Rh)+fg*(g-gh))
            varrho=max(varrho,abs(r)); Ltheta=max(Ltheta,np.hypot(fR,fg))
            rho_eps=max(rho_eps,abs(fv/S.C1+p['k1'])); fvmax=max(fvmax,abs(fv))
            psi=max(psi,p['gR']*fR**2+p['gG']*fg**2); fRmax=max(fRmax,abs(fR)); fgmax=max(fgmax,abs(fg))
        # corners of the box in v and theta (the suprema sit on the boundary)
        for v in (v_lo,v_hi):
          for Rh in (Rs_lo,Rs_hi):
            for gh in (g_lo,g_hi):
                I,fv,fR,fg=S.sdm_all(v,Rh,gh,Iph,I0,a)
                psi=max(psi,p['gR']*fR**2+p['gG']*fg**2); fRmax=max(fRmax,abs(fR)); fgmax=max(fgmax,abs(fg))
                rho_eps=max(rho_eps,abs(fv/S.C1+p['k1'])); fvmax=max(fvmax,abs(fv))
        # L_V: sup ||dVmpp/dtheta|| by central differences; range of the reference V_r over Theta
        for _ in range(60):
            Rs=rng.uniform(Rs_lo,Rs_hi); g=rng.uniform(g_lo,g_hi)
            dR=1e-4; dg=1e-6
            v0=S.vmpp(Rs,g,Iph,I0,a,29.)
            dVR=(S.vmpp(Rs+dR,g,Iph,I0,a,v0)-S.vmpp(Rs-dR,g,Iph,I0,a,v0))/(2*dR)
            dVg=(S.vmpp(Rs,g+dg,Iph,I0,a,v0)-S.vmpp(Rs,g-dg,Iph,I0,a,v0))/(2*dg)
            LV=max(LV,np.hypot(dVR,dVg))
        for Rs in np.linspace(Rs_lo,Rs_hi,23):
            for g in (g_lo,g_hi):
                vr=S.vmpp(Rs,g,Iph,I0,a,27.); Vr_lo=min(Vr_lo,vr); Vr_hi=max(Vr_hi,vr)
    mbar=max(p['gR']*p['sigl_R'],p['gG']*p['sigl_G'])
    Iph,I0,a=S.pv_consts(1000,25); vm=S.vmpp(.35,1/320,Iph,I0,a,29.)
    _,_,fR,fg=S.sdm_all(vm,.35,1/320,Iph,I0,a); psi_mpp=p['gR']*fR**2+p['gG']*fg**2
    gmax_=max(p['gR'],p['gG']); smax=max(p['sigl_R'],p['sigl_G'])
    return dict(D=D,varrho=varrho,Ltheta=Ltheta,rho_eps=rho_eps,LV=LV,fvmax=fvmax,eps_bar=Ltheta*D,
                psi_bar=psi,psi_mpp=psi_mpp,fRmax=fRmax,fgmax=fgmax,mbar=mbar,Vr_lo=Vr_lo,Vr_hi=Vr_hi,
                v_lo=v_lo,v_hi=v_hi,
                Ts_est=1/(12*psi+6*mbar),Ts_est_mpp=1/(12*psi_mpp+6*mbar),
                Ts_est_unw=1/(12*gmax_*Ltheta**2+6*smax*gmax_))
if __name__=="__main__":
    import experiments as X, json
    p,_=X.params()
    for lo,hi in ((24.,31.),(19.,31.)):
        print(json.dumps(compute_constants(p,v_lo=lo,v_hi=hi),indent=1))
