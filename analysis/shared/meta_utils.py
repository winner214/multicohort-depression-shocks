#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, t


def normal_p(z: float) -> float:
    if not np.isfinite(z):
        return np.nan
    return math.erfc(abs(float(z)) / math.sqrt(2.0))


def _clean(dat: pd.DataFrame) -> pd.DataFrame:
    d = dat.copy()
    if "coef" not in d and "OR" in d:
        d["coef"] = np.log(pd.to_numeric(d["OR"], errors="coerce"))
    if "se" not in d and {"OR_CI_low", "OR_CI_high"}.issubset(d.columns):
        lo = pd.to_numeric(d["OR_CI_low"], errors="coerce"); hi = pd.to_numeric(d["OR_CI_high"], errors="coerce")
        d["se"] = (np.log(hi)-np.log(lo))/(2*1.96)
    d["coef"] = pd.to_numeric(d["coef"], errors="coerce"); d["se"] = pd.to_numeric(d["se"], errors="coerce")
    return d[np.isfinite(d["coef"]) & np.isfinite(d["se"]) & (d["se"] > 0)].copy()


def dl_meta(dat: pd.DataFrame) -> dict:
    d = _clean(dat)
    if d.empty:
        return {"method":"DL","k":0,"coef":np.nan,"se":np.nan,"p_value":np.nan,"OR":np.nan,"OR_CI_low":np.nan,"OR_CI_high":np.nan,"tau2":np.nan,"Q":np.nan,"Q_p":np.nan,"I2":np.nan}
    y=d["coef"].to_numpy(float); v=d["se"].to_numpy(float)**2; k=len(y); wf=1/v; fixed=np.sum(wf*y)/np.sum(wf)
    if k>1:
        q=float(np.sum(wf*(y-fixed)**2)); df=k-1; c=float(np.sum(wf)-np.sum(wf**2)/np.sum(wf)); tau=max(0.,(q-df)/c) if c>0 else 0.; qp=float(chi2.sf(q,df)); i2=max(0.,(q-df)/q)*100 if q>0 else 0.
    else:
        q=0.; df=0; tau=0.; qp=np.nan; i2=0.
    w=1/(v+tau); mu=float(np.sum(w*y)/np.sum(w)); se=float(math.sqrt(1/np.sum(w))); p=normal_p(mu/se if se>0 else np.nan)
    return {"method":"DL","k":k,"coef":mu,"se":se,"p_value":p,"OR":math.exp(mu),"OR_CI_low":math.exp(mu-1.96*se),"OR_CI_high":math.exp(mu+1.96*se),"tau2":tau,"Q":q,"Q_p":qp,"I2":i2}


def _reml_nll(tau2: float, y: np.ndarray, v: np.ndarray) -> float:
    w=1/(v+tau2); sw=np.sum(w)
    if not np.isfinite(sw) or sw<=0: return np.inf
    mu=np.sum(w*y)/sw
    return 0.5*(np.sum(np.log(v+tau2))+math.log(sw)+np.sum(w*(y-mu)**2))


def reml_hk_meta(dat: pd.DataFrame) -> dict:
    d=_clean(dat)
    if d.empty:
        return {"method":"REML_HK","k":0,"coef":np.nan,"se":np.nan,"p_value":np.nan,"OR":np.nan,"OR_CI_low":np.nan,"OR_CI_high":np.nan,"tau2":np.nan,"Q":np.nan,"Q_p":np.nan,"I2":np.nan,"PI_low":np.nan,"PI_high":np.nan}
    y=d["coef"].to_numpy(float); v=d["se"].to_numpy(float)**2; k=len(y); wf=1/v; fixed=np.sum(wf*y)/np.sum(wf)
    q=float(np.sum(wf*(y-fixed)**2)) if k>1 else 0.; qdf=k-1; qp=float(chi2.sf(q,qdf)) if qdf>0 else np.nan; i2=max(0.,(q-qdf)/q)*100 if q>0 and qdf>0 else 0.
    if k>1:
        upper=max(1.0, float(np.var(y,ddof=1))*10 + float(np.max(v))*10)
        opt=minimize_scalar(_reml_nll,bounds=(0.,upper),args=(y,v),method="bounded",options={"xatol":1e-12})
        tau=max(0.,float(opt.x)) if opt.success else 0.
    else: tau=0.
    w=1/(v+tau); mu=float(np.sum(w*y)/np.sum(w)); se_naive=float(math.sqrt(1/np.sum(w)))
    if k>1:
        qstar=float(np.sum(w*(y-mu)**2)); scale=qstar/(k-1); se_hk=math.sqrt(max(scale,0.)/np.sum(w))
        if not np.isfinite(se_hk) or se_hk<=0: se_hk=se_naive
        crit=float(t.ppf(.975,df=k-1)); lo=mu-crit*se_hk; hi=mu+crit*se_hk; p=2*float(t.sf(abs(mu/se_hk),df=k-1)) if se_hk>0 else np.nan
    else:
        se_hk=se_naive; lo=mu-1.96*se_hk; hi=mu+1.96*se_hk; p=normal_p(mu/se_hk if se_hk>0 else np.nan)
    if k>=3:
        critpi=float(t.ppf(.975,df=k-2)); predse=math.sqrt(tau+se_hk**2); pil=mu-critpi*predse; pih=mu+critpi*predse
    elif k==2:
        predse=math.sqrt(tau+se_hk**2); pil=mu-1.96*predse; pih=mu+1.96*predse
    else: pil=pih=np.nan
    return {"method":"REML_HK","k":k,"coef":mu,"se":se_hk,"p_value":p,"OR":math.exp(mu),"OR_CI_low":math.exp(lo),"OR_CI_high":math.exp(hi),"tau2":tau,"Q":q,"Q_p":qp,"I2":i2,"PI_low":math.exp(pil) if np.isfinite(pil) else np.nan,"PI_high":math.exp(pih) if np.isfinite(pih) else np.nan}


def pool_grouped(df: pd.DataFrame, group_cols: list[str], method: str="DL") -> pd.DataFrame:
    rows=[]; fn=dl_meta if method.upper()=="DL" else reml_hk_meta
    for keys,g in df.groupby(group_cols,dropna=False):
        if not isinstance(keys,tuple): keys=(keys,)
        row={c:k for c,k in zip(group_cols,keys)}; row.update(fn(g)); row["cohorts_used"]=",".join(g["cohort"].astype(str).tolist()) if "cohort" in g else ""; rows.append(row)
    return pd.DataFrame(rows)
