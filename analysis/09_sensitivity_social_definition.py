#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small reviewer-targeted social-shock definition checks.
A) Remove unemployment entirely from the social domain (7-shock domain sensitivity).
B) Split marital disruption into widowhood and divorce/separation components where supported.
Model 3 only; primary covariate-MI strategy.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
CODE_ROOT=Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path: sys.path.insert(0,str(CODE_ROOT))
from shared.cohort_loader import load_primary8_module
from shared.io_helpers import build_long_and_primary
from shared.project_config import COHORT_ORDER, DEFAULT_PROJECT_ROOT, default_data_path, MARITAL_STATUS_SPECS, IND_COMPONENTS_8
from shared.primary_overlay import _domain_from_components
from shared.sensitivity_core import save_compact_results,pool_compact_directory


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cohort",choices=COHORT_ORDER); ap.add_argument("--input",default=""); ap.add_argument("--outdir",default=str(DEFAULT_PROJECT_ROOT/"output"/"10_sensitivity_social_definition")); ap.add_argument("--pool",action="store_true"); ap.add_argument("--m",type=int,default=10); ap.add_argument("--seed",type=int,default=20261001); ap.add_argument("--mi-max-iter",type=int,default=8); ap.add_argument("--mi-n-nearest-features",type=int,default=4); ap.add_argument("--min-baseline-age",type=float,default=50.0); ap.add_argument("--optimizer-seq",default="bobyqa,Nelder_Mead,nloptwrap"); ap.add_argument("--maxfun",type=int,default=200000); ap.add_argument("--singular-tol",type=float,default=1e-4); ap.add_argument("--min-events",type=int,default=20)
    args=ap.parse_args(); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    if args.pool:
        p=pool_compact_directory(out,out/"pooled_model3.csv",pattern="*_model3.csv"); print(p.to_string(index=False)); return
    if not args.cohort: ap.error("--cohort required unless --pool")
    cohort=args.cohort; path=Path(args.input) if args.input else default_data_path(cohort); mod=load_primary8_module(cohort); _,d=build_long_and_primary(mod,path,args.min_baseline_age); ms=MARITAL_STATUS_SPECS[cohort]
    # No-unemployment domain.
    d["SocShock_no_unemployment"]=_domain_from_components(d,["marital_shock","child_death","living_alone_transition"])
    h=pd.to_numeric(d["IndShock_main"],errors="coerce"); s=pd.to_numeric(d["SocShock_no_unemployment"],errors="coerce"); ov=pd.Series(np.nan,index=d.index,dtype="float32"); ov.loc[h.eq(1)|s.eq(1)]=1.; ov.loc[h.eq(0)&s.eq(0)]=0.; d["OverallShock_no_unemployment"]=ov
    # Split marital events in the exact primary marital risk set.
    prev=pd.to_numeric(d["prev_marital_status_strict_raw"],errors="coerce"); curr=pd.to_numeric(d["marital_status_strict_raw"],errors="coerce"); risk=pd.to_numeric(d["risk_marital_shock"],errors="coerce").eq(1); d["widowhood_event"]=np.nan; d["divorce_separation_event"]=np.nan; d.loc[risk,"widowhood_event"]=curr.loc[risk].isin(ms["widow_codes"]).astype(float); d.loc[risk,"divorce_separation_event"]=curr.loc[risk].isin(ms["divsep_codes"]).astype(float)
    timer=mod.StepTimer(); imps=mod.impute_covariates(d,m=args.m,seed=args.seed,mi_max_iter=args.mi_max_iter,mi_n_nearest_features=args.mi_n_nearest_features,timer=timer); opts=[x.strip() for x in args.optimizer_seq.split(',') if x.strip()]; mod.SUMMARY_TERMS=list(mod.SUMMARY_TERMS)+["SocShock_no_unemployment","OverallShock_no_unemployment"]; mod.COMPONENT_TERMS=list(mod.COMPONENT_TERMS)+["widowhood_event","divorce_separation_event"]; rows=[]
    for exposure,rtype in [("SocShock_no_unemployment","domain"),("OverallShock_no_unemployment","domain")]:
        res,_=mod.fit_one_exposure_all_imputations(imps,exposure,"Model 3",timer,optimizer_seq=opts,maxfun=args.maxfun,singular_tol=args.singular_tol,label_prefix="socialdef_"); r=res.copy(); r["cohort"]=cohort; r["analysis"]="exclude_unemployment"; r["result_type"]=rtype; r["effect_name"]=exposure; rows.append(r)
    for exposure in ["widowhood_event","divorce_separation_event"]:
        events=int(pd.to_numeric(d[exposure],errors="coerce").eq(1).sum())
        if events<args.min_events:
            rows.append(pd.DataFrame([{"cohort":cohort,"analysis":"split_marital","result_type":"component","effect_name":exposure,"model":"Model 3","m":0,"coef":np.nan,"se":np.nan,"p_value":np.nan,"OR":np.nan,"OR_CI_low":np.nan,"OR_CI_high":np.nan}]))
            continue
        # Explicitly restrict to the primary marital risk set before fitting.
        sub=[x[pd.to_numeric(x["risk_marital_shock"],errors="coerce").eq(1)].copy() for x in imps]
        res,_=mod.fit_one_exposure_all_imputations(sub,exposure,"Model 3",timer,optimizer_seq=opts,maxfun=args.maxfun,singular_tol=args.singular_tol,label_prefix="maritalsplit_"); r=res.copy(); r["cohort"]=cohort; r["analysis"]="split_marital"; r["result_type"]="component"; r["effect_name"]=exposure; rows.append(r)
    final=pd.concat(rows,ignore_index=True,sort=False); save_compact_results(final,out/f"{cohort}_model3.csv"); print(f"[DONE] {cohort}")
if __name__=="__main__": main()
