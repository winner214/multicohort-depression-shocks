#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reviewer temporal sensitivities using the final 8-shock exposure definition.
1) lagged: shock at t -> depression risk at immediately following interview;
2) incident: same, restricted to depression-free respondents at t.
Model 3 only; covariates are multiply imputed, shocks/outcomes are not.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
CODE_ROOT=Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path: sys.path.insert(0,str(CODE_ROOT))
from shared.cohort_loader import load_primary8_module
from shared.io_helpers import read_raw_for_module
from shared.project_config import COHORT_ORDER, DEFAULT_PROJECT_ROOT, default_data_path, ALL_COMPONENTS_8, SUMMARY_TERMS
from shared.sensitivity_core import fit_model3,save_compact_results,pool_compact_directory


def make_frame(mod,long_df,kind,min_age):
    d=long_df.sort_values(["id","wave"],kind="mergesort").copy()
    d["next_wave"]=d.groupby("id",sort=False)["wave"].shift(-1)
    d["next_inw"]=d.groupby("id",sort=False)["inw"].shift(-1)
    d["next_depression"]=d.groupby("id",sort=False)[mod.OUTCOME].shift(-1)
    d=d[d["wave"].isin(mod.ANALYSIS_WAVES) & pd.to_numeric(d["baseline_age"],errors="coerce").ge(min_age)].copy()
    d=d[pd.to_numeric(d["valid_adjacent_interview_pair"],errors="coerce").eq(1)].copy()
    d=d[(pd.to_numeric(d["next_wave"],errors="coerce")==pd.to_numeric(d["wave"],errors="coerce")+1) & pd.to_numeric(d["next_inw"],errors="coerce").eq(1) & d["next_depression"].notna()].copy()
    if kind=="incident":
        d=d[pd.to_numeric(d[mod.OUTCOME],errors="coerce").eq(0)].copy()
    d[mod.OUTCOME]=pd.to_numeric(d["next_depression"],errors="coerce").astype("float32")
    return d.reset_index(drop=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cohort",choices=COHORT_ORDER); ap.add_argument("--input",default=""); ap.add_argument("--outdir",default=str(DEFAULT_PROJECT_ROOT/"output"/"06_sensitivity_lagged_incident")); ap.add_argument("--pool",action="store_true")
    ap.add_argument("--m",type=int,default=10); ap.add_argument("--seed",type=int,default=20260961); ap.add_argument("--mi-max-iter",type=int,default=8); ap.add_argument("--mi-n-nearest-features",type=int,default=4); ap.add_argument("--min-baseline-age",type=float,default=50.0); ap.add_argument("--optimizer-seq",default="bobyqa,Nelder_Mead,nloptwrap"); ap.add_argument("--maxfun",type=int,default=200000); ap.add_argument("--singular-tol",type=float,default=1e-4); ap.add_argument("--domains-only",action="store_true",default=False)
    args=ap.parse_args(); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    if args.pool:
        pooled=pool_compact_directory(out,out/"pooled_model3.csv",pattern="*_model3.csv"); print(pooled.to_string(index=False)); return
    if not args.cohort: ap.error("--cohort required unless --pool")
    cohort=args.cohort; inp=Path(args.input) if args.input else default_data_path(cohort); mod=load_primary8_module(cohort); raw,rawmap=read_raw_for_module(mod,inp); long_df=mod.build_long_panel(raw,rawmap=rawmap)
    opts=[x.strip() for x in args.optimizer_seq.split(',') if x.strip()]; rows=[]
    exposures=list(SUMMARY_TERMS) if args.domains_only else list(SUMMARY_TERMS)+list(ALL_COMPONENTS_8)
    for j,kind in enumerate(["lagged","incident"]):
        d=make_frame(mod,long_df,kind,args.min_baseline_age)
        timer=mod.StepTimer(); imps=mod.impute_covariates(d,m=args.m,seed=args.seed+1000*j,mi_max_iter=args.mi_max_iter,mi_n_nearest_features=args.mi_n_nearest_features,timer=timer)
        res=fit_model3(mod,imps,cohort,kind,exposures,opts,args.maxfun,args.singular_tol); rows.append(res)
    final=pd.concat(rows,ignore_index=True,sort=False); save_compact_results(final,out/f"{cohort}_model3.csv"); print(f"[DONE] {cohort}")
if __name__=="__main__": main()
