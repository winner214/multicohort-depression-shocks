#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sensitivity 4: historical uploaded 10-shock specification, Model 3 only.

This intentionally loads the historical cohort specification (revision country support is disabled). It is a
historical-specification comparison, not a single-factor missingness experiment.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
CODE_ROOT=Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path: sys.path.insert(0,str(CODE_ROOT))
from shared.cohort_loader import load_legacy_module
from shared.io_helpers import read_raw_for_module
from shared.project_config import COHORT_ORDER, DEFAULT_PROJECT_ROOT, default_data_path
from shared.sensitivity_core import save_compact_results,pool_compact_directory


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cohort",choices=COHORT_ORDER); ap.add_argument("--input",default=""); ap.add_argument("--outdir",default=str(DEFAULT_PROJECT_ROOT/"output"/"05_sensitivity_original_10shock")); ap.add_argument("--pool",action="store_true")
    ap.add_argument("--m",type=int,default=10,help="Retain the historical 10-imputation specification"); ap.add_argument("--seed",type=int,default=20260423); ap.add_argument("--mi-max-iter",type=int,default=8); ap.add_argument("--mi-n-nearest-features",type=int,default=4); ap.add_argument("--min-baseline-age",type=float,default=50.0); ap.add_argument("--optimizer-seq",default="bobyqa,Nelder_Mead,nloptwrap"); ap.add_argument("--maxfun",type=int,default=200000); ap.add_argument("--singular-tol",type=float,default=1e-4)
    args=ap.parse_args(); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    if args.pool:
        pooled=pool_compact_directory(out,out/"pooled_model3.csv",pattern="*_model3.csv"); print(pooled.to_string(index=False)); return
    if not args.cohort: ap.error("--cohort required unless --pool")
    cohort=args.cohort; inp=Path(args.input) if args.input else default_data_path(cohort); mod=load_legacy_module(cohort)
    raw,rawmap=read_raw_for_module(mod,inp); long_df=mod.build_long_panel(raw,rawmap=rawmap); ana=mod.prepare_main_dataset(long_df,min_baseline_age=args.min_baseline_age); ana=mod.add_shock_count_variables(ana)
    timer=mod.StepTimer(); imps=mod.impute_covariates(ana,m=args.m,seed=args.seed,mi_max_iter=args.mi_max_iter,mi_n_nearest_features=args.mi_n_nearest_features,timer=timer); opts=[x.strip() for x in args.optimizer_seq.split(',') if x.strip()]
    rows=[]
    for exposure in list(mod.SUMMARY_TERMS)+list(mod.COMPONENT_TERMS):
        res,_=mod.fit_one_exposure_all_imputations(imps,exposure,"Model 3",timer,optimizer_seq=opts,maxfun=args.maxfun,singular_tol=args.singular_tol,label_prefix="legacy10_")
        if len(res):
            r=res.copy(); r["cohort"]=cohort; r["analysis"]="original_10shock"; r["result_type"]="domain" if exposure in mod.SUMMARY_TERMS else "component"; r["effect_name"]=exposure; rows.append(r)
    import pandas as pd
    final=pd.concat(rows,ignore_index=True,sort=False) if rows else pd.DataFrame(); save_compact_results(final,out/f"{cohort}_model3.csv"); print(f"[DONE] {cohort}")
if __name__=="__main__": main()
