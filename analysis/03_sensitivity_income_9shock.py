#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sensitivity 2: eight core shocks + income decline; shock + covariate MI; Model 3 only."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
CODE_ROOT=Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path: sys.path.insert(0,str(CODE_ROOT))
from shared.cohort_loader import load_primary8_module
from shared.io_helpers import build_long_and_primary
from shared.project_config import COHORT_ORDER, DEFAULT_PROJECT_ROOT, default_data_path, ALL_COMPONENTS_8, MODEL3_COVARIATES, SUMMARY_TERMS
from shared.sensitivity_core import direct_mi,recompute_domains_for_spec,fit_model3,save_compact_results,pool_compact_directory

EXTRA="income_decline_shock"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cohort",choices=COHORT_ORDER); ap.add_argument("--input",default=""); ap.add_argument("--outdir",default=str(DEFAULT_PROJECT_ROOT/"output"/"03_sensitivity_income_9shock")); ap.add_argument("--pool",action="store_true")
    ap.add_argument("--m",type=int,default=10); ap.add_argument("--seed",type=int,default=20260931); ap.add_argument("--mi-max-iter",type=int,default=10); ap.add_argument("--min-baseline-age",type=float,default=50.0); ap.add_argument("--optimizer-seq",default="bobyqa,Nelder_Mead,nloptwrap"); ap.add_argument("--maxfun",type=int,default=200000); ap.add_argument("--singular-tol",type=float,default=1e-4)
    args=ap.parse_args(); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    if args.pool:
        pooled=pool_compact_directory(out,out/"pooled_model3.csv",pattern="*_model3.csv"); print(pooled.to_string(index=False)); return
    if not args.cohort: ap.error("--cohort is required unless --pool")
    cohort=args.cohort; inp=Path(args.input) if args.input else default_data_path(cohort); mod=load_primary8_module(cohort); _,primary=build_long_and_primary(mod,inp,args.min_baseline_age)
    if EXTRA not in primary: raise KeyError(f"{cohort}: {EXTRA} unavailable")
    imps=direct_mi(primary,list(ALL_COMPONENTS_8)+[EXTRA]+MODEL3_COVARIATES,args.m,args.seed,args.mi_max_iter)
    imps=[recompute_domains_for_spec(d,"income9") for d in imps]
    res=fit_model3(mod,imps,cohort,"income_9shock",SUMMARY_TERMS,[x.strip() for x in args.optimizer_seq.split(',') if x.strip()],args.maxfun,args.singular_tol)
    save_compact_results(res,out/f"{cohort}_model3.csv"); print(f"[DONE] {cohort}")
if __name__=="__main__": main()
