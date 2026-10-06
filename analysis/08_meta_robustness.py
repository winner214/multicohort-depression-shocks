#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Alternative meta-analysis for primary Model 3 domains: DL, REML+Hartung-Knapp,
prediction intervals, leave-one-cohort-out, and explicit leave-SHARE-out.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
CODE_ROOT=Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path: sys.path.insert(0,str(CODE_ROOT))
from shared.project_config import COHORT_ORDER, DEFAULT_PROJECT_ROOT, SUMMARY_TERMS
from shared.meta_utils import dl_meta,reml_hk_meta


def find_primary(root,cohort):
    p=root/cohort/f"{cohort}_03_main_summary_model123.csv"
    if p.exists(): return p
    hits=list(root.rglob(f"{cohort}_03_main_summary_model123.csv")); return hits[0] if hits else None

def load_primary(root):
    frames=[]
    for c in COHORT_ORDER:
        fp=find_primary(root,c)
        if fp is None: raise FileNotFoundError(f"Missing primary cohort result: {c}")
        d=pd.read_csv(fp); d=d[(d["model"].astype(str)=="Model 3") & d["exposure"].astype(str).isin(SUMMARY_TERMS)].copy(); d["cohort"]=c; frames.append(d)
    if not frames: raise FileNotFoundError(root)
    return pd.concat(frames,ignore_index=True,sort=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--primary-dir",default=str(DEFAULT_PROJECT_ROOT/"output"/"01_primary_8shock")); ap.add_argument("--outdir",default=str(DEFAULT_PROJECT_ROOT/"output"/"08_meta_robustness")); args=ap.parse_args(); root=Path(args.primary_dir); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    d=load_primary(root); rows=[]; loo=[]
    for effect,g in d.groupby("exposure"):
        for fn in [dl_meta,reml_hk_meta]:
            p=fn(g); rows.append({"effect_name":effect,"cohorts_used":",".join(g.cohort.astype(str)),**p})
        for omit in COHORT_ORDER:
            gg=g[g.cohort!=omit]
            if len(gg)<2: continue
            p=reml_hk_meta(gg); loo.append({"effect_name":effect,"omitted_cohort":omit,"cohorts_used":",".join(gg.cohort.astype(str)),**p})
    pd.DataFrame(rows).to_csv(out/"primary_model3_DL_REML_HK.csv",index=False); pd.DataFrame(loo).to_csv(out/"primary_model3_leave_one_out_REML_HK.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=="__main__": main()
