#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mood-focused/non-somatic outcome sensitivity using the final 8-shock primary exposure.

The outcome is a cohort-specific affective-core score built only from selected
item-level CES-D/EURO-D items and standardized within the eligible analysis sample.
Primary reviewer target: IndShock_main, Model 3 covariates, random-intercept linear model.
"""
from __future__ import annotations
import argparse, math, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
CODE_ROOT=Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path: sys.path.insert(0,str(CODE_ROOT))
from shared.cohort_loader import load_primary8_module
from shared.io_helpers import read_raw_for_module
from shared.project_config import COHORT_ORDER, DEFAULT_PROJECT_ROOT, default_data_path, MOOD_ITEM_SPECS, SUMMARY_TERMS
from shared.meta_utils import normal_p
from shared.data_integrity import checked_person_wave_merge, country_terms


def resolve_col(existing, pattern, w):
    lower={str(c).lower():str(c) for c in existing}; key=pattern.format(w=w).lower(); return lower.get(key)

def read_selected(mod,path,cols):
    if hasattr(mod,"read_data_file"): return mod.read_data_file(path,columns=cols)
    if path.suffix.lower()==".dta": return pd.read_stata(path,columns=cols,convert_categoricals=False)
    raise ValueError(path)

def score_item(s, coding, reverse):
    x=pd.to_numeric(s,errors="coerce"); out=pd.Series(np.nan,index=s.index,dtype=float)
    if coding=="charls_1to4":
        ok=x.isin([1,2,3,4]); out.loc[ok]=(4-x.loc[ok]) if reverse else (x.loc[ok]-1)
    else:
        ok=x.isin([0,1]); out.loc[ok]=x.loc[ok]
        if reverse: out.loc[ok]=1-out.loc[ok]
    return out

def build_mood_long(mod,cohort,path):
    spec=MOOD_ITEM_SPECS[cohort]; existing=mod.existing_columns(path); rawmap=mod.build_raw_column_map(path); idcol=rawmap["id"]
    selected=[idcol]; resolved={}
    for w in mod.ALL_WAVES:
        resolved[w]={}
        for name,(pat,rev) in spec["items"].items():
            c=resolve_col(existing,pat,w); resolved[w][name]=(c,rev)
            if c: selected.append(c)
    selected=list(dict.fromkeys(selected)); raw=read_selected(mod,path,selected); pid=raw[idcol]
    frames=[]
    for w in mod.ALL_WAVES:
        p=pd.DataFrame({"id":pid.values,"wave":int(w)}); keep=[]
        for name,(c,rev) in resolved[w].items():
            s=raw[c] if c else pd.Series(np.nan,index=raw.index); scored=score_item(s,spec["coding"],rev if spec["coding"]!="binary_depressive_indicator" else False); col=f"mood_{name}"; p[col]=scored.values
            if name in spec["selected"]: keep.append(col)
        mat=p[keep]; ok=mat.notna().all(axis=1); p["MoodCore_score"]=np.nan; p.loc[ok,"MoodCore_score"]=mat.loc[ok].sum(axis=1); p["MoodCore_n_items"]=len(keep); frames.append(p)
    return pd.concat(frames,ignore_index=True)

def fit_lmer_once(d,exposure,maxfun):
    with tempfile.TemporaryDirectory(prefix="mood_lmer_") as td:
        td=Path(td); dat=td/"d.csv"; out=td/"o.csv"; r=td/"fit.R"; d.to_csv(dat,index=False)
        formula=f"MoodCore_z ~ {exposure} + age + male + factor(education) + smoking + drinking + factor(wave)" + country_terms(d) + " + (1|id)"
        r.write_text(f'''suppressPackageStartupMessages(library(lme4))\nd<-read.csv("{dat.as_posix()}")\nd$id<-factor(d$id); d$wave<-factor(d$wave); d$education<-factor(d$education); d$age<-d$age/10\nfit<-lmer({formula!r},data=d,control=lmerControl(optimizer="bobyqa",optCtrl=list(maxfun={maxfun}),calc.derivs=FALSE))\nco<-as.data.frame(coef(summary(fit))); co$term<-rownames(co); rownames(co)<-NULL; write.csv(co,"{out.as_posix()}",row.names=FALSE)\n''',encoding="utf-8")
        subprocess.run(["Rscript",str(r)],check=True,capture_output=True,text=True); z=pd.read_csv(out); row=z[z["term"].astype(str)==exposure].iloc[0]; return float(row["Estimate"]),float(row["Std. Error"])

def pool_rubin(est):
    e=pd.DataFrame(est,columns=["coef","se"]).dropna(); m=len(e)
    if m==0:return {"m":0,"coef":np.nan,"se":np.nan,"p_value":np.nan,"CI_low":np.nan,"CI_high":np.nan}
    q=e.coef.to_numpy(); s=e.se.to_numpy(); qbar=float(q.mean()); within=float(np.mean(s*s)); between=float(np.var(q,ddof=1)) if m>1 else 0.; total=within+(1+1/m)*between; se=math.sqrt(max(total,0)); p=normal_p(qbar/se if se>0 else np.nan); return {"m":m,"coef":qbar,"se":se,"p_value":p,"CI_low":qbar-1.96*se,"CI_high":qbar+1.96*se}

def pool_beta(df):
    rows=[]
    for effect,g in df.groupby("effect_name"):
        y=pd.to_numeric(g.coef,errors="coerce").to_numpy(float); se=pd.to_numeric(g.se,errors="coerce").to_numpy(float); ok=np.isfinite(y)&np.isfinite(se)&(se>0); y=y[ok]; se=se[ok]; k=len(y)
        if not k: continue
        v=se**2; w=1/v; mu0=np.sum(w*y)/np.sum(w); q=float(np.sum(w*(y-mu0)**2)); tau=max(0.,(q-(k-1))/(np.sum(w)-np.sum(w*w)/np.sum(w))) if k>1 else 0.; wr=1/(v+tau); mu=float(np.sum(wr*y)/np.sum(wr)); pse=float(math.sqrt(1/np.sum(wr))); rows.append({"analysis":"mood_non_somatic","effect_name":effect,"method":"DL","k":k,"coef":mu,"se":pse,"CI_low":mu-1.96*pse,"CI_high":mu+1.96*pse,"p_value":normal_p(mu/pse),"tau2":tau,"I2":max(0.,(q-(k-1))/q)*100 if q>0 and k>1 else 0.})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cohort",choices=COHORT_ORDER); ap.add_argument("--input",default=""); ap.add_argument("--outdir",default=str(DEFAULT_PROJECT_ROOT/"output"/"07_sensitivity_mood_only")); ap.add_argument("--pool",action="store_true"); ap.add_argument("--all-domains",action="store_true"); ap.add_argument("--m",type=int,default=10); ap.add_argument("--seed",type=int,default=20260971); ap.add_argument("--mi-max-iter",type=int,default=8); ap.add_argument("--mi-n-nearest-features",type=int,default=4); ap.add_argument("--min-baseline-age",type=float,default=50.0); ap.add_argument("--maxfun",type=int,default=200000)
    args=ap.parse_args(); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    if args.pool:
        frames=[pd.read_csv(out/f"{c}_model3.csv") for c in COHORT_ORDER]; allr=pd.concat(frames,ignore_index=True); pooled=pool_beta(allr); pooled.to_csv(out/"pooled_model3.csv",index=False); print(pooled.to_string(index=False)); return
    if not args.cohort: ap.error("--cohort required unless --pool")
    cohort=args.cohort; path=Path(args.input) if args.input else default_data_path(cohort); mod=load_primary8_module(cohort); raw,rawmap=read_raw_for_module(mod,path); long=mod.build_long_panel(raw,rawmap=rawmap); mood=build_mood_long(mod,cohort,path); d,_=checked_person_wave_merge(long,mood,"mood_items")
    elig=d["wave"].isin(mod.ANALYSIS_WAVES)&pd.to_numeric(d["baseline_age"],errors="coerce").ge(args.min_baseline_age)&pd.to_numeric(d["valid_adjacent_interview_pair"],errors="coerce").eq(1)&d["MoodCore_score"].notna()&d[mod.OUTCOME].notna(); mu=pd.to_numeric(d.loc[elig,"MoodCore_score"],errors="coerce").mean(); sd=pd.to_numeric(d.loc[elig,"MoodCore_score"],errors="coerce").std(ddof=1); d["MoodCore_z"]=np.nan; d.loc[elig,"MoodCore_z"]=(pd.to_numeric(d.loc[elig,"MoodCore_score"],errors="coerce")-mu)/sd
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(f"{cohort}: no variable, complete mood score in the eligible sample")
    anal=d.loc[elig].copy(); timer=mod.StepTimer(); imps=mod.impute_covariates(anal,m=args.m,seed=args.seed,mi_max_iter=args.mi_max_iter,mi_n_nearest_features=args.mi_n_nearest_features,timer=timer); exposures=list(SUMMARY_TERMS) if args.all_domains else ["IndShock_main"]; rows=[]
    for exposure in exposures:
        est=[]
        for imp in imps:
            dd=imp.dropna(subset=["MoodCore_z",exposure,"age","male","education","smoking","drinking","wave","id"]).copy()
            if len(dd)<50 or dd[exposure].nunique()<2: continue
            try: est.append(fit_lmer_once(dd,exposure,args.maxfun))
            except Exception as exc: raise RuntimeError(f"{cohort} {exposure}: mood model failed") from exc
        if len(est) != args.m: raise RuntimeError(f"{cohort} {exposure}: only {len(est)}/{args.m} usable mood fits")
        p=pool_rubin(est); rows.append({"cohort":cohort,"analysis":"mood_non_somatic","result_type":"domain","effect_name":exposure,"effect_scale":"standardized_beta","n_rows":len(anal),"n_persons":anal.id.nunique(),**p})
    pd.DataFrame(rows).to_csv(out/f"{cohort}_model3.csv",index=False); print(f"[DONE] {cohort}")
if __name__=="__main__": main()
