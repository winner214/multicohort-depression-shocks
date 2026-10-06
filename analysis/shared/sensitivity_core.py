#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import gc
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

from .project_config import COHORT_ORDER, ALL_COMPONENTS_8, IND_COMPONENTS_8, SOC_COMPONENTS_8, MODEL3_COVARIATES
from .primary_overlay import recompute_primary_domains, domain_with_optional_spouse
from .meta_utils import pool_grouped


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def direct_mi(base, target_columns, m, seed, max_iter=10, structural_masks=None):
    """Derived-indicator MI used ONLY for domain Model 3 in 02--04.

    No component effect/risk-set inference is made from these imputed indicators.
    Structurally inapplicable (or unknown-applicability) spouse cells are never
    targets: the spouse imputation is fitted on applicable observations only.
    """
    from .enhanced_mi import impute_targets
    structural_masks = structural_masks or {}
    ordinary = [c for c in target_columns if c not in structural_masks]
    imps, _ = impute_targets(base, ordinary, m, seed, max_iter)
    for col, blocked in structural_masks.items():
        blocked = pd.Series(blocked, index=base.index).fillna(True).astype(bool)
        eligible = ~blocked
        for j, d in enumerate(imps):
            d.loc[blocked, col] = np.nan
            if not eligible.any() or not d.loc[eligible, col].isna().any():
                continue
            # Draw one stochastic completion with a disjoint seed.
            subset = d.loc[eligible].copy()
            draws, _ = impute_targets(subset, [col], 1,
                                      seed+100000+j*2, max_iter,
                                      extra_aux=ordinary, posterior=m>1)
            d.loc[eligible, col] = draws[0][col]
            del draws
    return imps


def recompute_domains_for_spec(d: pd.DataFrame, spec: str) -> pd.DataFrame:
    spec=spec.lower(); out=d.copy()
    if spec=="8":
        return recompute_primary_domains(out)
    if spec=="income9":
        out["IndShock_main"] = _conservative_domain(out, IND_COMPONENTS_8)
        out["SocShock_main"] = _conservative_domain(out, SOC_COMPONENTS_8+["income_decline_shock"])
    elif spec=="spouse9":
        out["IndShock_main"] = _conservative_domain(out, IND_COMPONENTS_8)
        out["SocShock_main"] = domain_with_optional_spouse(out)
    else:
        raise ValueError(spec)
    h=_num(out["IndShock_main"]); s=_num(out["SocShock_main"])
    overall=pd.Series(np.nan,index=out.index,dtype="float32")
    overall.loc[h.eq(1)|s.eq(1)]=1.; overall.loc[h.eq(0)&s.eq(0)]=0.
    out["OverallShock_main"]=overall
    return out


def _conservative_domain(d: pd.DataFrame, comps: list[str]) -> pd.Series:
    x=d[comps].apply(pd.to_numeric,errors="coerce"); any1=x.eq(1).any(axis=1); allok=x.notna().all(axis=1)
    out=pd.Series(np.nan,index=d.index,dtype="float32"); out.loc[any1]=1.; out.loc[(~any1)&allok]=0.; return out


def fit_model3(mod, imputed_list: list[pd.DataFrame], cohort: str, analysis: str, exposures: Iterable[str], optimizer_seq: list[str], maxfun: int, singular_tol: float) -> pd.DataFrame:
    rows=[]; timer=mod.StepTimer()
    for exposure in exposures:
        res,_=mod.fit_one_exposure_all_imputations(imputed_list, exposure, "Model 3", timer, optimizer_seq=optimizer_seq, maxfun=maxfun, singular_tol=singular_tol, label_prefix=f"{analysis}_")
        if res is None or len(res)==0: continue
        r=res.copy(); r["cohort"]=cohort; r["analysis"]=analysis; r["result_type"]="domain" if exposure in ["OverallShock_main","IndShock_main","SocShock_main"] else "component"; r["effect_name"]=exposure
        rows.append(r)
    return pd.concat(rows,ignore_index=True,sort=False) if rows else pd.DataFrame()


def save_compact_results(df: pd.DataFrame, path: Path) -> None:
    cols=[c for c in ["cohort","analysis","result_type","effect_name","model","m","coef","se","p_value","OR","OR_CI_low","OR_CI_high","formula"] if c in df.columns]
    path.parent.mkdir(parents=True,exist_ok=True)
    df[cols].to_csv(path,index=False)


def pool_compact_directory(input_dir: Path, out_path: Path, pattern: str="*_model3.csv", method: str="DL") -> pd.DataFrame:
    frames=[]
    expected = [input_dir/f"{c}_model3.csv" for c in COHORT_ORDER]
    missing = [p.name for p in expected if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Five-cohort pooling requires all cohort files; missing: {missing}")
    for fp in expected:
        try:
            d=pd.read_csv(fp)
            if "cohort" in d: frames.append(d)
        except Exception as exc:
            raise RuntimeError(f"Cannot read cohort result {fp}") from exc
    if not frames: raise FileNotFoundError(f"No {pattern} under {input_dir}")
    allr=pd.concat(frames,ignore_index=True,sort=False)
    groups=[c for c in ["analysis","result_type","effect_name"] if c in allr.columns]
    pooled=pool_grouped(allr,groups,method=method)
    out_path.parent.mkdir(parents=True,exist_ok=True); pooled.to_csv(out_path,index=False)
    return pooled
