#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sensitivity 3: eight core shocks + spouse cancer; Model 3.

Modes
-----
1) Default cohort mode
   Re-runs the existing three domain Model-3 analyses (Overall/Health/Social).

2) --component-only
   Runs ONLY the nine component Model-3 analyses using the same component-specific
   risk-set + exposure/covariate MI strategy used by the completed 03 income-9-shock
   analysis. This mode is intended for CHARLS, MHAS, HRS, and ELSA only. SHARE is
   deliberately skipped because its 04 component results already exist.

3) --pool
   Pools the five cohort result files after the four missing component runs have
   been appended to CHARLS/MHAS/HRS/ELSA. SHARE is read from its existing file.

Important component rules
-------------------------
* Component models use event-specific risk sets based on PREVIOUS-state eligibility.
* Current-wave exposure missingness is allowed inside an eligible risk set and is
  multiply imputed together with Model-3 covariates.
* Structural spouse-cancer inapplicability is excluded, never imputed as zero.
* Existing domain rows are preserved when --component-only is used.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from shared.cohort_loader import load_primary8_module
from shared.enhanced_mi import impute_targets
from shared.io_helpers import build_long_and_primary
from shared.project_config import (
    ALL_COMPONENTS_8,
    COHORT_ORDER,
    DEFAULT_PROJECT_ROOT,
    LABOR_STATUS_SPECS,
    MODEL3_COVARIATES,
    SUMMARY_TERMS,
    default_data_path,
)
from shared.sensitivity_core import (
    direct_mi,
    fit_model3,
    pool_compact_directory,
    recompute_domains_for_spec,
    save_compact_results,
)

EXTRA = "spouse_cancer"
COMPONENT_TERMS_9 = list(ALL_COMPONENTS_8) + [EXTRA]
FOUR_COHORTS = ["CHARLS", "MHAS", "HRS", "ELSA"]

def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _valid_pair(primary: pd.DataFrame) -> pd.Series:
    if "valid_adjacent_interview_pair" not in primary.columns:
        raise KeyError("valid_adjacent_interview_pair missing from revised primary dataset")
    return _num(primary["valid_adjacent_interview_pair"]).eq(1)


def component_risk_mask(primary: pd.DataFrame, cohort: str, component: str) -> pd.Series:
    """Risk set matching the completed 03 component analysis.

    Eligibility is defined from the prior state whenever possible, while a missing
    current exposure remains eligible for MI. This differs intentionally from the
    stricter observed-exposure risk_* flags used by the non-MI primary component
    models.
    """
    cohort = cohort.upper()
    valid = _valid_pair(primary)

    if component in {"cancer", "stroke", "heart", "diabetes"}:
        prev_col = f"prev_{component}_state"
        if prev_col not in primary.columns:
            raise KeyError(f"{cohort}: {prev_col} missing")
        return valid & _num(primary[prev_col]).eq(0)

    if component == "marital_shock":
        if "partnered_prev" not in primary.columns:
            raise KeyError(f"{cohort}: partnered_prev missing")
        return valid & _num(primary["partnered_prev"]).eq(1)

    if component == "child_death":
        # Child-death exposure may be missing at the current wave and is imputed
        # within any valid adjacent interview pair, exactly as in analysis 03.
        return valid

    if component == "job_exit":
        prev_col = "prev_labor_status_strict"
        if prev_col not in primary.columns:
            raise KeyError(f"{cohort}: {prev_col} missing")
        working_codes = LABOR_STATUS_SPECS[cohort]["working_codes"]
        return valid & _num(primary[prev_col]).isin(working_codes)

    if component == "living_alone_transition":
        if "prev_living_alone" not in primary.columns:
            raise KeyError(f"{cohort}: prev_living_alone missing")
        return valid & _num(primary["prev_living_alone"]).eq(0)

    if component == EXTRA:
        # Applicability=1 means valid adjacent interviews and partnered at both
        # interviews. It does NOT require spouse-cancer state to be observed, so
        # current exposure missingness remains eligible for MI. Unknown/no-partner
        # cells are structural N/A and are excluded.
        if "spouse_cancer_applicable" not in primary.columns:
            raise KeyError(f"{cohort}: spouse_cancer_applicable missing")
        return _num(primary["spouse_cancer_applicable"]).eq(1)

    raise ValueError(f"Unsupported component: {component}")


def _component_seed(base_seed: int, component_index: int) -> int:
    """Match completed analysis 03: +10000, then +1000 per component."""
    return int(base_seed + 10000 + component_index * 1000)


def _fit_one_component_mi(
    mod,
    primary: pd.DataFrame,
    cohort: str,
    component: str,
    component_index: int,
    m: int,
    seed: int,
    mi_max_iter: int,
    optimizer_seq: List[str],
    maxfun: int,
    singular_tol: float,
) -> pd.DataFrame:
    risk_mask = component_risk_mask(primary, cohort, component).fillna(False)
    risk = primary.loc[risk_mask].copy()
    if risk.empty:
        raise RuntimeError(f"{cohort}/{component}: empty component risk set")

    # Force the risk indicator to 1 inside the broader MI risk set. The revised
    # primary overlay otherwise defines some risk_* flags only when the current
    # exposure is observed; that would incorrectly drop newly imputed exposures.
    risk_col = f"risk_{component}"
    risk[risk_col] = 1.0

    if component not in risk.columns:
        raise KeyError(f"{cohort}: component exposure {component} missing")

    comp_seed = _component_seed(seed, component_index)
    targets = [component] + list(MODEL3_COVARIATES)

    # Same MI architecture as the completed 03 analysis:
    # targets = current component + Model-3 covariates;
    # auxiliaries = outcome, wave, baseline age, eight core components,
    # plus the event-specific risk flag. The target's own auxiliary copy is
    # automatically removed by impute_targets().
    imps, _ = impute_targets(
        risk,
        targets,
        m=m,
        seed=comp_seed,
        max_iter=mi_max_iter,
        extra_aux=[risk_col],
    )

    # Verify that the exposure is now complete in every imputation and that the
    # event-specific risk set did not change across imputations.
    for j, d in enumerate(imps, start=1):
        if d[component].isna().any():
            raise RuntimeError(f"{cohort}/{component}: MI {j} left exposure missing")
        if not _num(d[risk_col]).eq(1).all():
            raise AssertionError(f"{cohort}/{component}: MI {j} changed risk eligibility")

    res = fit_model3(
        mod,
        imps,
        cohort,
        "spouse_9shock",
        [component],
        optimizer_seq,
        maxfun,
        singular_tol,
    )
    if res.empty:
        raise RuntimeError(f"{cohort}/{component}: Model 3 returned no result")

    return res


def _merge_component_rows(existing_path: Path, component_results: pd.DataFrame) -> pd.DataFrame:
    """Preserve existing domains/SHARE-compatible schema; replace component rows."""
    if not existing_path.exists():
        raise FileNotFoundError(
            f"Component-only mode requires existing domain result: {existing_path}. "
            "Do not rerun the domain model if it is already completed; copy the existing "
            "04 result file into this outdir first."
        )
    old = pd.read_csv(existing_path)
    if "result_type" not in old.columns:
        raise ValueError(f"{existing_path}: result_type column missing")

    keep = old.loc[old["result_type"].astype(str).str.lower().ne("component")].copy()
    merged = pd.concat([keep, component_results], ignore_index=True, sort=False)

    # Stable paper-facing order: three domains first, then nine components.
    order = {name: i for i, name in enumerate(list(SUMMARY_TERMS) + COMPONENT_TERMS_9)}
    merged["__order"] = merged["effect_name"].map(order).fillna(9999)
    merged = merged.sort_values(["__order"], kind="mergesort").drop(columns="__order").reset_index(drop=True)
    return merged


def run_component_only(mod, primary: pd.DataFrame, cohort: str, out: Path, args) -> None:
    if cohort not in FOUR_COHORTS:
        raise ValueError(
            "--component-only is restricted to CHARLS/MHAS/HRS/ELSA. "
            "SHARE 04 component results are already complete and must not be rerun."
        )

    # The legacy lme4 bridge selects model columns from mod.COMPONENT_TERMS.
    # The primary-8 overlay intentionally contains only the eight core shocks,
    # so temporarily extend this module-global list to make spouse_cancer
    # available to the R model frame. This is exactly the extra-step required
    # for an added ninth component such as analysis 03/04.
    mod.COMPONENT_TERMS = list(COMPONENT_TERMS_9)

    opts = [x.strip() for x in args.optimizer_seq.split(",") if x.strip()]
    component_frames: List[pd.DataFrame] = []

    for idx, component in enumerate(COMPONENT_TERMS_9):
        print(f"[04 component] {cohort}: {component} ({idx+1}/{len(COMPONENT_TERMS_9)})", flush=True)
        res = _fit_one_component_mi(
            mod=mod,
            primary=primary,
            cohort=cohort,
            component=component,
            component_index=idx,
            m=args.m,
            seed=args.seed,
            mi_max_iter=args.mi_max_iter,
            optimizer_seq=opts,
            maxfun=args.maxfun,
            singular_tol=args.singular_tol,
        )
        component_frames.append(res)

    component_results = pd.concat(component_frames, ignore_index=True, sort=False)
    model_path = out / f"{cohort}_model3.csv"
    merged = _merge_component_rows(model_path, component_results)
    save_compact_results(merged, model_path)

    print(
        f"[DONE component-only] {cohort}: "
        f"domains={(merged['result_type']=='domain').sum()}, "
        f"components={(merged['result_type']=='component').sum()} -> {model_path}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=COHORT_ORDER)
    ap.add_argument("--input", default="")
    ap.add_argument(
        "--outdir",
        default=str(DEFAULT_PROJECT_ROOT / "output" / "04_sensitivity_spouse_9shock"),
    )
    ap.add_argument("--pool", action="store_true")
    ap.add_argument(
        "--component-only",
        action="store_true",
        help="Run only nine component Model-3 analyses for CHARLS/MHAS/HRS/ELSA; preserve existing domain rows.",
    )
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260941)
    ap.add_argument("--mi-max-iter", type=int, default=10)
    ap.add_argument("--min-baseline-age", type=float, default=50.0)
    ap.add_argument("--optimizer-seq", default="bobyqa,Nelder_Mead,nloptwrap")
    ap.add_argument("--maxfun", type=int, default=200000)
    ap.add_argument("--singular-tol", type=float, default=1e-4)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    if args.pool:
        if args.component_only:
            ap.error("--pool and --component-only cannot be used together")
        pooled = pool_compact_directory(out, out / "pooled_model3.csv", pattern="*_model3.csv")
        print(pooled.to_string(index=False))
        return

    if not args.cohort:
        ap.error("--cohort is required unless --pool")

    cohort = args.cohort
    inp = Path(args.input) if args.input else default_data_path(cohort)
    mod = load_primary8_module(cohort)
    _, primary = build_long_and_primary(mod, inp, args.min_baseline_age)

    if EXTRA not in primary or "spouse_cancer_applicable" not in primary:
        raise KeyError(f"{cohort}: strict spouse-cancer fields unavailable")

    if args.component_only:
        run_component_only(mod, primary, cohort, out, args)
        return

    # Existing domain-only analysis retained unchanged for reproducibility.
    structural = ~primary["spouse_cancer_applicable"].fillna(0).astype(int).eq(1)
    imps = direct_mi(
        primary,
        list(ALL_COMPONENTS_8) + [EXTRA] + MODEL3_COVARIATES,
        args.m,
        args.seed,
        args.mi_max_iter,
        structural_masks={EXTRA: structural},
    )
    imps = [recompute_domains_for_spec(d, "spouse9") for d in imps]
    res = fit_model3(
        mod,
        imps,
        cohort,
        "spouse_9shock",
        SUMMARY_TERMS,
        [x.strip() for x in args.optimizer_seq.split(",") if x.strip()],
        args.maxfun,
        args.singular_tol,
    )
    save_compact_results(res, out / f"{cohort}_model3.csv")
    print(f"[DONE domain] {cohort}")


if __name__ == "__main__":
    main()
