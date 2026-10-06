#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pool the five cohort-specific primary results using DL random effects.

This script reads the manuscript-facing cohort outputs from analysis 01 and writes
only three pooled result files: domains (Models 1-3), components (Model 3), and
cumulative shock burden (Model 3).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from shared.meta_utils import dl_meta
from shared.project_config import (
    ALL_COMPONENTS_8,
    COHORT_ORDER,
    DEFAULT_PROJECT_ROOT,
    SUMMARY_TERMS,
)

MODEL_ORDER = ["Model 1", "Model 2", "Model 3"]


def _find_one(root: Path, cohort: str, suffix: str) -> Path:
    canonical = root / cohort / f"{cohort}_{suffix}"
    if canonical.exists():
        return canonical
    hits = sorted(root.rglob(f"{cohort}_{suffix}"))
    if len(hits) != 1:
        raise FileNotFoundError(
            f"Expected one {cohort}_{suffix} under {root}; found {len(hits)}"
        )
    return hits[0]


def _require_columns(df: pd.DataFrame, cols: Iterable[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label} is missing required columns: {missing}")


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["coef", "se", "OR", "OR_CI_low", "OR_CI_high", "m", "shock_count_level"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _check_unique(df: pd.DataFrame, keys: list[str], label: str) -> None:
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        raise RuntimeError(
            f"Duplicate rows in {label} for {keys}:\n"
            + df.loc[dup, keys].sort_values(keys).to_string(index=False)
        )


def _load_domains(root: Path) -> pd.DataFrame:
    frames = []
    for cohort in COHORT_ORDER:
        fp = _find_one(root, cohort, "03_main_summary_model123.csv")
        d = _numeric(pd.read_csv(fp))
        _require_columns(d, ["exposure", "model", "coef", "se"], str(fp))
        d = d[
            d["exposure"].astype(str).isin(SUMMARY_TERMS)
            & d["model"].astype(str).isin(MODEL_ORDER)
        ].copy()
        _check_unique(d, ["exposure", "model"], cohort)
        if len(d) != len(SUMMARY_TERMS) * len(MODEL_ORDER):
            raise RuntimeError(f"{cohort}: incomplete domain results")
        d["cohort"] = cohort
        frames.append(d)
    return pd.concat(frames, ignore_index=True, sort=False)


def _load_components(root: Path) -> pd.DataFrame:
    frames = []
    for cohort in COHORT_ORDER:
        fp = _find_one(root, cohort, "04_main_component_model123.csv")
        d = _numeric(pd.read_csv(fp))
        _require_columns(d, ["exposure", "model", "coef", "se"], str(fp))
        d = d[
            d["exposure"].astype(str).isin(ALL_COMPONENTS_8)
            & d["model"].astype(str).eq("Model 3")
        ].copy()
        _check_unique(d, ["exposure", "model"], cohort)
        if len(d) != len(ALL_COMPONENTS_8):
            raise RuntimeError(f"{cohort}: incomplete Model 3 component results")
        d["cohort"] = cohort
        frames.append(d)
    return pd.concat(frames, ignore_index=True, sort=False)


def _load_shock_count(root: Path) -> pd.DataFrame:
    frames = []
    for cohort in COHORT_ORDER:
        fp = _find_one(root, cohort, "11_shock_count_model123.csv")
        d = _numeric(pd.read_csv(fp))
        _require_columns(d, ["model", "coef", "se"], str(fp))
        d = d[d["model"].astype(str).eq("Model 3")].copy()
        if len(d) != 4:
            raise RuntimeError(
                f"{cohort}: expected four Model 3 shock-count rows (1, 2, >=3, trend), found {len(d)}"
            )
        d["cohort"] = cohort
        frames.append(d)
    return pd.concat(frames, ignore_index=True, sort=False)


def _pool(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        pooled = dl_meta(group)
        if int(pooled.get("k", 0)) != len(COHORT_ORDER):
            raise RuntimeError(
                f"Five-cohort pooling required for {dict(zip(group_cols, keys))}; k={pooled.get('k')}"
            )
        row = {col: value for col, value in zip(group_cols, keys)}
        row.update(pooled)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Primary five-cohort DL random-effects pooling")
    ap.add_argument(
        "--primary-dir",
        default=str(DEFAULT_PROJECT_ROOT / "output" / "01_primary_8shock"),
    )
    ap.add_argument(
        "--outdir",
        default=str(DEFAULT_PROJECT_ROOT / "output" / "01_primary_8shock" / "pool"),
    )
    args = ap.parse_args()

    root = Path(args.primary_dir).resolve()
    out = Path(args.outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    domains = _load_domains(root)
    components = _load_components(root)
    shock_count = _load_shock_count(root)

    pooled_domains = _pool(domains, ["exposure", "model"])
    pooled_components = _pool(components, ["exposure", "model"])

    shock_group_cols = [
        c
        for c in ["analysis_type", "exposure", "contrast", "shock_count_level", "term", "model"]
        if c in shock_count.columns
    ]
    pooled_shock_count = _pool(shock_count, shock_group_cols)

    pooled_domains.to_csv(out / "pooled_domain_model123.csv", index=False)
    pooled_components.to_csv(out / "pooled_component_model3.csv", index=False)
    pooled_shock_count.to_csv(out / "pooled_shock_count_model3.csv", index=False)

    print(f"[DONE] pooled outputs -> {out}")


if __name__ == "__main__":
    main()
