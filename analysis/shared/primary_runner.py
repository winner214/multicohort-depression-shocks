#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from pathlib import Path
from .cohort_loader import load_primary8_module
from .project_config import DEFAULT_PROJECT_ROOT, default_data_path


def make_primary_parser(cohort: str) -> argparse.ArgumentParser:
    cohort = cohort.upper()
    p = argparse.ArgumentParser(description=f"{cohort} revised 8-shock primary analysis")
    p.add_argument("--input", default=str(default_data_path(cohort)))
    p.add_argument("--outdir", default=str(DEFAULT_PROJECT_ROOT / "output" / "01_primary_8shock" / cohort))
    p.add_argument("--m", "--imputations", dest="imputations", type=int, default=10, help="Number of covariate MI datasets (default: 10); either --m or --imputations may be used")
    p.add_argument("--min-baseline-age", type=float, default=50.0)
    p.add_argument("--bootstraps", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--mi-max-iter", type=int, default=8)
    p.add_argument("--mi-n-nearest-features", type=int, default=4)
    p.add_argument("--optimizer-seq", default="bobyqa,Nelder_Mead,nloptwrap")
    p.add_argument("--maxfun", type=int, default=200000)
    p.add_argument("--singular-tol", type=float, default=1e-4)
    return p


def run_primary(cohort: str, args: argparse.Namespace) -> None:
    cohort = cohort.upper()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    mod = load_primary8_module(cohort)
    mod.run_pipeline(args)
