#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified server entry point for the revised primary analysis.

Example:
  python 01_primary_8shock.py --cohort CHARLS --input /path/H_CHARLS_D_Data.dta \
      --outdir /path/output/01_primary_8shock/CHARLS --m 10 --bootstraps 200
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from shared.primary_runner import make_primary_parser, run_primary
from shared.project_config import COHORT_ORDER


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python 01_primary_8shock.py --cohort {CHARLS,MHAS,HRS,ELSA,SHARE} [cohort-specific options]")
        print("Core options: --input PATH --outdir DIR --m 10 --bootstraps 200 --min-baseline-age 50")
        return
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--cohort", required=True, choices=COHORT_ORDER)
    ns, rest = pre.parse_known_args()
    parser = make_primary_parser(ns.cohort)
    parser.add_argument("--cohort", default=ns.cohort, choices=COHORT_ORDER)
    args = parser.parse_args(rest)
    run_primary(ns.cohort, args)

if __name__ == "__main__":
    main()
