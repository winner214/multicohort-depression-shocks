#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import pandas as pd


def read_raw_for_module(mod, input_path: Path):
    raw_cols, rawmap = mod.required_raw_columns(input_path)
    if hasattr(mod, "read_data_file"):
        raw = mod.read_data_file(input_path, columns=raw_cols)
    elif input_path.suffix.lower() == ".dta":
        raw = pd.read_stata(input_path, columns=raw_cols, convert_categoricals=False)
    else:
        raise ValueError(f"Unsupported input type for {input_path}")
    if hasattr(mod, "cast_small_dtypes"):
        raw = mod.cast_small_dtypes(raw)
    return raw, rawmap


def build_long_and_primary(mod, input_path: Path, min_baseline_age: float = 50.0):
    raw, rawmap = read_raw_for_module(mod, input_path)
    long_df = mod.build_long_panel(raw, rawmap=rawmap)
    primary = mod.prepare_main_dataset(long_df, min_baseline_age=min_baseline_age)
    primary = mod.add_shock_count_variables(primary)
    return long_df, primary
