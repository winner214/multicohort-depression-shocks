#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import importlib.util
from pathlib import Path
from .primary_overlay import install_primary8_overlay


def project_code_root() -> Path:
    return Path(__file__).resolve().parents[1]


def legacy_script_path(cohort: str) -> Path:
    cohort = cohort.upper()
    return project_code_root() / "original_10shock" / f"{cohort}_original10.py"


def load_python_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_legacy_module(cohort: str):
    cohort = cohort.upper()
    return load_python_module(legacy_script_path(cohort), f"legacy_{cohort.lower()}")


def load_primary8_module(cohort: str):
    cohort = cohort.upper()
    mod = load_legacy_module(cohort)
    return install_primary8_overlay(mod, cohort)
