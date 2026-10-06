#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared configuration for the five-cohort analysis pipeline."""
from __future__ import annotations
from pathlib import Path
import os

COHORT_ORDER = ["CHARLS", "MHAS", "HRS", "ELSA", "SHARE"]

DEFAULT_PROJECT_ROOT = Path(os.environ.get("DEPRESSION_PROJECT_ROOT", Path(__file__).resolve().parents[2]))

DEFAULT_DATA_FILES = {
    "CHARLS": "H_CHARLS_D_Data.dta",
    "MHAS": "H_MHAS_d.dta",
    "HRS": "randhrs1992_2022v1_merged.dta",
    "ELSA": "h_elsa_g3.sav",
    "SHARE": "GH_SHARE_g.rdata",
}

IND_COMPONENTS_8 = ["cancer", "stroke", "heart", "diabetes"]
SOC_COMPONENTS_8 = ["marital_shock", "child_death", "job_exit", "living_alone_transition"]
ALL_COMPONENTS_8 = IND_COMPONENTS_8 + SOC_COMPONENTS_8
SUMMARY_TERMS = ["OverallShock_main", "IndShock_main", "SocShock_main"]
MODEL3_COVARIATES = ["age", "male", "education", "smoking", "drinking"]

LABOR_STATUS_SPECS = {
    "CHARLS": {
        "patterns": ["r{w}lbrf_c"],
        "working_codes": {1, 2, 3, 4, 5},
        "unemployed_codes": {6},
        "known_codes": set(range(1, 9)),
    },
    "MHAS": {
        "patterns": ["r{w}lbrf_m"],
        "working_codes": {1},
        "unemployed_codes": {2},
        "known_codes": {1, 2, 3, 4, 5, 6},
    },
    "HRS": {
        "patterns": ["r{w}lbrf"],
        "working_codes": {1, 2},
        "unemployed_codes": {3},
        "known_codes": {1, 2, 3, 4, 5, 6, 7},
    },
    "ELSA": {
        "patterns": ["r{w}lbrf_e", "r{w}lbrf"],
        "working_codes": {1, 2},
        "unemployed_codes": {3},
        "known_codes": {1, 2, 3, 4, 5, 6, 7},
    },
    "SHARE": {
        "patterns": ["r{w}lbrf_s", "r{w}lbrf"],
        "working_codes": {1},
        "unemployed_codes": {3},
        "known_codes": {1, 3, 5, 6, 8},
    },
}

MARITAL_STATUS_SPECS = {
    "CHARLS": {"partnered_codes": {1, 2, 3}, "adverse_codes": {4, 5, 7}, "known_codes": {1, 2, 3, 4, 5, 7, 8}, "widow_codes": {7}, "divsep_codes": {4, 5}},
    "MHAS":   {"partnered_codes": {1, 3},    "adverse_codes": {4, 5, 7}, "known_codes": {1, 3, 4, 5, 7, 8},    "widow_codes": {7}, "divsep_codes": {4, 5}},
    "HRS":    {"partnered_codes": {1, 2, 3}, "adverse_codes": {4, 5, 6, 7}, "known_codes": {1, 2, 3, 4, 5, 6, 7, 8}, "widow_codes": {7}, "divsep_codes": {4, 5, 6}},
    "ELSA":   {"partnered_codes": {1, 2, 3}, "adverse_codes": {4, 5, 7}, "known_codes": {1, 2, 3, 4, 5, 7, 8}, "widow_codes": {7}, "divsep_codes": {4, 5}},
    "SHARE":  {"partnered_codes": {1, 3},    "adverse_codes": {4, 5, 7}, "known_codes": {1, 3, 4, 5, 7, 8},    "widow_codes": {7}, "divsep_codes": {4, 5}},
}

MOOD_ITEM_SPECS = {
    "CHARLS": {
        "scale": "CES-D10", "coding": "charls_1to4",
        "items": {
            "depressed": ("r{w}depresl", False), "effort": ("r{w}effortl", False),
            "restless_sleep": ("r{w}sleeprl", False), "happy": ("r{w}whappyl", True),
            "lonely": ("r{w}flonel", False), "bothered": ("r{w}botherl", False),
            "could_not_get_going": ("r{w}goingl", False), "concentration": ("r{w}mindtsl", False),
            "hopeful": ("r{w}fhopel", True), "fearful": ("r{w}fearll", False),
        },
        "selected": ["depressed", "happy", "lonely", "bothered", "hopeful", "fearful"],
    },
    "MHAS": {
        "scale": "CES-D9", "coding": "binary",
        "items": {
            "depressed": ("r{w}depres", False), "effort": ("r{w}effort", False),
            "restless_sleep": ("r{w}sleepr", False), "happy": ("r{w}whappy", True),
            "lonely": ("r{w}flone", False), "enjoyed_life": ("r{w}enlife", True),
            "sad": ("r{w}fsad", False), "tired": ("r{w}ftired", False), "energy": ("r{w}energ", True),
        },
        "selected": ["depressed", "happy", "lonely", "enjoyed_life", "sad"],
    },
    "HRS": {
        "scale": "CES-D8", "coding": "binary",
        "items": {
            "depressed": ("r{w}depres", False), "effort": ("r{w}effort", False),
            "restless_sleep": ("r{w}sleepr", False), "happy": ("r{w}whappy", True),
            "lonely": ("r{w}flone", False), "sad": ("r{w}fsad", False),
            "could_not_get_going": ("r{w}going", False), "enjoyed_life": ("r{w}enlife", True),
        },
        "selected": ["depressed", "happy", "lonely", "sad", "enjoyed_life"],
    },
    "ELSA": {
        "scale": "CES-D8", "coding": "binary",
        "items": {
            "depressed": ("r{w}depres", False), "effort": ("r{w}effort", False),
            "restless_sleep": ("r{w}sleepr", False), "happy": ("r{w}whappy", True),
            "lonely": ("r{w}flone", False), "sad": ("r{w}fsad", False),
            "could_not_get_going": ("r{w}going", False), "enjoyed_life": ("r{w}enlife", True),
        },
        "selected": ["depressed", "happy", "lonely", "sad", "enjoyed_life"],
    },
    "SHARE": {
        "scale": "EURO-D12", "coding": "binary_depressive_indicator",
        "items": {
            "depression": ("r{w}depress", False), "pessimism": ("r{w}pessim", False),
            "suicidality": ("r{w}suicid", False), "guilt": ("r{w}guilt", False),
            "sleep": ("r{w}sleep", False), "interest": ("r{w}intrst", False),
            "irritability": ("r{w}irritb", False), "appetite": ("r{w}appett", False),
            "fatigue": ("r{w}fatig", False), "concentration": ("r{w}concnt", False),
            "enjoyment": ("r{w}enjoym", False), "tearfulness": ("r{w}tearfl", False),
        },
        "selected": ["depression", "pessimism", "suicidality", "guilt", "interest", "irritability", "enjoyment", "tearfulness"],
    },
}


def default_data_path(cohort: str, root: Path | str = DEFAULT_PROJECT_ROOT) -> Path:
    cohort = cohort.upper()
    return Path(root) / DEFAULT_DATA_FILES[cohort]
