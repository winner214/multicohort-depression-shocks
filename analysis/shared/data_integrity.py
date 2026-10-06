"""Canonical join keys and fail-fast checks for revised analyses."""
from decimal import Decimal, InvalidOperation
import re
import numpy as np
import pandas as pd

_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def normalize_ids(values):
    """Unify numeric IDs (including leading zeroes) without a float round-trip.

    Alphanumeric IDs such as SHARE mergeid are preserved. Numeric float values
    are converted through Python's scalar string representation, avoiding pandas'
    abbreviated float32 display. A lossy upstream conversion cannot be repaired.
    """
    def one(value):
        if pd.isna(value):
            raise ValueError("Missing participant ID")
        token = str(value).strip()
        if token.lower() in {"", "nan", "none", "<na>"}:
            raise ValueError("Empty participant ID")
        if _NUMBER.fullmatch(token):
            try:
                number = Decimal(token)
            except InvalidOperation as exc:
                raise ValueError("Invalid numeric participant ID") from exc
            if not number.is_finite() or number != number.to_integral_value():
                raise ValueError("Participant ID must be an integer or an alphanumeric key")
            return str(int(number))
        return token
    return values.map(one).astype("string")


def checked_person_wave_merge(left, right, label):
    left, right = left.copy(), right.copy()
    for frame in (left, right):
        frame["id"] = normalize_ids(frame["id"])
        if frame[["id", "wave"]].isna().any().any():
            raise ValueError(f"{label}: missing ID/wave join key")
        if frame.duplicated(["id", "wave"]).any():
            raise ValueError(f"{label}: duplicate ID/wave after normalization")
    merged = left.merge(right, on=["id", "wave"], how="left", validate="one_to_one", indicator="_join_status")
    matched = int(merged["_join_status"].eq("both").sum())
    audit = {"join": label, "left_rows": len(left), "matched_rows": matched,
             "unmatched_rows": len(left)-matched,
             "match_rate": matched/len(left) if len(left) else np.nan}
    if matched != len(left):
        raise ValueError(f"{label}: {len(left)-matched}/{len(left)} person-waves failed ID matching")
    return merged.drop(columns="_join_status"), audit


def country_values(values):
    """Country is a nominal, observed covariate; it is never imputed."""
    out = values.astype("string").str.strip()
    missing = values.isna() | out.str.lower().isin(["", "nan", "none", "<na>"])
    numeric = pd.to_numeric(out, errors="coerce")
    missing |= numeric.notna() & ((numeric <= 0) | ~np.isfinite(numeric))
    valid_numeric = numeric.notna() & ~missing
    out.loc[valid_numeric] = numeric.loc[valid_numeric].map(lambda x: format(x, ".15g"))
    return out.mask(missing)


def country_terms(frame, syntax="r"):
    if "country" not in frame:
        return ""
    if frame["country"].isna().any():
        raise ValueError("Missing SHARE country: country is not imputed")
    # A single-country test/subset has no between-country contrast to estimate.
    if frame["country"].nunique() < 2:
        return ""
    return " + factor(country)" if syntax == "r" else " + C(country)"
