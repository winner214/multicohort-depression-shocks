"""Enhanced covariate MI; observed exposures/outcomes are never overwritten.

Retains Bayesian-ridge chained equations and the historical rounding rules.
This is an enhanced row-level MI specification, not multilevel categorical MICE.
Missing auxiliary predictors use fixed fills plus explicit missing indicators;
these internal predictor encodings do not recode any exposure in output data.
"""
import gc
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from .project_config import ALL_COMPONENTS_8, MODEL3_COVARIATES


def auxiliary_matrix(base, extra_columns=()):
    columns = list(dict.fromkeys(["Depression_Risk", "wave", "baseline_age"] + list(ALL_COMPONENTS_8) + list(extra_columns)))
    out = pd.DataFrame(index=base.index)
    for c in columns:
        if c not in base:
            continue
        x = pd.to_numeric(base[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if c == "wave":
            out = pd.concat([out, pd.get_dummies(x.astype("string"), prefix="aux_wave", dtype=float)], axis=1)
        else:
            out[f"aux_{c}"] = x.fillna(float(x.median()) if x.notna().any() else 0.)
        if x.isna().any():
            out[f"aux_{c}_missing"] = x.isna().astype(float)
    if "country" in base:
        if base.country.isna().any():
            raise ValueError("SHARE country must be observed before MI")
        out = pd.concat([out, pd.get_dummies(base.country.astype("string"), prefix="aux_country", dtype=float)], axis=1)
    return out.astype(float)


def impute_targets(base, targets, m, seed, max_iter, extra_aux=(), posterior=None):
    if m < 1 or max_iter < 1:
        raise ValueError("m and max_iter must be positive")
    if base.empty:
        return [base.copy() for _ in range(m)], pd.DataFrame()
    targets = list(dict.fromkeys(targets))
    missing_columns = set(targets)-set(base.columns)
    if missing_columns:
        raise KeyError(f"Missing MI targets: {sorted(missing_columns)}")
    x = base[targets].apply(pd.to_numeric, errors="coerce").astype(float)
    if np.isinf(x.to_numpy()).any():
        raise ValueError("Non-finite MI target")
    empty = [c for c in targets if x[c].notna().sum() == 0]
    if empty:
        raise ValueError(f"Cannot impute entirely unobserved targets: {empty}")
    # Targets enter once, as stochastic chained-equation variables, never as
    # fixed copies of their own values/missingness in the auxiliary matrix.
    aux = auxiliary_matrix(base, extra_aux)
    for c in targets:
        aux = aux.drop(columns=[f"aux_{c}", f"aux_{c}_missing"], errors="ignore")
    matrix = pd.concat([x, aux], axis=1)
    outputs = []
    binary = set(ALL_COMPONENTS_8) | {"male", "smoking", "drinking", "income_decline_shock", "spouse_cancer"}
    for i in range(m):
        d = base.copy()
        if x.isna().any().any():
            imp = IterativeImputer(estimator=BayesianRidge(), max_iter=max_iter,
                                   sample_posterior=(m>1 if posterior is None else posterior), random_state=seed+i,
                                   n_nearest_features=None, skip_complete=True,
                                   initial_strategy="most_frequent")
            arr = imp.fit_transform(matrix)
            for j, c in enumerate(targets):
                mask = x[c].isna()
                if not mask.any():
                    continue
                values = arr[:, j]
                if c in binary:
                    values = (values >= .5).astype(float)
                elif c == "education":
                    values = np.round(values).clip(1, 3)
                elif c == "age":
                    values = values.clip(min=0)
                d.loc[mask, c] = values[mask.to_numpy()]
            del imp, arr
        for c in targets:
            if d[c].isna().any():
                raise RuntimeError(f"MI left missing target {c}")
            if not np.array_equal(pd.to_numeric(d.loc[x[c].notna(),c]).to_numpy(), x.loc[x[c].notna(),c].to_numpy()):
                raise AssertionError(f"MI changed observed {c}")
        if "age" in d:
            d["age_group"] = (pd.to_numeric(d.age)>60).astype("float32")
        d[".imp"] = i+1
        outputs.append(d)
        gc.collect()
    audit = pd.DataFrame([{"variable": c, "n_rows": len(base),
                           "observed_before": int(x[c].notna().sum()),
                           "missing_before": int(x[c].isna().sum()),
                           "missing_after": 0, "m": m, "seed": seed,
                           "auxiliary_columns": ";".join(aux.columns),
                           "method": "enhanced_BayesianRidge_row_level_MI"} for c in targets])
    return outputs, audit


def covariate_mi(base, m, seed, max_iter):
    extras = [c for c in ["MoodCore_z", "next_Depression_Risk", "next_self_rated_health",
                          "next_adl", "next_social_activity_decline"] if c in base]
    return impute_targets(base, MODEL3_COVARIATES, m, seed, max_iter, extra_aux=extras)
