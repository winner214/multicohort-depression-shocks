# Multicohort Shock and Depressive Symptom Analysis

This repository contains the analysis code used for a five-cohort study of newly recorded health and social transitions and depressive symptom risk in middle-aged and older adults. The analyses use harmonized data from CHARLS, MHAS, HRS, ELSA, and SHARE.

The public repository is intentionally compact: it contains analysis code only. Plotting scripts, SLURM submission files, audit outputs, validation logs, temporary diagnostics, and the discontinued weighted sensitivity analysis are not included.

## Analysis overview

The final workflow includes the following analyses:

| Script | Analysis |
|---|---|
| `01_primary_8shock.py` | Primary eight-shock analysis within each cohort, including Models 1-3, component analyses, subgroup analyses, cumulative shock burden, and exploratory pathway estimates |
| `01_primary_pool.py` | DerSimonian-Laird random-effects pooling of primary domain, component, and cumulative-burden results |
| `02_sensitivity_mi_8shock.py` | Sensitivity analysis with multiple imputation of the eight shock indicators and Model 3 covariates |
| `03_sensitivity_income_9shock.py` | Nine-shock sensitivity analysis adding income decline |
| `04_sensitivity_spouse_9shock.py` | Nine-shock sensitivity analysis adding spouse cancer |
| `05_sensitivity_original_10shock.py` | Historical ten-shock specification sensitivity analysis |
| `06_sensitivity_lagged_incident.py` | Lagged-outcome and incident-outcome analyses |
| `07_sensitivity_mood_only.py` | Mood-focused outcome sensitivity analysis with reduced somatic content |
| `08_meta_robustness.py` | Alternative meta-analysis using REML/Hartung-Knapp and leave-one-cohort-out analyses |
| `10_sensitivity_social_definition.py` | Social-shock definition sensitivity analyses, including exclusion of work exit and separation of widowhood from divorce/separation |

The numbering follows the frozen analysis archive. Analysis 09 is intentionally absent because the survey-weighted sensitivity analysis was not part of the final analysis set.

## Repository structure

```text
.
├── README.md
├── requirements.txt
└── analysis/
    ├── 01_primary_8shock.py
    ├── 01_primary_pool.py
    ├── 02_sensitivity_mi_8shock.py
    ├── 03_sensitivity_income_9shock.py
    ├── 04_sensitivity_spouse_9shock.py
    ├── 05_sensitivity_original_10shock.py
    ├── 06_sensitivity_lagged_incident.py
    ├── 07_sensitivity_mood_only.py
    ├── 08_meta_robustness.py
    ├── 10_sensitivity_social_definition.py
    ├── original_10shock/
    └── shared/
```

`original_10shock/` contains cohort-specific modelling backends retained for reproducibility. The primary eight-shock specification is applied through the shared overlay code.

## Data

Raw cohort data are **not** included in this repository. Users must obtain harmonized CHARLS, MHAS, HRS, ELSA, and SHARE data from the corresponding data providers and comply with their access and redistribution requirements.

By default, the scripts look for the following filenames under the project root:

```text
H_CHARLS_D_Data.dta
H_MHAS_d.dta
randhrs1992_2022v1_merged.dta
h_elsa_g3.sav
GH_SHARE_g.rdata
```

A different input file can be supplied with `--input` for cohort-level analyses.

## Software requirements

Recommended environment:

- Python 3.10 or later
- R 4.0 or later
- R package `lme4`

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

Install the required R package with:

```r
install.packages("lme4")
```

`Rscript` must be available on the system path because mixed-effects models are fitted through R/lme4.

## Project root

The code is portable and does not require the original server path. Set the project root with the environment variable `DEPRESSION_PROJECT_ROOT`:

```bash
export DEPRESSION_PROJECT_ROOT=/path/to/project
```

If the variable is not set, the repository root is used. Output directories are created under `output/` unless `--outdir` is supplied explicitly.

## Running the primary analysis

Run each cohort separately. For example:

```bash
cd analysis

python 01_primary_8shock.py \
  --cohort CHARLS \
  --input /path/to/H_CHARLS_D_Data.dta \
  --outdir ../output/01_primary_8shock/CHARLS \
  --m 10 \
  --bootstraps 200
```

Repeat for `MHAS`, `HRS`, `ELSA`, and `SHARE`, then pool the cohort-specific results:

```bash
python 01_primary_pool.py \
  --primary-dir ../output/01_primary_8shock \
  --outdir ../output/01_primary_8shock/pool
```

The primary cohort analysis retains only manuscript-relevant outputs:

```text
<COHORT>_01_sample_flow.csv
<COHORT>_02_characteristics_by_wave.xlsx
<COHORT>_03_main_summary_model123.csv
<COHORT>_04_main_component_model123.csv
<COHORT>_05_subgroup_summary_model3.csv
<COHORT>_07_path_results_bootstrap_ci.csv
<COHORT>_09_interaction_summary_model3.csv
<COHORT>_10_shock_count_distribution.csv
<COHORT>_11_shock_count_model123.csv
```

The pooling script writes only:

```text
pooled_domain_model123.csv
pooled_component_model3.csv
pooled_shock_count_model3.csv
```

## Running sensitivity analyses

For analyses 02-07 and 10, run each cohort and then use `--pool`. Example:

```bash
python 06_sensitivity_lagged_incident.py \
  --cohort CHARLS \
  --input /path/to/H_CHARLS_D_Data.dta \
  --outdir ../output/06_sensitivity_lagged_incident

# Repeat for the other four cohorts, then:
python 06_sensitivity_lagged_incident.py \
  --outdir ../output/06_sensitivity_lagged_incident \
  --pool
```

These analyses retain the cohort-level `*_model3.csv` files and a single `pooled_model3.csv` file.

Meta-analysis robustness is run after the primary cohort analyses:

```bash
python 08_meta_robustness.py \
  --primary-dir ../output/01_primary_8shock \
  --outdir ../output/08_meta_robustness
```

It writes the two substantive robustness files:

```text
primary_model3_DL_REML_HK.csv
primary_model3_leave_one_out_REML_HK.csv
```

## Statistical framework

The primary cohort analyses use participant-specific random-intercept logistic regression. Cohort-specific estimates are combined using random-effects meta-analysis. Model 1 includes shock exposure and wave fixed effects; Model 2 additionally adjusts for age, sex, and education; Model 3 additionally adjusts for smoking and drinking. Covariate multiple imputation uses 10 imputations in the final specification.

The primary shock domains are based on eight incident components:

- Health-related: cancer, stroke, heart disease, and diabetes
- Social disruption: marital disruption, death of a child, work exit, and transition to living alone

Incident shocks are defined from the immediately preceding interview to the current interview. Missing shock components are not automatically coded as non-events. Cumulative burden is calculated only when all eight primary shock components are determinable.

## Reproducibility notes

- Random seeds are defined within each analysis script.
- The repository does not contain cohort microdata or derived participant-level data.
- Audit tables, environment checks, plotting scripts, cluster submission scripts, and intermediate meta-ready duplicates have been removed from the public version.
- The code is intended to reproduce the analysis workflow once the required harmonized cohort datasets are available locally.
