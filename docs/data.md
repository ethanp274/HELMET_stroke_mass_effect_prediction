# Data layout

The real cohorts cannot be distributed. If you hold them under a data use
agreement, arrange them like this (relative to the repo root, or to
`$MASS_EFFECT_DATA` if set):

```
data/
  raw/
    LTV_2.0.csv          MGB longitudinal table (one row per patient-timepoint)
    OTV_2.1.csv          MGB one-time table (one row per patient)
    LTV_BMC_2.csv        BMC longitudinal table
    OTV_BMC_3.csv        BMC one-time table
    OTV_BMC_4.csv        BMC one-time table, revised extract
    mgb_insurance.csv    Insurance status, MGB (cohort summary only)
    bmc_insurance.csv    Insurance status, BMC (cohort summary only)
  censored/              Time-censored extracts (external_cohort.py, panos_mode=True)
  processed/             Written by the processing pipeline; regenerable
```

Direct identifiers present in the BMC extracts (`encounter_id`, `CDW_ID`,
`CDW_MRN`) are dropped in `src/helmet_mass_effect_pred/data/external_cohort.py` before any
modelling.

Everything under `data/` except `data/demo/` is git-ignored, as are all `*.csv`
files anywhere in the tree.

## Synthetic demo data

`data/demo/*.json` were produced by fitting an SDV `GaussianCopulaSynthesizer`
to each processed cohort and sampling 1000 (MGB) or 200 (BMC) rows. They share
the real data's schema and marginal distributions but contain no real records.
