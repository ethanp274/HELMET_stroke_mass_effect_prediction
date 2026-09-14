# HELMET: real-time prediction of edema trajectory after large MCA stroke

Code for **HELMET** (Hybrid Ensemble Learning Models for Edema Trajectory), which predicts
midline-shift severity — an established indicator of malignant cerebral edema — hour by
hour after a large middle cerebral artery stroke, over 8-hour (HELMET-8) and 24-hour
(HELMET-24) windows.

HELMET combines XGBoost over time-varying EHR data and human-curated radiographic features
with class probabilities from Clinical-Longformer classifiers fine-tuned on radiology
reports. The models were derived on 623 patients from Mass General Brigham and externally
validated at Boston Medical Center, with mean AUROCs of 96.6% and 92.5% respectively.

> **Paper** — Phillips E, O'Donoghue O, Zhang Y, *et al.* Hybrid machine learning for real-time prediction of edema trajectory in large middle cerebral artery stroke. *npj Digital Medicine* **8**, 288 (2025). [doi:10.1038/s41746-025-01687-y](https://doi.org/10.1038/s41746-025-01687-y)

## Status of this repository

This is a research codebase released for transparency and reuse alongside the
paper. It is the code that produced the published results, reorganised into an
installable package with pinned dependencies — it is not a maintained library or
a clinical product. **Nothing here is validated for clinical use.**

## Data availability

**No patient data is included in this repository, and none can be.** The Mass General
Brigham and Boston Medical Center cohorts contain protected health information and are
covered by data use agreements. As the paper's data availability statement sets out,
access requires HIPAA certification and approval from the collaborating institutions' IRBs.

What *is* included:

| Path | Contents |
| --- | --- |
| `data/demo/` | Fully synthetic cohorts (1000 MGB / 200 BMC rows) generated with an SDV Gaussian copula fitted to the real data. Structurally faithful, non-disclosive, and enough to exercise the pipeline end to end. |
| `results/aggregate/` | Per-fold summary metrics (sensitivity, specificity, AUC and friends) behind the published tables. |
| `results/models/` | Trained XGBoost models and the fitted mean-imputers for each cohort, horizon and CV fold. |
| `configs/` | W&B sweep definitions and the selected hyperparameters. |

Per-patient model outputs and the raw cohort extracts are **deliberately absent**.
They are available from the authors under the relevant data use agreement.

To run against real data, place the extracts under `data/raw/` (or point
`$MASS_EFFECT_DATA` elsewhere) using the layout in [docs/data.md](docs/data.md).

## Installation

The project uses [uv](https://docs.astral.sh/uv/) and pins every dependency to
the versions that produced the published results (Python 3.11).

```bash
git clone https://github.com/ethanp274/HELMET_stroke_mass_effect_prediction.git
cd HELMET_stroke_mass_effect_prediction
uv sync
```

`uv` fetches Python 3.11 itself if you do not have it. For the optional CUDA
path in `scripts/external_validation/xgb_llm_reduced.py`, add `uv sync --extra gpu`.

### Where the environment lives

Keep the virtual environment **outside** the repository. uv's default is a `.venv` folder
inside the project, which goes badly in a synced folder such as OneDrive: it churns tens of
thousands of small files, locks them mid-install and turns them into cloud placeholders that
then fail at import time. Point uv at a local directory before your first `uv` command, in
every new shell (or your shell profile):

```powershell
$env:UV_PROJECT_ENVIRONMENT = "$env:LOCALAPPDATA\uv-envs\HELMET_stroke_mass_effect_prediction"
```

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.local/share/uv-envs/HELMET_stroke_mass_effect_prediction"   # macOS / Linux
```

Then run `uv sync` and `uv run` as normal, and confirm the interpreter lives there:

```bash
uv run python -c "import sys; print(sys.executable)"
```

## Quick start

Train on the synthetic demo data — no credentials, no real data, a few minutes on CPU:

```bash
MASS_EFFECT_DEMO=1 WANDB_MODE=offline uv run python scripts/train_xgb_multiclass.py --lookahead_hours=24
```

`MASS_EFFECT_DEMO=1` switches the training scripts from `data/processed/` to the
synthetic cohorts. The synthetic targets are the four midline-shift severity
classes, so demo mode works with `train_xgb_multiclass.py` and
`train_xgb_disaggregated.py`; the binary `train_xgb_basic.py` needs real data.

Every hyperparameter of `train_xgb` can be passed as a flag (`--max_depth=3`,
`--num_folds=5`, and so on). Experiment tracking uses Weights & Biases; drop
`WANDB_MODE=offline` after `uv run wandb login` to log runs.

On PowerShell, set the variables first: `$env:MASS_EFFECT_DEMO=1; $env:WANDB_MODE="offline"`.

## Repository layout

```
src/helmet_mass_effect_pred/          Importable library
  data/processing.py        Feature engineering, time censoring, target construction
  data/splitting.py         Missingness filtering, imputation, cross-validation splits
  data/external_cohort.py   Harmonisation of the BMC cohort onto the MGB schema
  data/transition_kernel.py Empirical midline-shift transition kernels
  evaluation/scoring.py     Outcome scoring
  evaluation/cv_results.py  Per-fold metric aggregation
  evaluation/visualisation.py  t-SNE / PCA projections

scripts/                  Training entry points
  train_xgb_basic.py          Binary outcome
  train_xgb_multiclass.py     Multiclass midline-shift severity
  train_xgb_disaggregated.py  Multiclass, disaggregated by severity band
  summarise_cohort.py         Cohort characteristics table
  run_sweep.py                W&B sweep launcher
  external_validation/        The same, for the held-out BMC cohort

analysis/                 Post-training analysis and paper figures
configs/                  Sweep definitions and selected hyperparameters
data/demo/                Synthetic demonstration cohorts
results/                  Aggregate metrics and trained models
```

The three `train_xgb_*` scripts share most of their body and differ in how the
outcome is defined. They are kept as separate entry points on purpose: each one
is the script that produced its corresponding published result, and merging them
would put that correspondence at risk.

The XGBoost models use class probabilities from fine-tuned Clinical-Longformer
classifiers over radiology reports as features. `src/helmet_mass_effect_pred/data/processing.py`
loads those classifiers from the HuggingFace Hub
(`ethanp5/edema_prediction_LLM_{8,24,36}hr`); access is granted by the authors on
request. Training them — merging reports, converting tabular records into clinical
narratives and fine-tuning encoders — lives in a companion repository,
[HELMET_stroke_radiology_llm](https://github.com/ethanp274/HELMET_stroke_radiology_llm).

## Citation

If you use this code, please cite the paper:

```bibtex
@article{phillips2025helmet,
  title   = {Hybrid machine learning for real-time prediction of edema trajectory in large middle cerebral artery stroke},
  author  = {Phillips, Ethan and O'Donoghue, Odhran and Zhang, Yumeng and Tsimpos, Panos and Mallinger, Leigh Ann and Chatzidakis, Stefanos and Pohlmann, Jack and Du, Yili and Kim, Ivy and Song, Jonathan and Brush, Benjamin and Smirnakis, Stelios and Ong, Charlene J. and Orfanoudaki, Agni},
  journal = {npj Digital Medicine},
  volume  = {8},
  pages   = {288},
  year    = {2025},
  doi     = {10.1038/s41746-025-01687-y}
}
```

## Licence

Apache-2.0. See [LICENSE](LICENSE).
