# Summarize characteristics of final dataset used for xgb training
# Compiled by Ethan Phillips (Univ of Oxford) on 7 August 2024


#################################################################
# LOAD REQUIRED PACKAGES
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, poisson_means_test


######################################################
# DEFINE HELPER METHODS

def nonpara_pval(x, y):
    return mannwhitneyu(x, y)[1]

def prop_pval(x, n_x, y, n_y):
    return poisson_means_test(x, n_x, y, n_y)[1]


######################################################
# MAIN CODE TO BE EXECUTED BELOW
def summarize_data():
    OLTV = pd.read_json('data/processed/OLTV_24hr_demonstration.json')
    bmc_OLTV = pd.read_json('data/processed/bmc_OLTV_24hr_demonstration.json')

    n_patients = len(OLTV.ptid.unique())
    n_bmc_patients = len(bmc_OLTV.ptid.unique())

    print(f"Number of patients in MGB: {n_patients}")
    print(f"Number of patients in BMC: {n_bmc_patients}")

    hours_per_pt = OLTV.groupby('ptid').dt.max()
    hours_per_pt_bmc = bmc_OLTV.groupby('ptid').dt.max()

    print(f"Mean hours per patient in MGB: {hours_per_pt.mean():.2f} ({hours_per_pt.std():.2f})")
    print(f"Mean hours per patient in BMC: {hours_per_pt_bmc.mean():.2f} ({hours_per_pt_bmc.std():.2f})")

    print(f"Hours p-value: {nonpara_pval(hours_per_pt, hours_per_pt_bmc)}")

    mean_age = OLTV.age_calc.mean()
    std_age = OLTV.age_calc.std()
    bmc_mean_age = bmc_OLTV.age_calc.mean()
    bmc_std_age = bmc_OLTV.age_calc.std()

    print(f"Mean age in MGB: {mean_age:.2f} ({std_age:.2f})")
    print(f"Mean age in BMC: {bmc_mean_age:.2f} ({bmc_std_age:.2f})")

    print(f"Age p-value: {nonpara_pval(OLTV.age_calc, bmc_OLTV.age_calc)}")

    prop_female = OLTV[OLTV.sex >= 0].groupby('ptid').sex.last().mean()
    bmc_prop_female = bmc_OLTV[bmc_OLTV.sex >= 0].groupby('ptid').sex.last().mean()

    print(f'Proportion female in MGB: {prop_female:.3f}')
    print(f'Proportion female in BMC: {bmc_prop_female:.3f}')

    print(f"Sex prop p-value: {prop_pval(prop_female * n_patients, n_patients, bmc_prop_female * n_bmc_patients, n_bmc_patients)}")

    max_mls = OLTV.groupby('ptid').size_mls.max()
    bmc_max_mls = bmc_OLTV.groupby('ptid').size_mls.max()

    print(f'Average max mls in MGB: {max_mls.mean():.2f} ({max_mls.std():.2f})')
    print(f'Average max mls in BMC: {bmc_max_mls.mean():.2f} ({bmc_max_mls.std():.2f})')

    print(f"Max mls p-value: {nonpara_pval(max_mls, bmc_max_mls)}")


###################################################
# EXECUTE MAIN CODE

if __name__ == "__main__":
    summarize_data()


