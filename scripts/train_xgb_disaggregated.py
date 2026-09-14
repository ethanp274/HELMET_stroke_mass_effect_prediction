# Final XGB training script for edema prediction using LLM and XGB
# Compiled by Ethan Phillips (Univ of Oxford) on 29 July 2024

#################################################################
# LOAD REQUIRED PACKAGES

import sys
import os
import pandas as pd
import numpy as np
from typing import Literal
from pydantic import confloat, validate_call
from sklearn.model_selection import KFold, train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
import wandb
from sklearn.linear_model import LogisticRegression
import fire
import matplotlib.pyplot as plt
from wandb.integration.xgboost import WandbCallback
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from torch import tensor as tnsr
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy, MulticlassAveragePrecision
import shap
from sklearn.metrics import precision_recall_curve as pr_curve
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score


# Demo mode is necessary for the code to be run using synthetic data
# Set MASS_EFFECT_DEMO=1 to train on the synthetic cohorts in data/demo.
DEMO_MODE = os.environ.get("MASS_EFFECT_DEMO", "0") == "1"

# set working directory one level higher

# import processing method from other file
from helmet_mass_effect_pred.data.processing import process_data, target_transform_partial

from helmet_mass_effect_pred.data.splitting import drop_missing_impute_and_split

# Output directories for figures (git-ignored, so absent in a fresh clone).
for _output_dir in ('results/figures/curves', 'results/figures/shap'):
    os.makedirs(_output_dir, exist_ok=True)

######################################################
# DEFINE HELPER METHODS

# Method to impute missing values
def impute_missing(train_df, test_df, how):
    ## define imputer
    if how == "iterative":
        imputer = IterativeImputer().fit(train_df)
    elif how == "simple":
        imputer = SimpleImputer().fit(train_df)
    elif how == "knn":
        imputer = KNNImputer().fit(train_df)
    
    ## impute missing data and return
    return imputer.transform(train_df), imputer.transform(test_df)

# Method to split datasets into KFolds and apply imputation
def prepare_splits_and_impute(OLTV_df, OTV_df, num_folds, impute_method):
    # split dataset into train and test subsets (with 5 cv splits)
    ## get patient list
    ptids = OLTV_df.ptid.unique()
    
    ## define KFold splitter
    kf = KFold(n_splits = num_folds, shuffle = True)
    
    ## split data into train and test splits based on patients (no leakage of patients across splits)
    train_test_list = []
    for train_indices, test_indices in kf.split(ptids):
        train_test_list.append([ptids[train_indices], ptids[test_indices]])

    ## instantiate cv data lists
    cv_train_ptids = []
    cv_test_ptids = []
    cv_train_OLTVs = []
    cv_test_OLTVs = []
    if not DEMO_MODE:
        cv_train_OTVs = []
        cv_test_OTVs = []
    cv_train_targets = []
    cv_test_targets = []

    ## impute missing values and save cv sets
    for ptid_train, ptid_test in train_test_list:

        ### separate OLTVs based on ptid splits
        train_OLTV = OLTV_df[OLTV_df.ptid.isin(ptid_train)].copy()
        test_OLTV = OLTV_df[OLTV_df.ptid.isin(ptid_test)].copy()

        if not DEMO_MODE:
            ## separate OTVs based on ptid splits
            train_OTV = OTV_df[OTV_df.ptid.isin(ptid_train)].copy()
            test_OTV = OTV_df[OTV_df.ptid.isin(ptid_test)].copy()

        ### separate targets from data
        train_targets = train_OLTV['target'].copy().to_numpy()
        test_targets = test_OLTV['target'].copy().to_numpy()

        train_OLTV = train_OLTV.drop(columns = ['target'])
        test_OLTV = test_OLTV.drop(columns = ['target'])
        
        ### separate ptids from data
        train_ptids = train_OLTV['ptid'].copy()
        test_ptids = test_OLTV['ptid'].copy()

        train_OLTV = train_OLTV.drop(columns = ['ptid'])
        test_OLTV = test_OLTV.drop(columns = ['ptid'])

        ### impute data based on training set
        train_OLTV, test_OLTV = impute_missing(train_OLTV, test_OLTV, how = impute_method)
        
        ### save ptids, data, and targets for each cv
        cv_train_ptids.append(train_ptids)
        cv_test_ptids.append(test_ptids)
        cv_train_OLTVs.append(train_OLTV)
        cv_test_OLTVs.append(test_OLTV)
        if not DEMO_MODE:
            cv_train_OTVs.append(train_OTV)
            cv_test_OTVs.append(test_OTV)
        cv_train_targets.append(train_targets)
        cv_test_targets.append(test_targets)
    
    if not DEMO_MODE:
        return (
            cv_train_ptids, cv_test_ptids,
            cv_train_OLTVs, cv_test_OLTVs,
            cv_train_OTVs, cv_test_OTVs,
            cv_train_targets, cv_test_targets
        )
    else:
        return (
            cv_train_ptids, cv_test_ptids,
            cv_train_OLTVs, cv_test_OLTVs,
            cv_train_targets, cv_test_targets
        )

# Method to generate shap plots
def make_shap_plot(explainer, data, image_path):
    target_col_names = ["0", "0_3", "3_8", "8_up"]
    
    plt.figure()
    shap_values = explainer.shap_values(data)
    shap_values_list = [shap_values[:,:,i] for i in range(shap_values.shape[2])]
    shap.summary_plot(
        shap_values = shap_values_list,
        features = data,
        feature_names = data.columns.tolist(),
        plot_type = "bar",
        class_names = target_col_names,
        show = False,
    )
    plt.savefig("results/figures/shap/" + image_path)
    wandb.log({image_path: wandb.Image(plt)})
    plt.close()


# Method to generate prc and roc plot data
def get_plot_data(target_df, proba, cv_num, type: Literal["mgb", "bmc"], model: Literal["baseline", "xgb"], filtered: bool = False):
    num_classes = 4
    target_col_names = ["0", "0_3", "3_8", "8_up"]
    prc_data = [{},{},{},{}]
    roc_data = [{},{},{},{}]

    for i in range(num_classes):
        t_nom = target_col_names[i]
        one_hot_t = np.eye(num_classes)[target_df]

        ## get precision-recall curve data
        precision, recall, _ = pr_curve(one_hot_t[:, i], proba[:, i])
        prc_auc = average_precision_score(one_hot_t[:, i], proba[:, i])
        prc_data[i]["precision"] = precision
        prc_data[i]["recall"] = recall
        prc_data[i]["auc"] = prc_auc

        ## plot precision-recall curve
        if filtered:
            image_path = f"{model}_{type}_filtered_precision_recall_class_{t_nom}_cv_{cv_num}.png"
        else:
            image_path = f"{model}_{type}_precision_recall_class_{t_nom}_cv_{cv_num}.png"

        plt.figure()
        plt.plot(recall, precision, label = f"{type} Class {t_nom} (area = {prc_auc:.2f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")

        if filtered:
            plt.title(f"{model} {type} Filtered Precision-Recall Curve for Class {t_nom} for CV {cv_num}")
        else:
            plt.title(f"{model} {type} Precision-Recall Curve for Class {t_nom} for CV {cv_num}")

        plt.legend(loc = "lower right")
        plt.savefig("results/figures/curves/" + image_path)
        wandb.log({image_path: wandb.Image(plt)})
        plt.close()

        ## receiver operator curve data
        fpr, tpr, _ = roc_curve(one_hot_t[:, i], proba[:, i])

        try:
            roc_auc = roc_auc_score(one_hot_t[:, i], proba[:, i])
        except ValueError:
            roc_auc = 0.0

        roc_data[i]["fpr"] = fpr
        roc_data[i]["tpr"] = tpr
        roc_data[i]["auc"] = roc_auc

        ## plot receiver operator curve
        if filtered:
            image_path = f"{model}_{type}_filtered_receiver_operator_class_{t_nom}_cv_{cv_num}.png"
        else:
            image_path = f"{model}_{type}_receiver_operator_class_{t_nom}_cv_{cv_num}.png"

        plt.figure()
        plt.plot(fpr, tpr, label = f"{type} Class {t_nom} (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        if filtered:
            plt.title(f"{model} {type} Filtered Receiver Operating Characteristic for Class {t_nom} for CV {cv_num}")
        else:
            plt.title(f"{model} {type} Receiver Operating Characteristic for Class {t_nom} for CV {cv_num}")
        
        plt.legend(loc = "lower right")
        plt.savefig("results/figures/curves/" + image_path)
        wandb.log({image_path: wandb.Image(plt)})
        plt.close()

    return prc_data, roc_data

# Method to save computed metrics to WandB interface
def save_metrics(model, type, prefix, mean_acc, std_acc, mean_auroc, std_auroc, mean_auprc, std_auprc):
    wandb.log(
        {
            f"{model} {type} {prefix}mean accuracy": mean_acc,
            f"{model} {type} {prefix}sd accuracy": std_acc,
            f"{model} {type} {prefix}mean AUROC": mean_auroc,
            f"{model} {type} {prefix}sd AUROC": std_auroc,
            f"{model} {type} {prefix}mean AUPRC": mean_auprc,
            f"{model} {type} {prefix}sd AUPRC": std_auprc
        }
    )

def save_auroc_only(model, type, prefix, mean_auroc, std_auroc):
    wandb.log(
        {
            f"{model} {type} {prefix}mean AUROC": mean_auroc,
            f"{model} {type} {prefix}sd AUROC": std_auroc
        }
    )

# Method for getting mean and sd value across multiple lists (for averaging plot data)
def average_across_lists(list_of_lists):
    arrays = [np.array(x) for x in list_of_lists]
    means = [np.mean(k) for k in zip(*arrays)]
    stds = [np.std(k) for k in zip(*arrays)]
    return means, stds

# Method for creating and saving combined prc and roc plots
def create_combined_curve_plots(label, prc_baseline_mgb, prc_baseline_bmc, prc_xgb_mgb, prc_xgb_bmc, roc_baseline_mgb, roc_baseline_bmc, roc_xgb_mgb, roc_xgb_bmc):
    ## Precision-Recall curve
    plt.figure()
    plt.plot(prc_baseline_mgb[0][0], prc_baseline_mgb[1][0], label = f"Baseline MGB (area = {prc_baseline_mgb[2][0]:.2f}({prc_baseline_mgb[2][1]:.2f}))", color = 'red')
    plt.fill_between(prc_baseline_mgb[0][0], [mean - 1.96*sd for mean, sd in zip(prc_baseline_mgb[1][0],prc_baseline_mgb[1][1])], [mean + 1.96*sd for mean, sd in zip(prc_baseline_mgb[1][0],prc_baseline_mgb[1][1])], facecolor = 'red', alpha = 0.3)

    plt.plot(prc_baseline_bmc[0][0], prc_baseline_bmc[1][0], label = f"Baseline BMC (area = {prc_baseline_bmc[2][0]:.2f}({prc_baseline_bmc[2][1]:.2f}))", color = 'orange')
    plt.fill_between(prc_baseline_bmc[0][0], [mean - 1.96*sd for mean, sd in zip(prc_baseline_bmc[1][0], prc_baseline_bmc[1][1])], [mean + 1.96*sd for mean, sd in zip(prc_baseline_bmc[1][0], prc_baseline_bmc[1][1])], facecolor = 'orange', alpha = 0.3)
    
    plt.plot(prc_xgb_mgb[0][0], prc_xgb_mgb[1][0], label = f"XGBoost MGB (area = {prc_xgb_mgb[2][0]:.2f}({prc_xgb_mgb[2][1]:.2f}))", color = 'blue')
    plt.fill_between(prc_xgb_mgb[0][0], [mean - 1.96 * sd for mean, sd in zip(prc_xgb_mgb[1][0], prc_xgb_mgb[1][1])], [mean + 1.96*sd for mean, sd in zip(prc_xgb_mgb[1][0], prc_xgb_mgb[1][1])], facecolor = 'blue', alpha = 0.3)
    
    plt.plot(prc_xgb_bmc[0][0], prc_xgb_bmc[1][0], label = f"XGBoost BMC (area = {prc_xgb_bmc[2][0]:.2f}({prc_xgb_bmc[2][1]:.2f}))", color = 'green')
    plt.fill_between(prc_xgb_bmc[0][0], [mean - 1.96*sd for mean, sd in zip(prc_xgb_bmc[1][0], prc_xgb_bmc[1][1])], [mean + 1.96*sd for mean, sd in zip(prc_xgb_bmc[1][0], prc_xgb_bmc[1][1])], facecolor = 'green', alpha = 0.3)

    plt.xlabel(f"{label}Recall")
    plt.ylabel(f"{label}Precision")
    plt.title(f"Combined Average {label}Precision-Recall Curve")
    plt.legend(loc = "upper right")

    path_label = "filtered_" if label == "Filtered " else ""

    image_path = f"combined_avg_{path_label}precision_recall_curve.png"
    wandb.log({image_path: wandb.Image(plt)})
    plt.savefig("results/figures/curves/" + image_path)
    plt.close()

    ## Receiver Operating Characteristic
    plt.figure()
    plt.plot(roc_baseline_mgb[0][0], roc_baseline_mgb[1][0], label = f"Baseline MGB (area = {roc_baseline_mgb[2][0]:.2f}({roc_baseline_mgb[2][1]:.2f}))", color = 'red')
    plt.fill_between(roc_baseline_mgb[0][0], [mean - 1.96*sd for mean, sd in zip(roc_baseline_mgb[1][0], roc_baseline_mgb[1][1])], [mean + 1.96*sd for mean, sd in zip(roc_baseline_mgb[1][0], roc_baseline_mgb[1][1])], facecolor = 'red', alpha = 0.2)

    plt.plot(roc_baseline_bmc[0][0], roc_baseline_bmc[1][0], label = f"Baseline BMC (area = {roc_baseline_bmc[2][0]:.2f}({roc_baseline_bmc[2][1]:.2f}))", color = 'orange')
    plt.fill_between(roc_baseline_bmc[0][0], [mean - 1.96*sd for mean, sd in zip(roc_baseline_bmc[1][0], roc_baseline_bmc[1][1])], [mean + 1.96*sd for mean, sd in zip(roc_baseline_bmc[1][0], roc_baseline_bmc[1][1])], facecolor = 'orange', alpha = 0.2)
    
    plt.plot(roc_xgb_mgb[0][0], roc_xgb_mgb[1][0], label = f"XGBoost MGB (area = {roc_xgb_mgb[2][0]:.2f}({roc_xgb_mgb[2][1]:.2f}))", color = 'blue')
    plt.fill_between(roc_xgb_mgb[0][0], [mean - 1.96*sd for mean, sd in zip(roc_xgb_mgb[1][0], roc_xgb_mgb[1][1])], [mean + 1.96*sd for mean, sd in zip(roc_xgb_mgb[1][0], roc_xgb_mgb[1][1])], facecolor = 'blue', alpha = 0.2)
    
    plt.plot(roc_xgb_bmc[0][0], roc_xgb_bmc[1][0], label = f"XGBoost BMC (area = {roc_xgb_bmc[2][0]:.2f}({roc_xgb_bmc[2][1]:.2f}))", color = 'green')
    plt.fill_between(roc_xgb_bmc[0][0], [mean - 1.96*sd for mean, sd in zip(roc_xgb_bmc[1][0], roc_xgb_bmc[1][1])], [mean + 1.96*sd for mean, sd in zip(roc_xgb_bmc[1][0], roc_xgb_bmc[1][1])], facecolor = 'green', alpha = 0.2)

    plt.xlabel(f"{label}False Positive Rate")
    plt.ylabel(f"{label}True Positive Rate")
    plt.title(f"Combined Average {label}Receiver Operator Characteristic")
    plt.legend(loc = "lower right")

    image_path = f"combined_avg_{path_label}roc.png"
    wandb.log({image_path: wandb.Image(plt)})
    plt.savefig("results/figures/curves/" + image_path)
    plt.close()


#####################################################
# MAIN METHOD TO BE CALLED IN EXECUTION

@validate_call
def train_xgb(
    project: str = "final_model_training",
    num_folds: int = 5,
    impute_method: Literal['simple', 'iterative', 'knn'] = 'simple', 
    time_NaN_fill: int = -10,
    lookback_hours: int = 24,
    lookahead_hours: int  = 24,
    col_na_drop_fraction: confloat(ge=0.0, le=1.0) = 0.3, 
    patient_na_drop_fraction: confloat(ge=0.0, le=1.0) = 0.2,
    max_depth: int = 2,
    min_child_weight: int = 1,
    learning_rate: confloat(ge=0.001, le=0.5) = 0.05,
    gamma: confloat(ge=0.0, le=5.0) = 4.5,
    subsample: confloat(ge=0.5, le=1.0) = 0.9,
    reg_lambda: confloat(ge=1.0, le=10.0) = 8.0,
    reg_alpha: confloat(ge=0.0, le=5.0) = 4.0,
    no_transition_weighting: confloat(ge=0.3, le=0.8) = 0.6,
):
    # initialize wandb project
    wandb.init(project = project, config = locals(), save_code = True)

    # define target transform function
    def target_transform(val):
        return target_transform_partial(val, time_NaN_fill=time_NaN_fill)

    if not DEMO_MODE:
        # load processed data
        # OLTV = pd.read_json("data/processed/processed_OLTV_final.json")
        # bmc_OLTV = pd.read_json("data/processed/processed_bmc_OLTV_final.json")
        if lookahead_hours == 24:
            OLTV = pd.read_json("data/processed/OLTV_24_FINAL.json")
            bmc_OLTV = pd.read_json("data/processed/bmc_OLTV_24_FINAL.json")
            targets = pd.read_json("data/processed/targets_24_FINAL.json")
            bmc_targets = pd.read_json("data/processed/bmc_targets_24_FINAL.json")

        elif lookahead_hours == 8:
            OLTV = pd.read_json("data/processed/OLTV_8_FINAL.json")
            bmc_OLTV = pd.read_json("data/processed/bmc_OLTV_8_FINAL.json")
            targets = pd.read_json("data/processed/targets_8_FINAL.json")
            bmc_targets = pd.read_json("data/processed/bmc_targets_8_FINAL.json")


        OTV = pd.read_json("data/processed/OTV_FINAL.json")
        bmc_OTV = pd.read_json("data/processed/bmc_OTV_FINAL.json")

        mgb_insurance = pd.read_csv("data/raw/mgb_insurance.csv")
        bmc_insurance = pd.read_csv("data/raw/bmc_insurance.csv")

        mgb_medicaid = mgb_insurance[mgb_insurance['medicaid'] == 1]['ptid'].copy()
        bmc_medicaid = bmc_insurance[bmc_insurance['medicaid'] == 1]['ptid'].copy()
        mgb_medicaid = mgb_medicaid.tolist()
        bmc_medicaid = bmc_medicaid.tolist()

        mgb_uninsured = mgb_insurance[mgb_insurance['uninsured'] == 1]['ptid'].copy()
        bmc_uninsured = bmc_insurance[bmc_insurance['uninsured'] == 1]['ptid'].copy()
        mgb_uninsured = mgb_uninsured.tolist()
        bmc_uninsured = bmc_uninsured.tolist()

        mgb_other = mgb_insurance[mgb_insurance['ptid'].isin(mgb_medicaid) == False].copy()
        mgb_other = mgb_other[mgb_other['ptid'].isin(mgb_uninsured) == False]['ptid']
        mgb_other = mgb_other.tolist()

        bmc_other = bmc_insurance[bmc_insurance['ptid'].isin(bmc_medicaid) == False].copy()
        bmc_other = bmc_other[bmc_other['ptid'].isin(bmc_uninsured) == False]['ptid']
        bmc_other = bmc_other.tolist()

        #OLTV.rename(columns = {'targets': 'target'}, inplace = True)
        #bmc_OLTV.rename(columns = {'targets': 'target'}, inplace = True)

        #if lookahead_hours == 24:
            #targets = pd.read_json("data/processed/processed_targets_24hr.json")
            #bmc_targets = pd.read_json("data/processed/processed_bmc_24hr_targets.json")
        
        
        #elif lookahead_hours == 8:
            #targets = pd.read_json("data/processed/processed_targets_8hr.json")
            #bmc_targets = pd.read_json("data/processed/processed_bmc_8hr_targets.json")

        # add targets to data before splitting
        OLTV['target'] = np.array(targets)
        bmc_OLTV['target'] = np.array(bmc_targets)

        # transform targets and prev mls values
        OLTV['target'] = np.array(OLTV['target'].apply(target_transform))
        bmc_OLTV['target'] = np.array(bmc_OLTV['target'].apply(target_transform))

        OLTV['prev_mls'] = np.array(OLTV['prev_mls'].apply(target_transform))
        bmc_OLTV['prev_mls'] = np.array(bmc_OLTV['prev_mls'].apply(target_transform))

        # remove ffill_ from col names
        OLTV.columns = [x.replace("ffill_", "") for x in OLTV.columns]
        bmc_OLTV.columns = [x.replace("ffill_", "") for x in bmc_OLTV.columns]

        # drop cols and rows with too many missings
        def drop_missings(df, col_thresh, row_thesh):
            df.replace(int(-10), np.nan, inplace = True)
            print(f"Number of rows before dropping: {df.shape[0]}")
            print(f"Number of columns before dropping: {df.shape[1]}")
            df = df.dropna(axis = 0, thresh = df.shape[1] * row_thesh)
            df = df.dropna(axis = 1, thresh = df.shape[0] * col_thresh)
            print(f"Number of rows after dropping: {df.shape[0]}")
            print(f"Number of columns after dropping: {df.shape[1]}")
            df.replace(np.nan, int(-10), inplace = True)

            return df
        
        OLTV = drop_missings(OLTV, col_na_drop_fraction, patient_na_drop_fraction)
        bmc_OLTV = drop_missings(bmc_OLTV, col_na_drop_fraction, patient_na_drop_fraction)

    else:
        # load demo data
        if lookahead_hours == 24:
            OLTV = pd.read_json("data/demo/demo_OLTV_24.json")
            bmc_OLTV = pd.read_json("data/demo/demo_bmc_OLTV_24.json")
        
        elif lookahead_hours == 8:
            OLTV = pd.read_json("data/demo/demo_OLTV_8.json")
            bmc_OLTV = pd.read_json("data/demo/demo_bmc_OLTV_8.json")

    # save oltv columns
    OLTV_cols = list(set(OLTV.columns).intersection(set(bmc_OLTV.columns)))
    OLTV = OLTV[OLTV_cols]
    bmc_OLTV = bmc_OLTV[OLTV_cols]

    OLTV_cols = [col for col in OLTV_cols if col not in ['ptid', 'target']]

    # get cv splits and impute
    if not DEMO_MODE:
        cv_train_ptids, cv_test_ptids, cv_train_OLTVs, cv_test_OLTVs, cv_train_OTVs, cv_test_OTVs, cv_train_targets, cv_test_targets = prepare_splits_and_impute(OLTV, OTV, num_folds = num_folds, impute_method = impute_method)
        cv_bmc_train_ptids, cv_bmc_test_ptids, cv_bmc_train_OLTVs, cv_bmc_test_OLTVs, cv_bmc_train_OTVs, cv_bmc_test_OTVs, cv_bmc_train_targets, cv_bmc_test_targets = prepare_splits_and_impute(bmc_OLTV, bmc_OTV, num_folds = num_folds, impute_method = impute_method)

    else:
        cv_train_ptids, cv_test_ptids, cv_train_OLTVs, cv_test_OLTVs, cv_train_targets, cv_test_targets = prepare_splits_and_impute(OLTV, None, num_folds = num_folds, impute_method = impute_method)
        cv_bmc_train_ptids, cv_bmc_test_ptids, cv_bmc_train_OLTVs, cv_bmc_test_OLTVs, cv_bmc_train_targets, cv_bmc_test_targets = prepare_splits_and_impute(bmc_OLTV, None, num_folds = num_folds, impute_method = impute_method)
    
    # convert back to pandas dataframes
    cv_train_OLTVs = [pd.DataFrame(data = x, columns = OLTV_cols) for x in cv_train_OLTVs]
    cv_test_OLTVs = [pd.DataFrame(data = x, columns = OLTV_cols) for x in cv_test_OLTVs]
    cv_bmc_train_OLTVs = [pd.DataFrame(data = x, columns = OLTV_cols) for x in cv_bmc_train_OLTVs]
    cv_bmc_test_OLTVs = [pd.DataFrame(data = x, columns = OLTV_cols) for x in cv_bmc_test_OLTVs]

    if not DEMO_MODE:
        cv_train_OTVs = [pd.DataFrame(data = x, columns = OTV.columns) for x in cv_train_OTVs]
        cv_test_OTVs = [pd.DataFrame(data = x, columns = OTV.columns) for x in cv_test_OTVs]
        cv_bmc_train_OTVs = [pd.DataFrame(data = x, columns = OTV.columns) for x in cv_bmc_train_OTVs]
        cv_bmc_test_OTVs = [pd.DataFrame(data = x, columns = OTV.columns) for x in cv_bmc_test_OTVs]

        # instantiate baseline model metric lists
        ## metrics on mgb test set
        cv_baseline_per_class_accuracy = {"0": [], "0_3": [], "3_8": [], "8_up": []}
        cv_baseline_per_class_auroc = {"0": [], "0_3": [], "3_8": [], "8_up": []}
        cv_baseline_per_class_auprc = {"0": [], "0_3": [], "3_8": [], "8_up": []}

        cv_baseline_per_period_auroc = {"<24": [], "24_48": [], "48_96": [], ">96": []}
        cv_baseline_by_insurance_auroc = {"medicaid": [], "uninsured": [], "other": []}

        cv_baseline_per_class_filtered_accuracy = {"0": [], "0_3": [], "3_8": [], "8_up": []}
        cv_baseline_per_class_filtered_auroc = {"0": [], "0_3": [], "3_8": [], "8_up": []}
        cv_baseline_per_class_filtered_auprc = {"0": [], "0_3": [], "3_8": [], "8_up": []}

        cv_baseline_per_period_filtered_auroc = {"<24": [], "24_48": [], "48_96": [], ">96": []}
        cv_baseline_by_insurance_filtered_auroc = {"medicaid": [], "uninsured": [], "other": []}

        cv_baseline_overall_accuracy = []
        cv_baseline_overall_auroc = []
        cv_baseline_overall_auprc = []

        cv_baseline_filtered_accuracy = []
        cv_baseline_filtered_auroc = []
        cv_baseline_filtered_auprc = []

        cv_baseline_overall_prc_data = []
        cv_baseline_overall_roc_data = []
        
        cv_baseline_filtered_prc_data = []
        cv_baseline_filtered_roc_data = []

        ## metrics on bmc test set
        cv_baseline_bmc_per_class_accuracy = {"0": [], "0_3": [], "3_8": [], "8_up": []}
        cv_baseline_bmc_per_class_auroc = {"0": [], "0_3": [], "3_8": [], "8_up": []}
        cv_baseline_bmc_per_class_auprc = {"0": [], "0_3": [], "3_8": [], "8_up": []}

        cv_baseline_bmc_per_period_auroc = {"<24": [], "24_48": [], "48_96": [], ">96": []}
        cv_baseline_bmc_by_insurance_auroc = {"medicaid": [], "uninsured": [], "other": []}

        cv_baseline_bmc_per_class_filtered_accuracy = {"0": [], "0_3": [], "3_8": [], "8_up": []}
        cv_baseline_bmc_per_class_filtered_auroc = {"0": [], "0_3": [], "3_8": [], "8_up": []}
        cv_baseline_bmc_per_class_filtered_auprc = {"0": [], "0_3": [], "3_8": [], "8_up": []}

        cv_baseline_bmc_per_period_filtered_auroc = {"<24": [], "24_48": [], "48_96": [], ">96": []}
        cv_baseline_bmc_by_insurance_filtered_auroc = {"medicaid": [], "uninsured": [], "other": []}

        cv_baseline_bmc_overall_accuracy = []
        cv_baseline_bmc_overall_auroc = []
        cv_baseline_bmc_overall_auprc = []

        cv_baseline_bmc_filtered_accuracy = []
        cv_baseline_bmc_filtered_auroc = []
        cv_baseline_bmc_filtered_auprc = []

        cv_baseline_bmc_overall_prc_data = []
        cv_baseline_bmc_overall_roc_data = []

        cv_baseline_bmc_filtered_prc_data = []
        cv_baseline_bmc_filtered_roc_data = []

    # instantiate XGB performance metric lists 
    ## metrics on mgb training set to test for overfitting
    cv_xgb_training_per_class_accuracy = {"0": [], "0_3": [], "3_8": [], "8_up": []}
    cv_xgb_training_per_class_auroc = {"0": [], "0_3": [], "3_8": [], "8_up": []}
    cv_xgb_training_per_class_auprc = {"0": [], "0_3": [], "3_8": [], "8_up": []}

    cv_xgb_training_per_class_filtered_accuracy = {"0": [], "0_3": [], "3_8": [], "8_up": []}
    cv_xgb_training_per_class_filtered_auroc = {"0": [], "0_3": [], "3_8": [], "8_up": []}
    cv_xgb_training_per_class_filtered_auprc = {"0": [], "0_3": [], "3_8": [], "8_up": []}

    cv_xgb_training_overall_accuracy = []
    cv_xgb_training_overall_auroc = []
    cv_xgb_training_overall_auprc = []

    cv_xgb_training_filtered_accuracy = []
    cv_xgb_training_filtered_auroc = []
    cv_xgb_training_filtered_auprc = []

    ## metrics on mgb test set
    cv_xgb_per_class_accuracy = {"0": [], "0_3": [], "3_8": [], "8_up": []}
    cv_xgb_per_class_auroc = {"0": [], "0_3": [], "3_8": [], "8_up": []}
    cv_xgb_per_class_auprc = {"0": [], "0_3": [], "3_8": [], "8_up": []}

    cv_xgb_per_period_auroc = {"<24": [], "24_48": [], "48_96": [], ">96": []}
    cv_xgb_by_insurance_auroc = {"medicaid": [], "uninsured": [], "other": []}

    cv_xgb_per_class_filtered_accuracy = {"0": [], "0_3": [], "3_8": [], "8_up": []}
    cv_xgb_per_class_filtered_auroc = {"0": [], "0_3": [], "3_8": [], "8_up": []}
    cv_xgb_per_class_filtered_auprc = {"0": [], "0_3": [], "3_8": [], "8_up": []}

    cv_xgb_per_period_filtered_auroc = {"<24": [], "24_48": [], "48_96": [], ">96": []}
    cv_xgb_by_insurance_filtered_auroc = {"medicaid": [], "uninsured": [], "other": []}

    cv_xgb_overall_accuracy = []
    cv_xgb_overall_auroc = []
    cv_xgb_overall_auprc = []

    cv_xgb_filtered_accuracy = []
    cv_xgb_filtered_auroc = []
    cv_xgb_filtered_auprc = []

    cv_xgb_overall_prc_data = []
    cv_xgb_overall_roc_data = []

    cv_xgb_filtered_prc_data = []
    cv_xgb_filtered_roc_data = []

    ## metrics on bmc set
    cv_xgb_bmc_per_class_accuracy = {"0": [], "0_3": [], "3_8": [], "8_up": []}
    cv_xgb_bmc_per_class_auroc = {"0": [], "0_3": [], "3_8": [], "8_up": []}
    cv_xgb_bmc_per_class_auprc = {"0": [], "0_3": [], "3_8": [], "8_up": []}

    cv_xgb_bmc_per_period_auroc = {"<24": [], "24_48": [], "48_96": [], ">96": []}
    cv_xgb_bmc_by_insurance_auroc = {"medicaid": [], "uninsured": [], "other": []}

    cv_xgb_bmc_per_class_filtered_accuracy = {"0": [], "0_3": [], "3_8": [], "8_up": []}
    cv_xgb_bmc_per_class_filtered_auroc = {"0": [], "0_3": [], "3_8": [], "8_up": []}
    cv_xgb_bmc_per_class_filtered_auprc = {"0": [], "0_3": [], "3_8": [], "8_up": []}

    cv_xgb_bmc_per_period_filtered_auroc = {"<24": [], "24_48": [], "48_96": [], ">96": []}
    cv_xgb_bmc_by_insurance_filtered_auroc = {"medicaid": [], "uninsured": [], "other": []}

    cv_xgb_bmc_overall_accuracy = []
    cv_xgb_bmc_overall_auroc = []
    cv_xgb_bmc_overall_auprc = []

    cv_xgb_bmc_filtered_accuracy = []
    cv_xgb_bmc_filtered_auroc = []
    cv_xgb_bmc_filtered_auprc = []

    cv_xgb_bmc_overall_prc_data = []
    cv_xgb_bmc_overall_roc_data = []

    cv_xgb_bmc_filtered_prc_data = []
    cv_xgb_bmc_filtered_roc_data = []

    # define metric calculating functions
    ## per class metrics
    class_accuracy = MulticlassAccuracy(num_classes=4, average="none")
    class_AUROC = MulticlassAUROC(num_classes=4, average = "none")
    class_AUPRC = MulticlassAveragePrecision(num_classes=4, average = "none")

    ## weighted average metrics
    mean_accuracy = MulticlassAccuracy(num_classes=4, average = "weighted")
    mean_AUROC = MulticlassAUROC(num_classes=4, average = "weighted")
    mean_AUPRC = MulticlassAveragePrecision(num_classes=4, average = "weighted")

    # iterate through cv splits and run model training on each set
    if not DEMO_MODE:
        cv_num = 0
        for(
            train_ptids,
            test_ptids,
            train_OLTV,
            test_OLTV,
            train_OTV,
            test_OTV,
            train_targets,
            test_targets,
            bmc_train_ptids,
            bmc_test_ptids,
            bmc_train_OLTV,
            bmc_test_OLTV,
            bmc_train_OTV,
            bmc_test_OTV,
            bmc_train_targets,
            bmc_test_targets,
        ) in zip(
            cv_train_ptids,
            cv_test_ptids, 
            cv_train_OLTVs,
            cv_test_OLTVs,
            cv_train_OTVs,
            cv_test_OTVs,
            cv_train_targets,
            cv_test_targets,
            cv_bmc_train_ptids,
            cv_bmc_test_ptids,
            cv_bmc_train_OLTVs,
            cv_bmc_test_OLTVs,
            cv_bmc_train_OTVs,
            cv_bmc_test_OTVs,
            cv_bmc_train_targets,
            cv_bmc_test_targets,
        ):
            train_ptids = np.array(train_ptids)
            test_ptids = np.array(test_ptids)
            
            ## save OLTV columns
            OLTV_columns = train_OLTV.columns.copy()

            ## find patient-hours where transition occurs for sample weighting and filtered metrics
            ### apply target transform to mls value
            train_mls_transformed = np.array(train_OLTV['size_mls'].apply(target_transform))
            test_mls_transformed = np.array(test_OLTV['size_mls'].apply(target_transform))
            bmc_train_mls_transformed = np.array(bmc_train_OLTV['size_mls'].apply(target_transform))
            bmc_test_mls_transformed = np.array(bmc_test_OLTV['size_mls'].apply(target_transform))

            ### filter datasets where current mls class != target class
            filtered_train_OLTV = train_OLTV[train_mls_transformed != train_targets]
            filtered_train_targets = train_targets[train_mls_transformed != train_targets]
            filtered_train_ptids = train_ptids[train_mls_transformed != train_targets]

            filtered_test_OLTV = test_OLTV[test_mls_transformed != test_targets]
            filtered_test_targets = test_targets[test_mls_transformed != test_targets]
            filtered_test_ptids = test_ptids[test_mls_transformed != test_targets]

            filtered_bmc_train_OLTV = bmc_train_OLTV[bmc_train_mls_transformed != bmc_train_targets]
            filtered_bmc_train_targets = bmc_train_targets[bmc_train_mls_transformed != bmc_train_targets]
            filtered_bmc_train_ptids = bmc_train_ptids[bmc_train_mls_transformed != bmc_train_targets]

            filtered_bmc_test_OLTV = bmc_test_OLTV[bmc_test_mls_transformed != bmc_test_targets]
            filtered_bmc_test_targets = bmc_test_targets[bmc_test_mls_transformed != bmc_test_targets]
            filtered_bmc_test_ptids = bmc_test_ptids[bmc_test_mls_transformed != bmc_test_targets]

            ### create sample weighting for training 
            train_sample_weights = np.ones(train_OLTV.shape[0])
            train_sample_weights[train_mls_transformed == train_targets] = no_transition_weighting

            ## Method to create linear regression baseline model and log metrics
            @validate_call
            def create_edema_score_baseline(
                x_train_ptids, x_test_ptids,
                x_train_OLTV, x_test_OLTV,
                x_train_OTV, x_test_OTV,
                x_train_targets, x_test_targets,
                type: Literal["mgb", "bmc"], 
                cv_num,
            ):
                ### Method to create dataframes for linear regression edema score baseline (as defined by previous literature)
                def generate_edema_score_df(OLTV_df, OTV_df, ptids, targets):
                    #### fixed variables from OTV
                    df = OTV_df[['ptid', 'gluc1', 'stroke', 'hba1c', 'tpa', 'mt']] # variables taken from previous literature
                    
                    #### mls variable from OLTV
                    mls_df = pd.DataFrame(OLTV_df[['dt', 'size_mls']], columns = ['dt', 'size_mls'])
                    mls_df['ptid'] = np.array(ptids)
                    mls_df['target'] = np.array(targets)
                    df = df.merge(mls_df, on = "ptid", how = "inner")
                    
                    new_targets = np.array(df['target'].copy())
                    new_ptids = np.array(df['ptid'].copy())

                    #### create filtered dataset
                    transformed_mls_vals = np.array(df['size_mls'].copy().apply(target_transform))
                    
                    filtered_df = df[transformed_mls_vals != new_targets].copy()
                    filtered_targets = new_targets[transformed_mls_vals != new_targets].copy()
                    filtered_ptids = new_ptids[transformed_mls_vals != new_targets].copy()

                    df = df.drop(columns = ['target', 'ptid'])
                    filtered_df = filtered_df.drop(columns = ['target', 'ptid'])

                    return df, new_targets, new_ptids, filtered_df, filtered_targets, filtered_ptids
                
                ### apply method to make dataframes for regression edema score
                edema_train_df, edema_train_targets, edema_train_ptids, filtered_edema_train_df, filtered_edema_train_targets, filtered_edema_train_ptids = generate_edema_score_df(x_train_OLTV, x_train_OTV, x_train_ptids, x_train_targets)
                edema_test_df, edema_test_targets, edema_test_ptids, filtered_edema_test_df, filtered_edema_test_targets, filtered_edema_test_ptids = generate_edema_score_df(x_test_OLTV, x_test_OTV, x_test_ptids, x_test_targets)
                
                ### impute missing values in edema score dataframes
                imputer = SimpleImputer()
                imputer.fit(edema_train_df)

                cols = list(edema_train_df.columns)
                
                edema_train_df = imputer.transform(edema_train_df)
                edema_test_df = imputer.transform(edema_test_df)
                
                if len(filtered_edema_test_df) > 0:
                    filtered_edema_test_df = imputer.transform(filtered_edema_test_df)
                    filtered_edema_test_df = pd.DataFrame(filtered_edema_test_df, columns = cols)
                else:
                    filtered_edema_test_df = pd.DataFrame(columns = cols)

                edema_train_df = pd.DataFrame(edema_train_df, columns = cols)
                edema_test_df = pd.DataFrame(edema_test_df, columns = cols)

                ### filter by time periods
                list_24 = np.array(edema_test_df['dt'] < 24)
                list_24_48 = np.array(((edema_test_df['dt'] >= 24) & (edema_test_df['dt'] < 48)))
                list_48_96 = np.array(((edema_test_df['dt'] >= 48) & (edema_test_df['dt'] < 96)))
                list_96 = np.array(edema_test_df['dt'] >= 96)

                list_filter_24 = np.array(filtered_edema_test_df['dt'] < 24)
                list_filter_24_48 = np.array(((filtered_edema_test_df['dt'] >= 24) & (filtered_edema_test_df['dt'] < 48)))
                list_filter_48_96 = np.array(((filtered_edema_test_df['dt'] >= 48) & (filtered_edema_test_df['dt'] < 96)))
                list_filter_96 = np.array(filtered_edema_test_df['dt'] >= 96)

                edema_test_targets_24 = edema_test_targets[list_24]
                edema_test_targets_24_48 = edema_test_targets[list_24_48]
                edema_test_targets_48_96 = edema_test_targets[list_48_96]
                edema_test_targets_96 = edema_test_targets[list_96]

                filtered_edema_test_targets_24 = filtered_edema_test_targets[list_filter_24]
                filtered_edema_test_targets_24_48 = filtered_edema_test_targets[list_filter_24_48]
                filtered_edema_test_targets_48_96 = filtered_edema_test_targets[list_filter_48_96]
                filtered_edema_test_targets_96 = filtered_edema_test_targets[list_filter_96]

                ### drop dt column from dataframes
                edema_train_df = edema_train_df.drop(columns = ['dt'])
                edema_test_df = edema_test_df.drop(columns = ['dt'])
                filtered_edema_test_df = filtered_edema_test_df.drop(columns = ['dt'])

                ### create insurance dataframes
                if type == "mgb":
                    medicaid = mgb_medicaid
                    uninsured = mgb_uninsured
                    other = mgb_other
                elif type == "bmc":
                    medicaid = bmc_medicaid
                    uninsured = bmc_uninsured
                    other = bmc_other

                edema_test_df_temp = edema_test_df.copy()
                edema_test_df_temp['ptid'] = np.array(edema_test_ptids.copy())
                edema_test_df_temp['target'] = np.array(edema_test_targets.copy())
                
                filtered_edema_test_df_temp = filtered_edema_test_df.copy()
                filtered_edema_test_df_temp['ptid'] = np.array(filtered_edema_test_ptids.copy())
                filtered_edema_test_df_temp['target'] = np.array(filtered_edema_test_targets.copy())

                edema_test_df_medicaid = edema_test_df_temp[edema_test_df_temp['ptid'].isin(medicaid)]
                edema_test_df_uninsured = edema_test_df_temp[edema_test_df_temp['ptid'].isin(uninsured)]
                edema_test_df_other = edema_test_df_temp[edema_test_df_temp['ptid'].isin(other)]

                filtered_edema_test_df_medicaid = filtered_edema_test_df_temp[filtered_edema_test_df_temp['ptid'].isin(medicaid)]
                filtered_edema_test_df_uninsured = filtered_edema_test_df_temp[filtered_edema_test_df_temp['ptid'].isin(uninsured)]
                filtered_edema_test_df_other = filtered_edema_test_df_temp[filtered_edema_test_df_temp['ptid'].isin(other)]

                edema_test_ptids_medicaid = edema_test_df_medicaid['ptid'].copy()
                edema_test_ptids_uninsured = edema_test_df_uninsured['ptid'].copy()
                edema_test_ptids_other = edema_test_df_other['ptid'].copy()

                filtered_edema_test_ptids_medicaid = filtered_edema_test_df_medicaid['ptid'].copy()
                filtered_edema_test_ptids_uninsured = filtered_edema_test_df_uninsured['ptid'].copy()
                filtered_edema_test_ptids_other = filtered_edema_test_df_other['ptid'].copy()

                edema_test_targets_medicaid = edema_test_df_medicaid['target'].copy()
                edema_test_targets_uninsured = edema_test_df_uninsured['target'].copy()
                edema_test_targets_other = edema_test_df_other['target'].copy()

                filtered_edema_test_targets_medicaid = filtered_edema_test_df_medicaid['target'].copy()
                filtered_edema_test_targets_uninsured = filtered_edema_test_df_uninsured['target'].copy()
                filtered_edema_test_targets_other = filtered_edema_test_df_other['target'].copy()

                edema_test_df_medicaid = edema_test_df_medicaid.drop(columns = ['ptid', 'target'])
                edema_test_df_uninsured = edema_test_df_uninsured.drop(columns = ['ptid', 'target'])
                edema_test_df_other = edema_test_df_other.drop(columns = ['ptid', 'target'])

                filtered_edema_test_df_medicaid = filtered_edema_test_df_medicaid.drop(columns = ['ptid', 'target'])
                filtered_edema_test_df_uninsured = filtered_edema_test_df_uninsured.drop(columns = ['ptid', 'target'])
                filtered_edema_test_df_other = filtered_edema_test_df_other.drop(columns = ['ptid', 'target'])

                ### fit regression model and calculate class probabilities for filtered and unfiltered datasets
                edema_score_model = LogisticRegression().fit(edema_train_df, edema_train_targets)

                baseline_probas = edema_score_model.predict_proba(edema_test_df)
                baseline_predict = edema_score_model.predict(edema_test_df)

                baseline_filtered_probas = edema_score_model.predict_proba(filtered_edema_test_df)
                baseline_filtered_predict = edema_score_model.predict(filtered_edema_test_df)

                #### by insurance 
                if len(edema_test_df_medicaid) > 0:
                    baseline_probas_medicaid = edema_score_model.predict_proba(edema_test_df_medicaid)
                    baseline_predict_medicaid = edema_score_model.predict(edema_test_df_medicaid)
                else:
                    baseline_probas_medicaid = []
                    baseline_predict_medicaid = []
                
                if len(edema_test_df_uninsured) > 0:
                    baseline_probas_uninsured = edema_score_model.predict_proba(edema_test_df_uninsured)
                    baseline_predict_uninsured = edema_score_model.predict(edema_test_df_uninsured)
                else:
                    baseline_probas_uninsured = []
                    baseline_predict_uninsured = []

                if len(edema_test_df_other) > 0:
                    baseline_probas_other = edema_score_model.predict_proba(edema_test_df_other)
                    baseline_predict_other = edema_score_model.predict(edema_test_df_other)
                else:
                    baseline_probas_other = []
                    baseline_predict_other = []


                if len(filtered_edema_test_df_medicaid) > 0:
                    baseline_filtered_probas_medicaid = edema_score_model.predict_proba(filtered_edema_test_df_medicaid)
                    baseline_filtered_predict_medicaid = edema_score_model.predict(filtered_edema_test_df_medicaid)
                else:
                    baseline_filtered_probas_medicaid = []
                    baseline_filtered_predict_medicaid = []

                if len(filtered_edema_test_df_uninsured) > 0:
                    baseline_filtered_probas_uninsured = edema_score_model.predict_proba(filtered_edema_test_df_uninsured)
                    baseline_filtered_predict_uninsured = edema_score_model.predict(filtered_edema_test_df_uninsured)
                else:
                    baseline_filtered_probas_uninsured = []
                    baseline_filtered_predict_uninsured = []

                if len(filtered_edema_test_df_other) > 0:
                    baseline_filtered_probas_other = edema_score_model.predict_proba(filtered_edema_test_df_other)
                    baseline_filtered_predict_other = edema_score_model.predict(filtered_edema_test_df_other)
                else:
                    baseline_filtered_probas_other = []
                    baseline_filtered_predict_other = []

                # by time period
                baseline_probas_24 = baseline_probas[list_24]
                baseline_probas_24_48 = baseline_probas[list_24_48]
                baseline_probas_48_96 = baseline_probas[list_48_96]
                baseline_probas_96 = baseline_probas[list_96]

                baseline_filtered_probas_24 = baseline_filtered_probas[list_filter_24]
                baseline_filtered_probas_24_48 = baseline_filtered_probas[list_filter_24_48]
                baseline_filtered_probas_48_96 = baseline_filtered_probas[list_filter_48_96]
                baseline_filtered_probas_96 = baseline_filtered_probas[list_filter_96]

                baseline_predict_24 = baseline_predict[list_24]
                baseline_predict_24_48 = baseline_predict[list_24_48]
                baseline_predict_48_96 = baseline_predict[list_48_96]
                baseline_predict_96 = baseline_predict[list_96]

                baseline_filtered_predict_24 = baseline_filtered_predict[list_filter_24]
                baseline_filtered_predict_24_48 = baseline_filtered_predict[list_filter_24_48]
                baseline_filtered_predict_48_96 = baseline_filtered_predict[list_filter_48_96]
                baseline_filtered_predict_96 = baseline_filtered_predict[list_filter_96]

                ### method to evaluate metrics for baseline models
                def eval_baseline_probas(probas, predicts, filt_probas, filt_predicts, targets, filt_targets, all_hours = True):
                    if not all_hours and ((len(probas) == 0) or (len(filt_probas) == 0)):
                        return 0, 0
                    
                    targets = targets.astype(int)
                    filt_targets = filt_targets.astype(int)

                    #### convert to tensors
                    y_probas = tnsr(probas).float()
                    y_predict = tnsr(predicts)

                    if isinstance(targets, pd.DataFrame) or isinstance(targets, pd.Series):
                        y_true = tnsr(targets.to_numpy().flatten())
                    else:
                        y_true = tnsr(targets.flatten())

                    filtered_y_probas = tnsr(filt_probas).float()
                    filtered_y_predict = tnsr(filt_predicts)

                    if isinstance(filt_targets, pd.DataFrame) or isinstance(filt_targets, pd.Series):
                        filtered_y_true = tnsr(filt_targets.to_numpy().flatten())
                    else:
                        filtered_y_true = tnsr(filt_targets.flatten())

                    ### calculate per-class metrics for logistic regression baseline
                    #### unfiltered (overall)
                    if all_hours:
                        baseline_class_accuracies = class_accuracy(y_predict, y_true).tolist()
                        baseline_class_aurocs = class_AUROC(y_probas, y_true).tolist()
                        baseline_class_auprcs = class_AUPRC(y_probas, y_true).tolist()

                        #### filtered (only when transition occurs)
                        baseline_filtered_class_accuracies = class_accuracy(filtered_y_predict, filtered_y_true).tolist()
                        baseline_filtered_class_aurocs = class_AUROC(filtered_y_probas, filtered_y_true).tolist()
                        baseline_filtered_class_auprcs = class_AUPRC(filtered_y_probas, filtered_y_true).tolist()
                        
                        ### calculate weighted mean metrics 
                        #### unfiltered (overall)
                        baseline_mean_accuracy = mean_accuracy(y_predict, y_true).item()
                        baseline_mean_AUROC = mean_AUROC(y_probas, y_true).item()
                        baseline_mean_AUPRC = mean_AUPRC(y_probas, y_true).item()

                        #### filtered (only when transition occurs)
                        baseline_filtered_mean_accuracy = mean_accuracy(filtered_y_predict, filtered_y_true).item()
                        baseline_filtered_mean_AUROC = mean_AUROC(filtered_y_probas, filtered_y_true).item()
                        baseline_filtered_mean_AUPRC = mean_AUPRC(filtered_y_probas, filtered_y_true).item()

                        ### convert to numpy arrays
                        y_probas = np.array(y_probas.tolist())
                        y_true = np.array(y_true.tolist())
                        filtered_y_probas = np.array(filtered_y_probas.tolist())
                        filtered_y_true = np.array(filtered_y_true.tolist())
                        
                        ### generate plot data and save curve plots
                        prc_data, roc_data = get_plot_data(y_true, y_probas, cv_num = cv_num, type = type, model = "baseline")
                        filtered_prc_data, filtered_roc_data = get_plot_data(filtered_y_true, filtered_y_probas, cv_num = cv_num, type = type, model = "baseline", filtered = True)
                        
                        return baseline_class_accuracies, baseline_class_aurocs, baseline_class_auprcs, baseline_mean_accuracy, baseline_mean_AUROC, baseline_mean_AUPRC, prc_data, roc_data, filtered_prc_data, filtered_roc_data, baseline_filtered_class_accuracies, baseline_filtered_class_aurocs, baseline_filtered_class_auprcs, baseline_filtered_mean_accuracy, baseline_filtered_mean_AUROC, baseline_filtered_mean_AUPRC

                    else:
                        baseline_mean_AUROC = mean_AUROC(y_probas, y_true).item()
                        baseline_filtered_mean_AUROC = mean_AUROC(filtered_y_probas, filtered_y_true).item()

                        return baseline_mean_AUROC, baseline_filtered_mean_AUROC

                ### calculate metrics for unfiltered and filtered datasets
                baseline_class_accuracies, baseline_class_aurocs, baseline_class_auprcs, baseline_mean_accuracy, baseline_mean_AUROC, baseline_mean_AUPRC, prc_data, roc_data, filtered_prc_data, filtered_roc_data, baseline_filtered_class_accuracies, baseline_filtered_class_aurocs, baseline_filtered_class_auprcs, baseline_filtered_mean_accuracy, baseline_filtered_mean_AUROC, baseline_filtered_mean_AUPRC = eval_baseline_probas(
                    baseline_probas, baseline_predict, baseline_filtered_probas, baseline_filtered_predict, edema_test_targets, filtered_edema_test_targets, all_hours = True
                )

                ### calculate metrics for by insurance datasets
                baseline_mean_AUROC_medicaid, baseline_filtered_mean_AUROC_medicaid = eval_baseline_probas(
                    baseline_probas_medicaid, baseline_predict_medicaid, baseline_filtered_probas_medicaid, baseline_filtered_predict_medicaid, edema_test_targets_medicaid, filtered_edema_test_targets_medicaid, all_hours = False
                )

                baseline_mean_AUROC_uninsured, baseline_filtered_mean_AUROC_uninsured = eval_baseline_probas(
                    baseline_probas_uninsured, baseline_predict_uninsured, baseline_filtered_probas_uninsured, baseline_filtered_predict_uninsured, edema_test_targets_uninsured, filtered_edema_test_targets_uninsured, all_hours = False
                )

                baseline_mean_AUROC_other, baseline_filtered_mean_AUROC_other = eval_baseline_probas(
                    baseline_probas_other, baseline_predict_other, baseline_filtered_probas_other, baseline_filtered_predict_other, edema_test_targets_other, filtered_edema_test_targets_other, all_hours = False
                )

                ### calculate metrics for by time period datasets
                baseline_mean_AUROC_24, baseline_filtered_mean_AUROC_24 = eval_baseline_probas(
                    baseline_probas_24, baseline_predict_24, baseline_filtered_probas_24, baseline_filtered_predict_24, edema_test_targets_24, filtered_edema_test_targets_24, all_hours = False
                )

                baseline_mean_AUROC_24_48, baseline_filtered_mean_AUROC_24_48 = eval_baseline_probas(
                    baseline_probas_24_48, baseline_predict_24_48, baseline_filtered_probas_24_48, baseline_filtered_predict_24_48, edema_test_targets_24_48, filtered_edema_test_targets_24_48, all_hours = False
                )

                baseline_mean_AUROC_48_96, baseline_filtered_mean_AUROC_48_96 = eval_baseline_probas(
                    baseline_probas_48_96, baseline_predict_48_96, baseline_filtered_probas_48_96, baseline_filtered_predict_48_96, edema_test_targets_48_96, filtered_edema_test_targets_48_96, all_hours = False
                )

                baseline_mean_AUROC_96, baseline_filtered_mean_AUROC_96 = eval_baseline_probas(
                    baseline_probas_96, baseline_predict_96, baseline_filtered_probas_96, baseline_filtered_predict_96, edema_test_targets_96, filtered_edema_test_targets_96, all_hours = False
                )


                ### Log metrics using previously instantiated lists
                if type == "mgb":
                    #### log per-class metrics 
                    keys = list(cv_baseline_per_class_accuracy.keys())
                    for i in range(4):
                        cv_baseline_per_class_accuracy[keys[i]].append(baseline_class_accuracies[i])
                        cv_baseline_per_class_auroc[keys[i]].append(baseline_class_aurocs[i])
                        cv_baseline_per_class_auprc[keys[i]].append(baseline_class_auprcs[i])

                        cv_baseline_per_class_filtered_accuracy[keys[i]].append(baseline_filtered_class_accuracies[i])
                        cv_baseline_per_class_filtered_auroc[keys[i]].append(baseline_filtered_class_aurocs[i])
                        cv_baseline_per_class_filtered_auprc[keys[i]].append(baseline_filtered_class_auprcs[i])

                    #### log per-period mean metrics
                    hour_keys = list(cv_baseline_per_period_auroc.keys())
                    cv_baseline_per_period_auroc[hour_keys[0]].append(baseline_mean_AUROC_24)
                    cv_baseline_per_period_auroc[hour_keys[1]].append(baseline_mean_AUROC_24_48)
                    cv_baseline_per_period_auroc[hour_keys[2]].append(baseline_mean_AUROC_48_96)
                    cv_baseline_per_period_auroc[hour_keys[3]].append(baseline_mean_AUROC_96)

                    #### log mean metrics by insurance
                    insurance_keys = list(cv_baseline_by_insurance_auroc.keys())
                    cv_baseline_by_insurance_auroc[insurance_keys[0]].append(baseline_mean_AUROC_medicaid)
                    cv_baseline_by_insurance_auroc[insurance_keys[1]].append(baseline_mean_AUROC_uninsured)
                    cv_baseline_by_insurance_auroc[insurance_keys[2]].append(baseline_mean_AUROC_other)
                    
                    #### log weighted mean metrics
                    cv_baseline_overall_accuracy.append(baseline_mean_accuracy)
                    cv_baseline_overall_auroc.append(baseline_mean_AUROC)
                    cv_baseline_overall_auprc.append(baseline_mean_AUPRC)

                    cv_baseline_filtered_accuracy.append(baseline_filtered_mean_accuracy)
                    cv_baseline_filtered_auroc.append(baseline_filtered_mean_AUROC)
                    cv_baseline_filtered_auprc.append(baseline_filtered_mean_AUPRC)

                    #### log plot data
                    cv_baseline_overall_prc_data.append(prc_data)
                    cv_baseline_overall_roc_data.append(roc_data)

                    cv_baseline_filtered_prc_data.append(filtered_prc_data)
                    cv_baseline_filtered_roc_data.append(filtered_roc_data)


                ### Repeat for base of type = bmc
                elif type == "bmc":
                    keys = list(cv_baseline_bmc_per_class_accuracy.keys())
                    for i in range(4):
                        cv_baseline_bmc_per_class_accuracy[keys[i]].append(baseline_class_accuracies[i])
                        cv_baseline_bmc_per_class_auroc[keys[i]].append(baseline_class_aurocs[i])
                        cv_baseline_bmc_per_class_auprc[keys[i]].append(baseline_class_auprcs[i])

                        cv_baseline_bmc_per_class_filtered_accuracy[keys[i]].append(baseline_filtered_class_accuracies[i])
                        cv_baseline_bmc_per_class_filtered_auroc[keys[i]].append(baseline_filtered_class_aurocs[i])
                        cv_baseline_bmc_per_class_filtered_auprc[keys[i]].append(baseline_filtered_class_auprcs[i])
                    
                    #### log per-period mean metrics
                    hour_keys = list(cv_baseline_bmc_per_period_auroc.keys())
                    cv_baseline_bmc_per_period_auroc[hour_keys[0]].append(baseline_mean_AUROC_24)
                    cv_baseline_bmc_per_period_auroc[hour_keys[1]].append(baseline_mean_AUROC_24_48)
                    cv_baseline_bmc_per_period_auroc[hour_keys[2]].append(baseline_mean_AUROC_48_96)
                    cv_baseline_bmc_per_period_auroc[hour_keys[3]].append(baseline_mean_AUROC_96)

                    #### log mean metrics by insurance
                    insurance_keys = list(cv_baseline_bmc_by_insurance_auroc.keys())
                    cv_baseline_bmc_by_insurance_auroc[insurance_keys[0]].append(baseline_mean_AUROC_medicaid)
                    cv_baseline_bmc_by_insurance_auroc[insurance_keys[1]].append(baseline_mean_AUROC_uninsured)
                    cv_baseline_bmc_by_insurance_auroc[insurance_keys[2]].append(baseline_mean_AUROC_other)

                    #### log weighted mean metrics
                    cv_baseline_bmc_overall_accuracy.append(baseline_mean_accuracy)
                    cv_baseline_bmc_overall_auroc.append(baseline_mean_AUROC)
                    cv_baseline_bmc_overall_auprc.append(baseline_mean_AUPRC)

                    cv_baseline_bmc_filtered_accuracy.append(baseline_filtered_mean_accuracy)
                    cv_baseline_bmc_filtered_auroc.append(baseline_filtered_mean_AUROC)
                    cv_baseline_bmc_filtered_auprc.append(baseline_filtered_mean_AUPRC)

                    #### log plot data
                    cv_baseline_bmc_overall_prc_data.append(prc_data)
                    cv_baseline_bmc_overall_roc_data.append(roc_data)

                    cv_baseline_bmc_filtered_prc_data.append(filtered_prc_data)
                    cv_baseline_bmc_filtered_roc_data.append(filtered_roc_data)

                ### Log metrics to wandb
                for i in range(4):
                    wandb.log({
                        f"{type} edema score baseline accuracy (class {i}) for cv {cv_num}": baseline_class_accuracies[i],
                        f"{type} edema score baseline AUROC (class {i}) for cv {cv_num}": baseline_class_aurocs[i],
                        f"{type} edema score baseline AUPRC (class {i}) for cv {cv_num}": baseline_class_auprcs[i],
                        f"filtered {type} edema score baseline accuracy (class {i}) for cv {cv_num}": baseline_filtered_class_accuracies[i],
                        f"filtered {type} edema score baseline AUROC (class {i}) for cv {cv_num}": baseline_filtered_class_aurocs[i],
                        f"filtered {type} edema score baseline AUPRC (class {i}) for cv {cv_num}": baseline_filtered_class_auprcs[i]
                    })

                wandb.log({
                    f"{type} edema score baseline mean AUROC for period < 24 for cv {cv_num}": baseline_mean_AUROC_24,
                    f"filtered {type} edema score baseline mean AUROC for period < 24 for cv {cv_num}": baseline_filtered_mean_AUROC_24,
                    f"{type} edema score baseline mean AUROC for period 24-48 for cv {cv_num}": baseline_mean_AUROC_24_48,
                    f"filtered {type} edema score baseline mean AUROC for period 24-48 for cv {cv_num}": baseline_filtered_mean_AUROC_24_48,
                    f"{type} edema score baseline mean AUROC for period 48-96 for cv {cv_num}": baseline_mean_AUROC_48_96,
                    f"filtered {type} edema score baseline mean AUROC for period 48-96 for cv {cv_num}": baseline_filtered_mean_AUROC_48_96,
                    f"{type} edema score baseline mean AUROC for period > 96 for cv {cv_num}": baseline_mean_AUROC_96,
                    f"filtered {type} edema score baseline mean AUROC for period > 96 for cv {cv_num}": baseline_filtered_mean_AUROC_96
                })

                wandb.log({
                    f"{type} edema score baseline mean AUROC for medicaid for cv {cv_num}": baseline_mean_AUROC_medicaid,
                    f"filtered {type} edema score baseline mean AUROC for medicaid for cv {cv_num}": baseline_filtered_mean_AUROC_medicaid,
                    f"{type} edema score baseline mean AUROC for uninsured for cv {cv_num}": baseline_mean_AUROC_uninsured,
                    f"filtered {type} edema score baseline mean AUROC for uninsured for cv {cv_num}": baseline_filtered_mean_AUROC_uninsured,
                    f"{type} edema score baseline mean AUROC for other for cv {cv_num}": baseline_mean_AUROC_other,
                    f"filtered {type} edema score baseline mean AUROC for other for cv {cv_num}": baseline_filtered_mean_AUROC_other
                })

                wandb.log({
                    f"{type} edema score baseline mean accuracy for cv {cv_num}": baseline_mean_accuracy,
                    f"{type} edema score baseline mean AUROC for cv {cv_num}": baseline_mean_AUROC,
                    f"{type} edema score baseline mean AUPRC for cv {cv_num}": baseline_mean_AUPRC,
                    f"filtered {type} edema score baseline mean accuracy for cv {cv_num}": baseline_filtered_mean_accuracy,
                    f"filtered {type} edema score baseline mean AUROC for cv {cv_num}": baseline_filtered_mean_AUROC,
                    f"filtered {type} edema score baseline mean AUPRC for cv {cv_num}": baseline_filtered_mean_AUPRC
                })

                print(f"Done logging baseline regression metrics for {type} on cv {cv_num}")

            ## apply method to calculate and log edema score baseline regression
            create_edema_score_baseline(
                x_train_ptids = train_ptids,
                x_test_ptids = test_ptids,
                x_train_OLTV = train_OLTV,
                x_test_OLTV = test_OLTV,
                x_train_OTV = train_OTV,
                x_test_OTV = test_OTV,
                x_train_targets = train_targets,
                x_test_targets = test_targets,
                type = "mgb",
                cv_num = cv_num
            )

            create_edema_score_baseline(
                x_train_ptids = bmc_train_ptids,
                x_test_ptids = bmc_test_ptids,
                x_train_OLTV = bmc_train_OLTV,
                x_test_OLTV = bmc_test_OLTV,
                x_train_OTV = bmc_train_OTV,
                x_test_OTV = bmc_test_OTV,
                x_train_targets = bmc_train_targets,
                x_test_targets = bmc_test_targets,
                type = "bmc",
                cv_num = cv_num
            )

            ## create whole bmc set for testing
            bmc_OLTV = pd.DataFrame(np.concatenate([bmc_train_OLTV, bmc_test_OLTV]), columns = OLTV_columns)
            bmc_targets = pd.DataFrame(np.concatenate([bmc_train_targets, bmc_test_targets]), columns = ['target'])
            bmc_ptids = pd.DataFrame(np.concatenate([bmc_train_ptids, bmc_test_ptids]), columns = ['ptid'])

            filtered_bmc_OLTV = pd.DataFrame(np.concatenate([filtered_bmc_train_OLTV, filtered_bmc_test_OLTV]), columns = OLTV_columns)
            filtered_bmc_targets = pd.DataFrame(np.concatenate([filtered_bmc_train_targets, filtered_bmc_test_targets]), columns = ['target'])
            filtered_bmc_ptids = pd.DataFrame(np.concatenate([filtered_bmc_train_ptids, filtered_bmc_test_ptids]), columns = ['ptid'])

            ## get dt lists
            list_24 = np.array(test_OLTV['dt'] < 24)
            list_24_48 = np.array(((test_OLTV['dt'] >= 24) & (test_OLTV['dt'] < 48)))
            list_48_96 = np.array(((test_OLTV['dt'] >= 48) & (test_OLTV['dt'] < 96)))
            list_96 = np.array(test_OLTV['dt'] >= 96)

            list_filter_24 = np.array(filtered_test_OLTV['dt'] < 24)
            list_filter_24_48 = np.array(((filtered_test_OLTV['dt'] >= 24) & (filtered_test_OLTV['dt'] < 48)))
            list_filter_48_96 = np.array(((filtered_test_OLTV['dt'] >= 48) & (filtered_test_OLTV['dt'] < 96)))
            list_filter_96 = np.array(filtered_test_OLTV['dt'] >= 96)

            mgb_dt_lists = {
                '24': list_24,
                '24_48': list_24_48,
                '48_96': list_48_96,
                '96': list_96,
                'filter_24': list_filter_24,
                'filter_24_48': list_filter_24_48,
                'filter_48_96': list_filter_48_96,
                'filter_96': list_filter_96
            }

            bmc_list_24 = np.array(bmc_OLTV['dt'] < 24)
            bmc_list_24_48 = np.array(((bmc_OLTV['dt'] >= 24) & (bmc_OLTV['dt'] < 48)))
            bmc_list_48_96 = np.array(((bmc_OLTV['dt'] >= 48) & (bmc_OLTV['dt'] < 96)))
            bmc_list_96 = np.array(bmc_OLTV['dt'] >= 96)
            
            bmc_list_filter_24 = np.array(filtered_bmc_OLTV['dt'] < 24)
            bmc_list_filter_24_48 = np.array(((filtered_bmc_OLTV['dt'] >= 24) & (filtered_bmc_OLTV['dt'] < 48)))
            bmc_list_filter_48_96 = np.array(((filtered_bmc_OLTV['dt'] >= 48) & (filtered_bmc_OLTV['dt'] < 96)))
            bmc_list_filter_96 = np.array(filtered_bmc_OLTV['dt'] >= 96)

            bmc_dt_lists = {
                '24': bmc_list_24,
                '24_48': bmc_list_24_48,
                '48_96': bmc_list_48_96,
                '96': bmc_list_96,
                'filter_24': bmc_list_filter_24,
                'filter_24_48': bmc_list_filter_24_48,
                'filter_48_96': bmc_list_filter_48_96,
                'filter_96': bmc_list_filter_96
            }
            
            ## apply data normalizaiton scaling
            scaler = StandardScaler().fit(train_OLTV)
            
            scaler_dict = {}
            for i, col in enumerate(train_OLTV.columns):
                scaler_dict[col] = [scaler.mean_[i], scaler.scale_[i]]
            
            train_OLTV = scaler.transform(train_OLTV)
            test_OLTV = scaler.transform(test_OLTV)

            bmc_OLTV = scaler.transform(bmc_OLTV)

            ## convert from np.array to pd.DataFrame
            train_OLTV = pd.DataFrame(train_OLTV, columns = OLTV_columns)
            test_OLTV = pd.DataFrame(test_OLTV, columns = OLTV_columns)
            bmc_OLTV = pd.DataFrame(bmc_OLTV, columns = OLTV_columns)

            ## instantiate xgb classifier
            trained_model = XGBClassifier(
                callbacks = [WandbCallback(
                    log_model = True,
                    log_feature_importance= True,
                )],
                learning_rate = learning_rate,
                max_depth = max_depth,
                min_child_weight = min_child_weight,
                gamma = gamma,
                subsample = subsample,
                reg_lambda = reg_lambda,
                reg_alpha = reg_alpha,
            )

            ## train xgb classifier
            trained_model.fit(train_OLTV, train_targets, sample_weight = train_sample_weights)

            ## Method to calculate and log xgb performance metrics 
            @validate_call
            def log_xgb_metrics(
                model,
                test_df,
                test_targets,
                test_ptids,
                filtered_test_df,
                filtered_test_targets,
                filtered_test_ptids,
                lists,
                type: Literal["train", "mgb", "bmc"],
                cv_num
            ):
                if isinstance(test_ptids, pd.DataFrame) or isinstance(test_ptids, pd.Series):
                    test_ptids = np.array(test_ptids['ptid'])
                    filtered_test_ptids = np.array(filtered_test_ptids['ptid'])
                
                if isinstance(test_targets, pd.DataFrame) or isinstance(test_targets, pd.Series):
                    test_targets = np.array(test_targets['target'])
                    filtered_test_targets = np.array(filtered_test_targets['target'])
                
                if type == "mgb" or type == "train":
                    medicaid = mgb_medicaid
                    uninsured = mgb_uninsured
                    other = mgb_other
                elif type == "bmc":
                    medicaid = bmc_medicaid
                    uninsured = bmc_uninsured
                    other = bmc_other

                ### generate model predictions
                probas = model.predict_proba(test_df)
                predictions = model.predict(test_df)

                filtered_probas = model.predict_proba(filtered_test_df)
                filtered_predictions = model.predict(filtered_test_df)

                #### by time period
                probas_24 = probas[lists['24']]
                probas_24_48 = probas[lists['24_48']]
                probas_48_96 = probas[lists['48_96']]
                probas_96 = probas[lists['96']]

                predictions_24 = predictions[lists['24']]
                predictions_24_48 = predictions[lists['24_48']]
                predictions_48_96 = predictions[lists['48_96']]
                predictions_96 = predictions[lists['96']]

                targets_24 = test_targets[lists['24']]
                targets_24_48 = test_targets[lists['24_48']]
                targets_48_96 = test_targets[lists['48_96']]
                targets_96 = test_targets[lists['96']]

                filtered_probas_24 = filtered_probas[lists['filter_24']]
                filtered_probas_24_48 = filtered_probas[lists['filter_24_48']]
                filtered_probas_48_96 = filtered_probas[lists['filter_48_96']]
                filtered_probas_96 = filtered_probas[lists['filter_96']]

                filtered_predictions_24 = filtered_predictions[lists['filter_24']]
                filtered_predictions_24_48 = filtered_predictions[lists['filter_24_48']]
                filtered_predictions_48_96 = filtered_predictions[lists['filter_48_96']]
                filtered_predictions_96 = filtered_predictions[lists['filter_96']]

                filtered_targets_24 = filtered_test_targets[lists['filter_24']]
                filtered_targets_24_48 = filtered_test_targets[lists['filter_24_48']]
                filtered_targets_48_96 = filtered_test_targets[lists['filter_48_96']]
                filtered_targets_96 = filtered_test_targets[lists['filter_96']]

                #### by insurance
                probas_medicaid = probas[np.isin(test_ptids, medicaid)]
                probas_uninsured = probas[np.isin(test_ptids, uninsured)]
                probas_other = probas[np.isin(test_ptids, other)]

                predictions_medicaid = predictions[np.isin(test_ptids, medicaid)]
                predictions_uninsured = predictions[np.isin(test_ptids, uninsured)]
                predictions_other = predictions[np.isin(test_ptids, other)]

                targets_medicaid = test_targets[np.isin(test_ptids, medicaid)]
                targets_uninsured = test_targets[np.isin(test_ptids, uninsured)]
                targets_other = test_targets[np.isin(test_ptids, other)]

                filtered_probas_medicaid = filtered_probas[np.isin(filtered_test_ptids, medicaid)]
                filtered_probas_uninsured = filtered_probas[np.isin(filtered_test_ptids, uninsured)]
                filtered_probas_other = filtered_probas[np.isin(filtered_test_ptids, other)]

                filtered_predictions_medicaid = filtered_predictions[np.isin(filtered_test_ptids, medicaid)]
                filtered_predictions_uninsured = filtered_predictions[np.isin(filtered_test_ptids, uninsured)]
                filtered_predictions_other = filtered_predictions[np.isin(filtered_test_ptids, other)]

                filtered_targets_medicaid = filtered_test_targets[np.isin(filtered_test_ptids, medicaid)]
                filtered_targets_uninsured = filtered_test_targets[np.isin(filtered_test_ptids, uninsured)]
                filtered_targets_other = filtered_test_targets[np.isin(filtered_test_ptids, other)]

                ### method to evaluate metrics for xgb models
                def eval_xgb_probas(probas, predictions, filtered_probas, filtered_predictions, targets, filtered_targets, all_hours = True):
                    if not all_hours and ((len(probas) == 0) or (len(filtered_probas) == 0)):
                        return 0, 0
                    
                    targets = targets.astype(int)
                    filtered_targets = filtered_targets.astype(int)
                    
                    y_probas = tnsr(probas).float()
                    y_predict = tnsr(predictions)

                    if isinstance(targets, pd.DataFrame) or isinstance(targets, pd.Series):
                        y_true = tnsr(targets.to_numpy().flatten())
                    else:
                        y_true = tnsr(targets.flatten())

                    filtered_y_probas = tnsr(filtered_probas).float()
                    filtered_y_predict = tnsr(filtered_predictions)

                    if isinstance(filtered_targets, pd.DataFrame)  or isinstance(filtered_targets, pd.Series):
                        filtered_y_true = tnsr(filtered_targets.to_numpy().flatten())
                    else:
                        filtered_y_true = tnsr(filtered_targets.flatten())

                    if all_hours:
                        ### calculate per-class performance metrics
                        per_class_accuracies = class_accuracy(y_predict, y_true).tolist()
                        per_class_aurocs = class_AUROC(y_probas, y_true).tolist()
                        per_class_auprcs = class_AUPRC(y_probas, y_true).tolist()

                        filtered_per_class_accuracies = class_accuracy(filtered_y_predict, filtered_y_true).tolist()
                        filtered_per_class_aurocs = class_AUROC(filtered_y_probas, filtered_y_true).tolist()
                        filtered_per_class_auprcs = class_AUPRC(filtered_y_probas, filtered_y_true).tolist()

                        ### calculate weighted mean metrics
                        xgb_mean_accuracy = mean_accuracy(y_predict, y_true).item()
                        xgb_mean_auroc = mean_AUROC(y_probas, y_true).item()
                        xgb_mean_auprc = mean_AUPRC(y_probas, y_true).item()

                        filtered_xgb_mean_accuracy = mean_accuracy(filtered_y_predict, filtered_y_true).item()
                        filtered_xgb_mean_auroc = mean_AUROC(filtered_y_probas, filtered_y_true).item()
                        filtered_xgb_mean_auprc = mean_AUPRC(filtered_y_probas, filtered_y_true).item()

                        ### convert to numpy arrays
                        y_probas = np.array(y_probas.tolist())
                        y_true = np.array(y_true.tolist())
                        filtered_y_probas = np.array(filtered_y_probas.tolist())
                        filtered_y_true = np.array(filtered_y_true.tolist())

                        ### generate plot data and save curve plots
                        prc_data, roc_data = get_plot_data(y_true, y_probas, cv_num = cv_num, type = type, model = "xgb")
                        filtered_prc_data, filtered_roc_data = get_plot_data(filtered_y_true, filtered_y_probas, cv_num = cv_num, type = type, model = "xgb", filtered = True)

                        return per_class_accuracies, per_class_aurocs, per_class_auprcs, xgb_mean_accuracy, xgb_mean_auroc, xgb_mean_auprc, prc_data, roc_data, filtered_prc_data, filtered_roc_data, filtered_per_class_accuracies, filtered_per_class_aurocs, filtered_per_class_auprcs, filtered_xgb_mean_accuracy, filtered_xgb_mean_auroc, filtered_xgb_mean_auprc

                    else:
                        xgb_mean_auroc = mean_AUROC(y_probas, y_true).item()
                        filtered_xgb_mean_auroc = mean_AUROC(filtered_y_probas, filtered_y_true).item()

                        return xgb_mean_auroc, filtered_xgb_mean_auroc
                    
                ### calculate metrics for unfiltered and filtered datasets
                per_class_accuracies, per_class_aurocs, per_class_auprcs, xgb_mean_accuracy, xgb_mean_auroc, xgb_mean_auprc, prc_data, roc_data, filtered_prc_data, filtered_roc_data, filtered_per_class_accuracies, filtered_per_class_aurocs, filtered_per_class_auprcs, filtered_xgb_mean_accuracy, filtered_xgb_mean_auroc, filtered_xgb_mean_auprc = eval_xgb_probas(
                    probas, predictions, filtered_probas, filtered_predictions, test_targets, filtered_test_targets, all_hours = True
                )

                ### calculate metrics for by insurance datasets
                xgb_mean_auroc_medicaid, filtered_xgb_mean_auroc_medicaid = eval_xgb_probas(
                    probas_medicaid, predictions_medicaid, filtered_probas_medicaid, filtered_predictions_medicaid, targets_medicaid, filtered_targets_medicaid, all_hours = False
                )

                xgb_mean_auroc_uninsured, filtered_xgb_mean_auroc_uninsured = eval_xgb_probas(
                    probas_uninsured, predictions_uninsured, filtered_probas_uninsured, filtered_predictions_uninsured, targets_uninsured, filtered_targets_uninsured, all_hours = False
                )

                xgb_mean_auroc_other, filtered_xgb_mean_auroc_other = eval_xgb_probas(
                    probas_other, predictions_other, filtered_probas_other, filtered_predictions_other, targets_other, filtered_targets_other, all_hours = False
                )

                ### calculate metrics for by time period datasets
                xgb_mean_auroc_24, filtered_xgb_mean_auroc_24 = eval_xgb_probas(
                    probas_24, predictions_24, filtered_probas_24, filtered_predictions_24, targets_24, filtered_targets_24, all_hours = False
                )

                xgb_mean_auroc_24_48, filtered_xgb_mean_auroc_24_48 = eval_xgb_probas(
                    probas_24_48, predictions_24_48, filtered_probas_24_48, filtered_predictions_24_48, targets_24_48, filtered_targets_24_48, all_hours = False
                )

                xgb_mean_auroc_48_96, filtered_xgb_mean_auroc_48_96 = eval_xgb_probas(
                    probas_48_96, predictions_48_96, filtered_probas_48_96, filtered_predictions_48_96, targets_48_96, filtered_targets_48_96, all_hours = False
                )

                xgb_mean_auroc_96, filtered_xgb_mean_auroc_96 = eval_xgb_probas(
                    probas_96, predictions_96, filtered_probas_96, filtered_predictions_96, targets_96, filtered_targets_96, all_hours = False
                )

                ### log metrics to respective lists
                if type == "train":
                    keys = list(cv_xgb_training_per_class_accuracy.keys())
                    for i in range(4):
                        cv_xgb_training_per_class_accuracy[keys[i]].append(per_class_accuracies[i])
                        cv_xgb_training_per_class_auroc[keys[i]].append(per_class_aurocs[i])
                        cv_xgb_training_per_class_auprc[keys[i]].append(per_class_auprcs[i])

                        cv_xgb_training_per_class_filtered_accuracy[keys[i]].append(filtered_per_class_accuracies[i])
                        cv_xgb_training_per_class_filtered_auroc[keys[i]].append(filtered_per_class_aurocs[i])
                        cv_xgb_training_per_class_filtered_auprc[keys[i]].append(filtered_per_class_auprcs[i])

                    cv_xgb_training_overall_accuracy.append(xgb_mean_accuracy)
                    cv_xgb_training_overall_auroc.append(xgb_mean_auroc)
                    cv_xgb_training_overall_auprc.append(xgb_mean_auprc)

                    cv_xgb_training_filtered_accuracy.append(filtered_xgb_mean_accuracy)
                    cv_xgb_training_filtered_auroc.append(filtered_xgb_mean_auroc)
                    cv_xgb_training_filtered_auprc.append(filtered_xgb_mean_auprc)

                elif type == "mgb":
                    keys = list(cv_xgb_per_class_accuracy.keys())
                    for i in range(4):
                        cv_xgb_per_class_accuracy[keys[i]].append(per_class_accuracies[i])
                        cv_xgb_per_class_auroc[keys[i]].append(per_class_aurocs[i])
                        cv_xgb_per_class_auprc[keys[i]].append(per_class_auprcs[i])

                        cv_xgb_per_class_filtered_accuracy[keys[i]].append(filtered_per_class_accuracies[i])
                        cv_xgb_per_class_filtered_auroc[keys[i]].append(filtered_per_class_aurocs[i])
                        cv_xgb_per_class_filtered_auprc[keys[i]].append(filtered_per_class_auprcs[i])

                    #### log per-period mean metrics
                    hour_keys = list(cv_xgb_per_period_auroc.keys())
                    cv_xgb_per_period_auroc[hour_keys[0]].append(xgb_mean_auroc_24)
                    cv_xgb_per_period_auroc[hour_keys[1]].append(xgb_mean_auroc_24_48)
                    cv_xgb_per_period_auroc[hour_keys[2]].append(xgb_mean_auroc_48_96)
                    cv_xgb_per_period_auroc[hour_keys[3]].append(xgb_mean_auroc_96)

                    #### log mean metrics by insurance
                    insurance_keys = list(cv_xgb_by_insurance_auroc.keys())
                    cv_xgb_by_insurance_auroc[insurance_keys[0]].append(xgb_mean_auroc_medicaid)
                    cv_xgb_by_insurance_auroc[insurance_keys[1]].append(xgb_mean_auroc_uninsured)
                    cv_xgb_by_insurance_auroc[insurance_keys[2]].append(xgb_mean_auroc_other)

                    #### log weighted mean metrics
                    cv_xgb_overall_accuracy.append(xgb_mean_accuracy)
                    cv_xgb_overall_auroc.append(xgb_mean_auroc)
                    cv_xgb_overall_auprc.append(xgb_mean_auprc)

                    cv_xgb_filtered_accuracy.append(filtered_xgb_mean_accuracy)
                    cv_xgb_filtered_auroc.append(filtered_xgb_mean_auroc)
                    cv_xgb_filtered_auprc.append(filtered_xgb_mean_auprc)

                    #### log plot data
                    cv_xgb_overall_prc_data.append(prc_data)
                    cv_xgb_overall_roc_data.append(roc_data)

                    cv_xgb_filtered_prc_data.append(filtered_prc_data)
                    cv_xgb_filtered_roc_data.append(filtered_roc_data)

                elif type == "bmc":
                    keys = list(cv_xgb_bmc_per_class_accuracy.keys())
                    for i in range(4):
                        cv_xgb_bmc_per_class_accuracy[keys[i]].append(per_class_accuracies[i])
                        cv_xgb_bmc_per_class_auroc[keys[i]].append(per_class_aurocs[i])
                        cv_xgb_bmc_per_class_auprc[keys[i]].append(per_class_auprcs[i])

                        cv_xgb_bmc_per_class_filtered_accuracy[keys[i]].append(filtered_per_class_accuracies[i])
                        cv_xgb_bmc_per_class_filtered_auroc[keys[i]].append(filtered_per_class_aurocs[i])
                        cv_xgb_bmc_per_class_filtered_auprc[keys[i]].append(filtered_per_class_auprcs[i])

                    #### log per-period mean metrics
                    hour_keys = list(cv_xgb_bmc_per_period_auroc.keys())
                    cv_xgb_bmc_per_period_auroc[hour_keys[0]].append(xgb_mean_auroc_24)
                    cv_xgb_bmc_per_period_auroc[hour_keys[1]].append(xgb_mean_auroc_24_48)
                    cv_xgb_bmc_per_period_auroc[hour_keys[2]].append(xgb_mean_auroc_48_96)
                    cv_xgb_bmc_per_period_auroc[hour_keys[3]].append(xgb_mean_auroc_96)

                    cv_xgb_bmc_per_period_filtered_auroc[hour_keys[0]].append(filtered_xgb_mean_auroc_24)
                    cv_xgb_bmc_per_period_filtered_auroc[hour_keys[1]].append(filtered_xgb_mean_auroc_24_48)
                    cv_xgb_bmc_per_period_filtered_auroc[hour_keys[2]].append(filtered_xgb_mean_auroc_48_96)
                    cv_xgb_bmc_per_period_filtered_auroc[hour_keys[3]].append(filtered_xgb_mean_auroc_96)

                    #### log mean metrics by insurance
                    insurance_keys = list(cv_xgb_bmc_by_insurance_auroc.keys())
                    cv_xgb_bmc_by_insurance_auroc[insurance_keys[0]].append(xgb_mean_auroc_medicaid)
                    cv_xgb_bmc_by_insurance_auroc[insurance_keys[1]].append(xgb_mean_auroc_uninsured)
                    cv_xgb_bmc_by_insurance_auroc[insurance_keys[2]].append(xgb_mean_auroc_other)

                    cv_xgb_bmc_by_insurance_filtered_auroc[insurance_keys[0]].append(filtered_xgb_mean_auroc_medicaid)
                    cv_xgb_bmc_by_insurance_filtered_auroc[insurance_keys[1]].append(filtered_xgb_mean_auroc_uninsured)
                    cv_xgb_bmc_by_insurance_filtered_auroc[insurance_keys[2]].append(filtered_xgb_mean_auroc_other)

                    #### log weighted mean metrics
                    cv_xgb_bmc_overall_accuracy.append(xgb_mean_accuracy)
                    cv_xgb_bmc_overall_auroc.append(xgb_mean_auroc)
                    cv_xgb_bmc_overall_auprc.append(xgb_mean_auprc)

                    cv_xgb_bmc_filtered_accuracy.append(filtered_xgb_mean_accuracy)
                    cv_xgb_bmc_filtered_auroc.append(filtered_xgb_mean_auroc)
                    cv_xgb_bmc_filtered_auprc.append(filtered_xgb_mean_auprc)

                    #### log plot data
                    cv_xgb_bmc_overall_prc_data.append(prc_data)
                    cv_xgb_bmc_overall_roc_data.append(roc_data)

                    cv_xgb_bmc_filtered_prc_data.append(filtered_prc_data)
                    cv_xgb_bmc_filtered_roc_data.append(filtered_roc_data)
            
                ### Log metrics to wandb
                for i in range(4):
                    wandb.log({
                        f"{type} XGB accuracy (class {i}) for cv {cv_num}": per_class_accuracies[i],
                        f"{type} XGB AUROC (class {i}) for cv {cv_num}": per_class_aurocs[i],
                        f"{type} XGB AUPRC (class {i}) for cv {cv_num}": per_class_auprcs[i],
                        f"filtered {type} XGB accuracy (class {i}) for cv {cv_num}": filtered_per_class_accuracies[i],
                        f"filtered {type} XGB AUROC (class {i}) for cv {cv_num}": filtered_per_class_aurocs[i],
                        f"filtered {type} XGB AUPRC (class {i}) for cv {cv_num}": filtered_per_class_auprcs[i]
                    })
                
                wandb.log({
                    f"{type} XGB mean AUROC for period < 24 for cv {cv_num}": xgb_mean_auroc_24,
                    f"filtered {type} XGB mean AUROC for period < 24 for cv {cv_num}": filtered_xgb_mean_auroc_24,
                    f"{type} XGB mean AUROC for period 24-48 for cv {cv_num}": xgb_mean_auroc_24_48,
                    f"filtered {type} XGB mean AUROC for period 24-48 for cv {cv_num}": filtered_xgb_mean_auroc_24_48,
                    f"{type} XGB mean AUROC for period 48-96 for cv {cv_num}": xgb_mean_auroc_48_96,
                    f"filtered {type} XGB mean AUROC for period 48-96 for cv {cv_num}": filtered_xgb_mean_auroc_48_96,
                    f"{type} XGB mean AUROC for period > 96 for cv {cv_num}": xgb_mean_auroc_96,
                    f"filtered {type} XGB mean AUROC for period > 96 for cv {cv_num}": filtered_xgb_mean_auroc_96
                })

                wandb.log({
                    f"{type} XGB mean AUROC for medicaid for cv {cv_num}": xgb_mean_auroc_medicaid,
                    f"filtered {type} XGB mean AUROC for medicaid for cv {cv_num}": filtered_xgb_mean_auroc_medicaid,
                    f"{type} XGB mean AUROC for uninsured for cv {cv_num}": xgb_mean_auroc_uninsured,
                    f"filtered {type} XGB mean AUROC for uninsured for cv {cv_num}": filtered_xgb_mean_auroc_uninsured,
                    f"{type} XGB mean AUROC for other for cv {cv_num}": xgb_mean_auroc_other,
                    f"filtered {type} XGB mean AUROC for other for cv {cv_num}": filtered_xgb_mean_auroc_other
                })

                wandb.log({
                    f"{type} XGB mean accuracy for cv {cv_num}": xgb_mean_accuracy,
                    f"{type} XGB mean AUROC for cv {cv_num}": xgb_mean_auroc,
                    f"{type} XGB mean AUPRC for cv {cv_num}": xgb_mean_auprc,
                    f"filtered {type} XGB mean accuracy for cv {cv_num}": filtered_xgb_mean_accuracy,
                    f"filtered {type} XGB mean AUROC for cv {cv_num}": filtered_xgb_mean_auroc,
                    f"filtered {type} XGB mean AUPRC for cv {cv_num}": filtered_xgb_mean_auprc
                })

                print(f"Done logging xgb metrics for {type} on cv {cv_num}")
            

            ## calculate and log metrics for mgb test set
            log_xgb_metrics(
                model = trained_model, 
                test_df = test_OLTV,
                test_targets = test_targets,
                test_ptids = test_ptids,
                filtered_test_df = filtered_test_OLTV,
                filtered_test_targets = filtered_test_targets,
                filtered_test_ptids = filtered_test_ptids,
                lists = mgb_dt_lists,
                type = "mgb", 
                cv_num = cv_num
            )

            ## calculate and log metrics for bmc
            log_xgb_metrics(
                model = trained_model,
                test_df = bmc_OLTV,
                test_targets = bmc_targets,
                test_ptids = bmc_ptids,
                filtered_test_df = filtered_bmc_OLTV,
                filtered_test_targets = filtered_bmc_targets,
                filtered_test_ptids = filtered_bmc_ptids,
                lists = bmc_dt_lists,
                type = "bmc",
                cv_num = cv_num
            )

            ## generate shap plot for trained model
            explainer = shap.TreeExplainer(trained_model)

            ### apply shap plot generator for each test dataset
            make_shap_plot(explainer, test_OLTV, f"mgb_shap_plot_cv_{cv_num}.png")
            make_shap_plot(explainer, filtered_test_OLTV, f"filtered_mgb_shap_plot_cv_{cv_num}.png")
            make_shap_plot(explainer, bmc_test_OLTV, f"bmc_shap_plot_cv_{cv_num}.png")
            make_shap_plot(explainer, bmc_test_OLTV, f"filtered_bmc_shap_plot_cv_{cv_num}.png")

            cv_num = cv_num + 1
    
    else:
        cv_num = 0
        for(
            train_ptids,
            test_ptids,
            train_OLTV,
            test_OLTV,
            train_targets,
            test_targets,
            bmc_train_ptids,
            bmc_test_ptids,
            bmc_train_OLTV,
            bmc_test_OLTV,
            bmc_train_targets,
            bmc_test_targets,
        ) in zip(
            cv_train_ptids,
            cv_test_ptids, 
            cv_train_OLTVs,
            cv_test_OLTVs,
            cv_train_targets,
            cv_test_targets,
            cv_bmc_train_ptids,
            cv_bmc_test_ptids,
            cv_bmc_train_OLTVs,
            cv_bmc_test_OLTVs,
            cv_bmc_train_targets,
            cv_bmc_test_targets,
        ):
            
            ## save OLTV columns
            OLTV_columns = train_OLTV.columns.copy()

            ## find patient-hours where transition occurs for sample weighting and filtered metrics
            ### apply target transform to mls value
            train_mls_transformed = train_OLTV['size_mls'].apply(target_transform)
            test_mls_transformed = test_OLTV['size_mls'].apply(target_transform)
            bmc_train_mls_transformed = bmc_train_OLTV['size_mls'].apply(target_transform)
            bmc_test_mls_transformed = bmc_test_OLTV['size_mls'].apply(target_transform)

            ### filter datasets where current mls class != target class
            filtered_train_OLTV = train_OLTV[train_mls_transformed != train_targets]
            filtered_train_targets = train_targets[train_mls_transformed != train_targets]
        
            filtered_test_OLTV = test_OLTV[test_mls_transformed != test_targets]
            filtered_test_targets = test_targets[test_mls_transformed != test_targets]
            
            filtered_bmc_train_OLTV = bmc_train_OLTV[bmc_train_mls_transformed != bmc_train_targets]
            filtered_bmc_train_targets = bmc_train_targets[bmc_train_mls_transformed != bmc_train_targets]
            
            filtered_bmc_test_OLTV = bmc_test_OLTV[bmc_test_mls_transformed != bmc_test_targets]
            filtered_bmc_test_targets = bmc_test_targets[bmc_test_mls_transformed != bmc_test_targets]

            ### create sample weighting for training 
            train_sample_weights = np.ones(train_OLTV.shape[0])
            train_sample_weights[train_mls_transformed == train_targets] = no_transition_weighting

            ## IMPORTANT: linear regression baseline cannot be made in DEMO MODE due to lack of access to OTVs

            ## apply data normalizaiton scaling
            scaler = StandardScaler().fit(train_OLTV)
            
            scaler_dict = {}
            for i, col in enumerate(train_OLTV.columns):
                scaler_dict[col] = [scaler.mean_[i], scaler.scale_[i]]
            
            train_OLTV = scaler.transform(train_OLTV)
            test_OLTV = scaler.transform(test_OLTV)

            bmc_train_OLTV = scaler.transform(bmc_train_OLTV)
            bmc_test_OLTV = scaler.transform(bmc_test_OLTV)

            ## convert from np.array to pd.DataFrame
            train_OLTV = pd.DataFrame(train_OLTV, columns = OLTV_columns)
            test_OLTV = pd.DataFrame(test_OLTV, columns = OLTV_columns)
            bmc_train_OLTV = pd.DataFrame(bmc_train_OLTV, columns = OLTV_columns)
            bmc_test_OLTV = pd.DataFrame(bmc_test_OLTV, columns = OLTV_columns)

            ## instantiate xgb classifier
            trained_model = XGBClassifier(
                callbacks = [WandbCallback(
                    log_model = True,
                    log_feature_importance= True,
                )],
                learning_rate = learning_rate,
                max_depth = max_depth,
                min_child_weight = min_child_weight,
                gamma = gamma,
                subsample = subsample,
                reg_lambda = reg_lambda,
                reg_alpha = reg_alpha,
            )

            ## train xgb classifier
            trained_model.fit(train_OLTV, train_targets, sample_weight = train_sample_weights)

            ## create whole bmc set for testing
            bmc_OLTV = pd.DataFrame(np.concatenate([bmc_train_OLTV, bmc_test_OLTV]), columns = OLTV_columns)
            bmc_targets = pd.DataFrame(np.concatenate([bmc_train_targets, bmc_test_targets]), columns = ['target'])

            filtered_bmc_OLTV = pd.DataFrame(np.concatenate([filtered_bmc_train_OLTV, filtered_bmc_test_OLTV]), columns = OLTV_columns)
            filtered_bmc_targets = pd.DataFrame(np.concatenate([filtered_bmc_train_targets, filtered_bmc_test_targets]), columns = ['target'])

            ## Method to calculate and log xgb performance metrics 
            @validate_call
            def log_xgb_metrics(
                model,
                test_df,
                test_targets,
                filtered_test_df,
                filtered_test_targets,
                type: Literal["train", "mgb", "bmc"],
                cv_num
            ):
                ### generate model predictions
                probas = model.predict_proba(test_df)
                predictions = model.predict(test_df)

                y_probas = tnsr(probas).float()
                y_predict = tnsr(predictions)

                if isinstance(test_targets, pd.DataFrame) or isinstance(test_targets, pd.Series):
                    y_true = tnsr(test_targets.to_numpy().flatten())
                else:
                    y_true = tnsr(test_targets.flatten())

                filtered_probas = model.predict_proba(filtered_test_df)
                filtered_y_probas = tnsr(filtered_probas).float()

                filtered_predictions = model.predict(filtered_test_df)
                filtered_y_predict = tnsr(filtered_predictions)

                if isinstance(filtered_test_targets, pd.DataFrame)  or isinstance(filtered_test_targets, pd.Series):
                    filtered_y_true = tnsr(filtered_test_targets.to_numpy().flatten())
                else:
                    filtered_y_true = tnsr(filtered_test_targets.flatten())

                ### calculate per-class performance metrics
                per_class_accuracies = class_accuracy(y_predict, y_true).tolist()
                per_class_aurocs = class_AUROC(y_probas, y_true).tolist()
                per_class_auprcs = class_AUPRC(y_probas, y_true).tolist()

                filtered_per_class_accuracies = class_accuracy(filtered_y_predict, filtered_y_true).tolist()
                filtered_per_class_aurocs = class_AUROC(filtered_y_probas, filtered_y_true).tolist()
                filtered_per_class_auprcs = class_AUPRC(filtered_y_probas, filtered_y_true).tolist()

                ### calculate weighted mean metrics
                xgb_mean_accuracy = mean_accuracy(y_predict, y_true).item()
                xgb_mean_auroc = mean_AUROC(y_probas, y_true).item()
                xgb_mean_auprc = mean_AUPRC(y_probas, y_true).item()

                filtered_xgb_mean_accuracy = mean_accuracy(filtered_y_predict, filtered_y_true).item()
                filtered_xgb_mean_auroc = mean_AUROC(filtered_y_probas, filtered_y_true).item()
                filtered_xgb_mean_auprc = mean_AUPRC(filtered_y_probas, filtered_y_true).item()

                ### convert to numpy arrays
                y_probas = np.array(y_probas.tolist())
                y_true = np.array(y_true.tolist())
                filtered_y_probas = np.array(filtered_y_probas.tolist())
                filtered_y_true = np.array(filtered_y_true.tolist())

                ### generate plot data and save curve plots
                prc_data, roc_data = get_plot_data(y_true, y_probas, cv_num = cv_num, type = type, model = "xgb")
                filtered_prc_data, filtered_roc_data = get_plot_data(filtered_y_true, filtered_y_probas, cv_num = cv_num, type = type, model = "xgb", filtered = True)

                ### log metrics to respective lists
                if type == "train":
                    keys = list(cv_xgb_training_per_class_accuracy.keys())
                    for i in range(4):
                        cv_xgb_training_per_class_accuracy[keys[i]].append(per_class_accuracies[i])
                        cv_xgb_training_per_class_auroc[keys[i]].append(per_class_aurocs[i])
                        cv_xgb_training_per_class_auprc[keys[i]].append(per_class_auprcs[i])

                        cv_xgb_training_per_class_filtered_accuracy[keys[i]].append(filtered_per_class_accuracies[i])
                        cv_xgb_training_per_class_filtered_auroc[keys[i]].append(filtered_per_class_aurocs[i])
                        cv_xgb_training_per_class_filtered_auprc[keys[i]].append(filtered_per_class_auprcs[i])

                    cv_xgb_training_overall_accuracy.append(xgb_mean_accuracy)
                    cv_xgb_training_overall_auroc.append(xgb_mean_auroc)
                    cv_xgb_training_overall_auprc.append(xgb_mean_auprc)

                    cv_xgb_training_filtered_accuracy.append(filtered_xgb_mean_accuracy)
                    cv_xgb_training_filtered_auroc.append(filtered_xgb_mean_auroc)
                    cv_xgb_training_filtered_auprc.append(filtered_xgb_mean_auprc)

                elif type == "mgb":
                    keys = list(cv_xgb_per_class_accuracy.keys())
                    for i in range(4):
                        cv_xgb_per_class_accuracy[keys[i]].append(per_class_accuracies[i])
                        cv_xgb_per_class_auroc[keys[i]].append(per_class_aurocs[i])
                        cv_xgb_per_class_auprc[keys[i]].append(per_class_auprcs[i])

                        cv_xgb_per_class_filtered_accuracy[keys[i]].append(filtered_per_class_accuracies[i])
                        cv_xgb_per_class_filtered_auroc[keys[i]].append(filtered_per_class_aurocs[i])
                        cv_xgb_per_class_filtered_auprc[keys[i]].append(filtered_per_class_auprcs[i])

                    cv_xgb_overall_accuracy.append(xgb_mean_accuracy)
                    cv_xgb_overall_auroc.append(xgb_mean_auroc)
                    cv_xgb_overall_auprc.append(xgb_mean_auprc)

                    cv_xgb_filtered_accuracy.append(filtered_xgb_mean_accuracy)
                    cv_xgb_filtered_auroc.append(filtered_xgb_mean_auroc)
                    cv_xgb_filtered_auprc.append(filtered_xgb_mean_auprc)

                    #### log plot data
                    cv_xgb_overall_prc_data.append(prc_data)
                    cv_xgb_overall_roc_data.append(roc_data)

                    cv_xgb_filtered_prc_data.append(filtered_prc_data)
                    cv_xgb_filtered_roc_data.append(filtered_roc_data)

                elif type == "bmc":
                    keys = list(cv_xgb_bmc_per_class_accuracy.keys())
                    for i in range(4):
                        cv_xgb_bmc_per_class_accuracy[keys[i]].append(per_class_accuracies[i])
                        cv_xgb_bmc_per_class_auroc[keys[i]].append(per_class_aurocs[i])
                        cv_xgb_bmc_per_class_auprc[keys[i]].append(per_class_auprcs[i])

                        cv_xgb_bmc_per_class_filtered_accuracy[keys[i]].append(filtered_per_class_accuracies[i])
                        cv_xgb_bmc_per_class_filtered_auroc[keys[i]].append(filtered_per_class_aurocs[i])
                        cv_xgb_bmc_per_class_filtered_auprc[keys[i]].append(filtered_per_class_auprcs[i])

                    cv_xgb_bmc_overall_accuracy.append(xgb_mean_accuracy)
                    cv_xgb_bmc_overall_auroc.append(xgb_mean_auroc)
                    cv_xgb_bmc_overall_auprc.append(xgb_mean_auprc)

                    cv_xgb_bmc_filtered_accuracy.append(filtered_xgb_mean_accuracy)
                    cv_xgb_bmc_filtered_auroc.append(filtered_xgb_mean_auroc)
                    cv_xgb_bmc_filtered_auprc.append(filtered_xgb_mean_auprc)

                    #### log plot data
                    cv_xgb_bmc_overall_prc_data.append(prc_data)
                    cv_xgb_bmc_overall_roc_data.append(roc_data)

                    cv_xgb_bmc_filtered_prc_data.append(filtered_prc_data)
                    cv_xgb_bmc_filtered_roc_data.append(filtered_roc_data)
            
                ### Log metrics to wandb
                for i in range(4):
                    wandb.log({
                        f"{type} XGB accuracy (class {i}) for cv {cv_num}": per_class_accuracies[i],
                        f"{type} XGB AUROC (class {i}) for cv {cv_num}": per_class_aurocs[i],
                        f"{type} XGB AUPRC (class {i}) for cv {cv_num}": per_class_auprcs[i],
                        f"filtered {type} XGB accuracy (class {i}) for cv {cv_num}": filtered_per_class_accuracies[i],
                        f"filtered {type} XGB AUROC (class {i}) for cv {cv_num}": filtered_per_class_aurocs[i],
                        f"filtered {type} XGB AUPRC (class {i}) for cv {cv_num}": filtered_per_class_auprcs[i]
                    })
                
                wandb.log({
                    f"{type} XGB mean accuracy for cv {cv_num}": xgb_mean_accuracy,
                    f"{type} XGB mean AUROC for cv {cv_num}": xgb_mean_auroc,
                    f"{type} XGB mean AUPRC for cv {cv_num}": xgb_mean_auprc,
                    f"filtered {type} XGB mean accuracy for cv {cv_num}": filtered_xgb_mean_accuracy,
                    f"filtered {type} XGB mean AUROC for cv {cv_num}": filtered_xgb_mean_auroc,
                    f"filtered {type} XGB mean AUPRC for cv {cv_num}": filtered_xgb_mean_auprc
                })

                print(f"Done logging xgb metrics for {type} on cv {cv_num}")
            

            ## calculate and log metrics for mgb test set
            log_xgb_metrics(
                model = trained_model, 
                test_df = test_OLTV,
                test_targets = test_targets,
                filtered_test_df = filtered_test_OLTV,
                filtered_test_targets = filtered_test_targets,
                type = "mgb", 
                cv_num = cv_num
            )

            ## calculate and log metrics for bmc
            log_xgb_metrics(
                model = trained_model,
                test_df = bmc_OLTV,
                test_targets = bmc_targets,
                filtered_test_df = filtered_bmc_OLTV,
                filtered_test_targets = filtered_bmc_targets,
                type = "bmc",
                cv_num = cv_num
            )

            ## generate shap plot for trained model
            explainer = shap.TreeExplainer(trained_model)

            ### apply shap plot generator for each test dataset
            make_shap_plot(explainer, test_OLTV, f"mgb_shap_plot_cv_{cv_num}.png")
            make_shap_plot(explainer, filtered_test_OLTV, f"filtered_mgb_shap_plot_cv_{cv_num}.png")
            make_shap_plot(explainer, bmc_test_OLTV, f"bmc_shap_plot_cv_{cv_num}.png")
            make_shap_plot(explainer, bmc_test_OLTV, f"filtered_bmc_shap_plot_cv_{cv_num}.png")

            cv_num = cv_num + 1
    

    # Calculate average metrics across cvs 
    classes = ["0", "0_3", "3_8", "8_up"]
    periods = ["<24", "24_48", "48_96", ">96"]
    insurance = ["medicaid", "uninsured", "other"]
    
    if not DEMO_MODE:
        ## regression baseline metrics
        ### instantiate dictionaries for per-class metrics
        mean_baseline_per_class_accuracy = {}
        mean_baseline_per_class_filtered_accuracy = {}
        mean_baseline_bmc_per_class_accuracy = {}
        mean_baseline_bmc_per_class_filtered_accuracy = {}
        sd_baseline_per_class_accuracy = {}
        sd_baseline_per_class_filtered_accuracy = {}
        sd_baseline_bmc_per_class_accuracy = {}
        sd_baseline_bmc_per_class_filtered_accuracy = {}

        mean_baseline_per_class_auroc = {}
        mean_baseline_per_class_filtered_auroc = {}
        mean_baseline_bmc_per_class_auroc = {}
        mean_baseline_bmc_per_class_filtered_auroc = {}
        sd_baseline_per_class_auroc = {}
        sd_baseline_per_class_filtered_auroc = {}
        sd_baseline_bmc_per_class_auroc = {}
        sd_baseline_bmc_per_class_filtered_auroc = {}

        mean_baseline_per_class_auprc = {}
        mean_baseline_per_class_filtered_auprc = {}
        mean_baseline_bmc_per_class_auprc = {}
        mean_baseline_bmc_per_class_filtered_auprc = {}
        sd_baseline_per_class_auprc = {}
        sd_baseline_per_class_filtered_auprc = {}
        sd_baseline_bmc_per_class_auprc = {}
        sd_baseline_bmc_per_class_filtered_auprc = {}

        ### fill per-class metric dictionaries with means + sd
        for i in range(4):
            name = classes[i]

            mean_baseline_per_class_accuracy[name] = np.mean(cv_baseline_per_class_accuracy[name])
            mean_baseline_per_class_filtered_accuracy[name] = np.mean(cv_baseline_per_class_filtered_accuracy[name])
            mean_baseline_bmc_per_class_accuracy[name] = np.mean(cv_baseline_bmc_per_class_accuracy[name])
            mean_baseline_bmc_per_class_filtered_accuracy[name] = np.mean(cv_baseline_bmc_per_class_filtered_accuracy[name])
            sd_baseline_per_class_accuracy[name] = np.std(cv_baseline_per_class_accuracy[name])
            sd_baseline_per_class_filtered_accuracy[name] = np.std(cv_baseline_per_class_filtered_accuracy[name])
            sd_baseline_bmc_per_class_accuracy[name] = np.std(cv_baseline_bmc_per_class_accuracy[name])
            sd_baseline_bmc_per_class_filtered_accuracy[name] = np.std(cv_baseline_bmc_per_class_filtered_accuracy[name])

            mean_baseline_per_class_auroc[name] = np.mean(cv_baseline_per_class_auroc[name])
            mean_baseline_per_class_filtered_auroc[name] = np.mean(cv_baseline_per_class_filtered_auroc[name])
            mean_baseline_bmc_per_class_auroc[name] = np.mean(cv_baseline_bmc_per_class_auroc[name])
            mean_baseline_bmc_per_class_filtered_auroc[name] = np.mean(cv_baseline_bmc_per_class_filtered_auroc[name])
            sd_baseline_per_class_auroc[name] = np.std(cv_baseline_per_class_auroc[name])
            sd_baseline_per_class_filtered_auroc[name] = np.std(cv_baseline_per_class_filtered_auroc[name])
            sd_baseline_bmc_per_class_auroc[name] = np.std(cv_baseline_bmc_per_class_auroc[name])
            sd_baseline_bmc_per_class_filtered_auroc[name] = np.std(cv_baseline_bmc_per_class_filtered_auroc[name])

            mean_baseline_per_class_auprc[name] = np.mean(cv_baseline_per_class_auprc[name])
            mean_baseline_per_class_filtered_auprc[name] = np.mean(cv_baseline_per_class_filtered_auprc[name])
            mean_baseline_bmc_per_class_auprc[name] = np.mean(cv_baseline_bmc_per_class_auprc[name])
            mean_baseline_bmc_per_class_filtered_auprc[name] = np.mean(cv_baseline_bmc_per_class_filtered_auprc[name])
            sd_baseline_per_class_auprc[name] = np.std(cv_baseline_per_class_auprc[name])
            sd_baseline_per_class_filtered_auprc[name] = np.std(cv_baseline_per_class_filtered_auprc[name])
            sd_baseline_bmc_per_class_auprc[name] = np.std(cv_baseline_bmc_per_class_auprc[name])
            sd_baseline_bmc_per_class_filtered_auprc[name] = np.std(cv_baseline_bmc_per_class_filtered_auprc[name])

        
        ### instantiate dictionaries for per-period metrics
        mean_baseline_per_period_auroc = {}
        mean_baseline_per_period_filtered_auroc = {}
        mean_baseline_bmc_per_period_auroc = {}
        mean_baseline_bmc_per_period_filtered_auroc = {}
        sd_baseline_per_period_auroc = {}
        sd_baseline_per_period_filtered_auroc = {}
        sd_baseline_bmc_per_period_auroc = {}
        sd_baseline_bmc_per_period_filtered_auroc = {}

        ### instantiate dictionaries for by insurance metrics
        mean_baseline_by_insurance_auroc = {}
        mean_baseline_by_insurance_filtered_auroc = {}
        mean_baseline_bmc_by_insurance_auroc = {}
        mean_baseline_bmc_by_insurance_filtered_auroc = {}
        sd_baseline_by_insurance_auroc = {}
        sd_baseline_by_insurance_filtered_auroc = {}
        sd_baseline_bmc_by_insurance_auroc = {}
        sd_baseline_bmc_by_insurance_filtered_auroc = {}

        ### fill per-period metric dictionaries with means + sd
        for i in range(4):
            mean_baseline_per_period_auroc[periods[i]] = np.mean(cv_baseline_per_period_auroc[periods[i]])
            mean_baseline_per_period_filtered_auroc[periods[i]] = np.mean(cv_baseline_per_period_filtered_auroc[periods[i]])
            mean_baseline_bmc_per_period_auroc[periods[i]] = np.mean(cv_baseline_bmc_per_period_auroc[periods[i]])
            mean_baseline_bmc_per_period_filtered_auroc[periods[i]] = np.mean(cv_baseline_bmc_per_period_filtered_auroc[periods[i]])
            sd_baseline_per_period_auroc[periods[i]] = np.std(cv_baseline_per_period_auroc[periods[i]])
            sd_baseline_per_period_filtered_auroc[periods[i]] = np.std(cv_baseline_per_period_filtered_auroc[periods[i]])
            sd_baseline_bmc_per_period_auroc[periods[i]] = np.std(cv_baseline_bmc_per_period_auroc[periods[i]])
            sd_baseline_bmc_per_period_filtered_auroc[periods[i]] = np.std(cv_baseline_bmc_per_period_filtered_auroc[periods[i]])

        ### fill by insurance metric dictionaries with means + sd
        for i in range(3):
            mean_baseline_by_insurance_auroc[insurance[i]] = np.mean(cv_baseline_by_insurance_auroc[insurance[i]])
            mean_baseline_by_insurance_filtered_auroc[insurance[i]] = np.mean(cv_baseline_by_insurance_filtered_auroc[insurance[i]])
            mean_baseline_bmc_by_insurance_auroc[insurance[i]] = np.mean(cv_baseline_bmc_by_insurance_auroc[insurance[i]])
            mean_baseline_bmc_by_insurance_filtered_auroc[insurance[i]] = np.mean(cv_baseline_bmc_by_insurance_filtered_auroc[insurance[i]])
            sd_baseline_by_insurance_auroc[insurance[i]] = np.std(cv_baseline_by_insurance_auroc[insurance[i]])
            sd_baseline_by_insurance_filtered_auroc[insurance[i]] = np.std(cv_baseline_by_insurance_filtered_auroc[insurance[i]])
            sd_baseline_bmc_by_insurance_auroc[insurance[i]] = np.std(cv_baseline_bmc_by_insurance_auroc[insurance[i]])
            sd_baseline_bmc_by_insurance_filtered_auroc[insurance[i]] = np.std(cv_baseline_bmc_by_insurance_filtered_auroc[insurance[i]])

        ### calculate mean and sd for average metrics
        mean_baseline_overall_accuracy = np.mean(cv_baseline_overall_accuracy)
        mean_baseline_filtered_accuracy = np.mean(cv_baseline_filtered_accuracy)
        mean_baseline_bmc_overall_accuracy = np.mean(cv_baseline_bmc_overall_accuracy)
        mean_baseline_bmc_filtered_accuracy = np.mean(cv_baseline_bmc_filtered_accuracy)
        sd_baseline_overall_accuracy = np.std(cv_baseline_overall_accuracy)
        sd_baseline_filtered_accuracy = np.std(cv_baseline_filtered_accuracy)
        sd_baseline_bmc_overall_accuracy = np.std(cv_baseline_bmc_overall_accuracy)
        sd_baseline_bmc_filtered_accuracy = np.std(cv_baseline_bmc_filtered_accuracy)

        mean_baseline_overall_auroc = np.mean(cv_baseline_overall_auroc)
        mean_baseline_filtered_auroc = np.mean(cv_baseline_filtered_auroc)
        mean_baseline_bmc_overall_auroc = np.mean(cv_baseline_bmc_overall_auroc)
        mean_baseline_bmc_filtered_auroc = np.mean(cv_baseline_bmc_filtered_auroc)
        sd_baseline_overall_auroc = np.std(cv_baseline_overall_auroc)
        sd_baseline_filtered_auroc = np.std(cv_baseline_filtered_auroc)
        sd_baseline_bmc_overall_auroc = np.std(cv_baseline_bmc_overall_auroc)
        sd_baseline_bmc_filtered_auroc = np.std(cv_baseline_bmc_filtered_auroc)

        mean_baseline_overall_auprc = np.mean(cv_baseline_overall_auprc)
        mean_baseline_filtered_auprc = np.mean(cv_baseline_filtered_auprc)
        mean_baseline_bmc_overall_auprc = np.mean(cv_baseline_bmc_overall_auprc)
        mean_baseline_bmc_filtered_auprc = np.mean(cv_baseline_bmc_filtered_auprc)
        sd_baseline_overall_auprc = np.std(cv_baseline_overall_auprc)
        sd_baseline_filtered_auprc = np.std(cv_baseline_filtered_auprc)
        sd_baseline_bmc_overall_auprc = np.std(cv_baseline_bmc_overall_auprc)
        sd_baseline_bmc_filtered_auprc = np.std(cv_baseline_bmc_filtered_auprc)

        ### calculate mean + sd for curve plot data (use only cv 4)
        #### precision-recall curves
        mean_baseline_precision, sd_baseline_precision = average_across_lists([[d['precision'] for d in dict] for dict in cv_baseline_overall_prc_data][4])
        mean_baseline_recall, sd_baseline_recall = average_across_lists([[d['recall'] for d in dict] for dict in cv_baseline_overall_prc_data][4])
        mean_baseline_prc_auc, sd_baseline_prc_auc = np.mean([[d['auc'] for d in dict] for dict in cv_baseline_overall_prc_data][4]), np.std([[d['auc'] for d in dict] for dict in cv_baseline_overall_prc_data][4])

        mean_baseline_filtered_precision, sd_baseline_filtered_precision = average_across_lists([[d['precision'] for d in dict] for dict in cv_baseline_filtered_prc_data][4])
        mean_baseline_filtered_recall, sd_baseline_filtered_recall = average_across_lists([[d['recall'] for d in dict] for dict in cv_baseline_filtered_prc_data][4])
        mean_baseline_filtered_prc_auc, sd_baseline_filtered_prc_auc = np.mean([[d['auc'] for d in dict] for dict in cv_baseline_filtered_prc_data][4]), np.std([[d['auc'] for d in dict] for dict in cv_baseline_filtered_prc_data][4])

        mean_baseline_bmc_precision, sd_baseline_bmc_precision = average_across_lists([[d['precision'] for d in dict] for dict in cv_baseline_bmc_overall_prc_data][4])
        mean_baseline_bmc_recall, sd_baseline_bmc_recall = average_across_lists([[d['recall'] for d in dict] for dict in cv_baseline_bmc_overall_prc_data][4])
        mean_baseline_bmc_prc_auc, sd_baseline_bmc_prc_auc = np.mean([[d['auc'] for d in dict] for dict in cv_baseline_bmc_overall_prc_data][4]), np.std([[d['auc'] for d in dict] for dict in cv_baseline_bmc_overall_prc_data][4])

        mean_baseline_bmc_filtered_precision, sd_baseline_bmc_filtered_precision = average_across_lists([[d['precision'] for d in dict] for dict in cv_baseline_bmc_filtered_prc_data][4])
        mean_baseline_bmc_filtered_recall, sd_baseline_bmc_filtered_recall = average_across_lists([[d['recall'] for d in dict] for dict in cv_baseline_bmc_filtered_prc_data][4])
        mean_baseline_bmc_filtered_prc_auc, sd_baseline_bmc_filtered_prc_auc = np.mean([[d['auc'] for d in dict] for dict in cv_baseline_bmc_filtered_prc_data][4]), np.std([[d['auc'] for d in dict] for dict in cv_baseline_bmc_filtered_prc_data][4])

        #### receiver operating curves
        mean_baseline_fpr, sd_baseline_fpr = average_across_lists([[d['fpr'] for d in dict] for dict in cv_baseline_overall_roc_data][4])
        mean_baseline_tpr, sd_baseline_tpr = average_across_lists([[d['tpr'] for d in dict] for dict in cv_baseline_overall_roc_data][4])
        mean_baseline_roc_auc, sd_baseline_roc_auc = np.mean([[d['auc'] for d in dict] for dict in cv_baseline_overall_roc_data][4]), np.std([[d['auc'] for d in dict] for dict in cv_baseline_overall_roc_data][4])

        mean_baseline_filtered_fpr, sd_baseline_filtered_fpr = average_across_lists([[d['fpr'] for d in dict] for dict in cv_baseline_filtered_roc_data][4])
        mean_baseline_filtered_tpr, sd_baseline_filtered_tpr = average_across_lists([[d['tpr'] for d in dict] for dict in cv_baseline_filtered_roc_data][4])
        mean_baseline_filtered_roc_auc, sd_baseline_filtered_roc_auc = np.mean([[d['auc'] for d in dict] for dict in cv_baseline_filtered_roc_data][4]), np.std([[d['auc'] for d in dict] for dict in cv_baseline_filtered_roc_data][4])

        mean_baseline_bmc_fpr, sd_baseline_bmc_fpr = average_across_lists([[d['fpr'] for d in dict] for dict in cv_baseline_bmc_overall_roc_data][4])
        mean_baseline_bmc_tpr, sd_baseline_bmc_tpr = average_across_lists([[d['tpr'] for d in dict] for dict in cv_baseline_bmc_overall_roc_data][4])
        mean_baseline_bmc_roc_auc, sd_baseline_bmc_roc_auc = np.mean([[d['auc'] for d in dict] for dict in cv_baseline_bmc_overall_roc_data][4]), np.std([[d['auc'] for d in dict] for dict in cv_baseline_bmc_overall_roc_data][4])

        mean_baseline_bmc_filtered_fpr, sd_baseline_bmc_filtered_fpr = average_across_lists([[d['fpr'] for d in dict] for dict in cv_baseline_bmc_filtered_roc_data][4])
        mean_baseline_bmc_filtered_tpr, sd_baseline_bmc_filtered_tpr = average_across_lists([[d['tpr'] for d in dict] for dict in cv_baseline_bmc_filtered_roc_data][4])
        mean_baseline_bmc_filtered_roc_auc, sd_baseline_bmc_filtered_roc_auc = np.mean([[d['auc'] for d in dict] for dict in cv_baseline_bmc_filtered_roc_data][4]), np.std([[d['auc'] for d in dict] for dict in cv_baseline_bmc_filtered_roc_data][4])

    ## xgboost metrics
    ### per class metrics
    mean_xgb_per_class_accuracy = {}
    mean_xgb_per_class_filtered_accuracy = {}
    mean_xgb_bmc_per_class_accuracy = {}
    mean_xgb_bmc_per_class_filtered_accuracy = {}
    sd_xgb_per_class_accuracy = {}
    sd_xgb_per_class_filtered_accuracy = {}
    sd_xgb_bmc_per_class_accuracy = {}
    sd_xgb_bmc_per_class_filtered_accuracy = {}

    mean_xgb_per_class_auroc = {}
    mean_xgb_per_class_filtered_auroc = {}
    mean_xgb_bmc_per_class_auroc = {}
    mean_xgb_bmc_per_class_filtered_auroc = {}
    sd_xgb_per_class_auroc = {}
    sd_xgb_per_class_filtered_auroc = {}
    sd_xgb_bmc_per_class_auroc = {}
    sd_xgb_bmc_per_class_filtered_auroc = {}

    mean_xgb_per_class_auprc = {}
    mean_xgb_per_class_filtered_auprc = {}
    mean_xgb_bmc_per_class_auprc = {}
    mean_xgb_bmc_per_class_filtered_auprc = {}
    sd_xgb_per_class_auprc = {}
    sd_xgb_per_class_filtered_auprc = {}
    sd_xgb_bmc_per_class_auprc = {}
    sd_xgb_bmc_per_class_filtered_auprc = {}

    ### fill per-class metric dictionaries with means + sd
    for i in range(4):
        name = classes[i]

        mean_xgb_per_class_accuracy[name] = np.mean(cv_xgb_per_class_accuracy[name])
        mean_xgb_per_class_filtered_accuracy[name] = np.mean(cv_xgb_per_class_filtered_accuracy[name])
        mean_xgb_bmc_per_class_accuracy[name] = np.mean(cv_xgb_bmc_per_class_accuracy[name])
        mean_xgb_bmc_per_class_filtered_accuracy[name] = np.mean(cv_xgb_bmc_per_class_filtered_accuracy[name])
        sd_xgb_per_class_accuracy[name] = np.std(cv_xgb_per_class_accuracy[name])
        sd_xgb_per_class_filtered_accuracy[name] = np.std(cv_xgb_per_class_filtered_accuracy[name])
        sd_xgb_bmc_per_class_accuracy[name] = np.std(cv_xgb_bmc_per_class_accuracy[name])
        sd_xgb_bmc_per_class_filtered_accuracy[name] = np.std(cv_xgb_bmc_per_class_filtered_accuracy[name])

        mean_xgb_per_class_auroc[name] = np.mean(cv_xgb_per_class_auroc[name])
        mean_xgb_per_class_filtered_auroc[name] = np.mean(cv_xgb_per_class_filtered_auroc[name])
        mean_xgb_bmc_per_class_auroc[name] = np.mean(cv_xgb_bmc_per_class_auroc[name])
        mean_xgb_bmc_per_class_filtered_auroc[name] = np.mean(cv_xgb_bmc_per_class_filtered_auroc[name])
        sd_xgb_per_class_auroc[name] = np.std(cv_xgb_per_class_auroc[name])
        sd_xgb_per_class_filtered_auroc[name] = np.std(cv_xgb_per_class_filtered_auroc[name])
        sd_xgb_bmc_per_class_auroc[name] = np.std(cv_xgb_bmc_per_class_auroc[name])
        sd_xgb_bmc_per_class_filtered_auroc[name] = np.std(cv_xgb_bmc_per_class_filtered_auroc[name])

        mean_xgb_per_class_auprc[name] = np.mean(cv_xgb_per_class_auprc[name])
        mean_xgb_per_class_filtered_auprc[name] = np.mean(cv_xgb_per_class_filtered_auprc[name])
        mean_xgb_bmc_per_class_auprc[name] = np.mean(cv_xgb_bmc_per_class_auprc[name])
        mean_xgb_bmc_per_class_filtered_auprc[name] = np.mean(cv_xgb_bmc_per_class_filtered_auprc[name])
        sd_xgb_per_class_auprc[name] = np.std(cv_xgb_per_class_auprc[name])
        sd_xgb_per_class_filtered_auprc[name] = np.std(cv_xgb_per_class_filtered_auprc[name])
        sd_xgb_bmc_per_class_auprc[name] = np.std(cv_xgb_bmc_per_class_auprc[name])
        sd_xgb_bmc_per_class_filtered_auprc[name] = np.std(cv_xgb_bmc_per_class_filtered_auprc[name])

    ### per period metrics
    mean_xgb_per_period_auroc = {}
    mean_xgb_per_period_filtered_auroc = {}
    mean_xgb_bmc_per_period_auroc = {}
    mean_xgb_bmc_per_period_filtered_auroc = {}
    sd_xgb_per_period_auroc = {}
    sd_xgb_per_period_filtered_auroc = {}
    sd_xgb_bmc_per_period_auroc = {}
    sd_xgb_bmc_per_period_filtered_auroc = {}

    ### fill per-period metric dictionaries with means + sd
    for i in range(4):
        mean_xgb_per_period_auroc[periods[i]] = np.mean(cv_xgb_per_period_auroc[periods[i]])
        mean_xgb_per_period_filtered_auroc[periods[i]] = np.mean(cv_xgb_per_period_filtered_auroc[periods[i]])
        mean_xgb_bmc_per_period_auroc[periods[i]] = np.mean(cv_xgb_bmc_per_period_auroc[periods[i]])
        mean_xgb_bmc_per_period_filtered_auroc[periods[i]] = np.mean(cv_xgb_bmc_per_period_filtered_auroc[periods[i]])
        sd_xgb_per_period_auroc[periods[i]] = np.std(cv_xgb_per_period_auroc[periods[i]])
        sd_xgb_per_period_filtered_auroc[periods[i]] = np.std(cv_xgb_per_period_filtered_auroc[periods[i]])
        sd_xgb_bmc_per_period_auroc[periods[i]] = np.std(cv_xgb_bmc_per_period_auroc[periods[i]])
        sd_xgb_bmc_per_period_filtered_auroc[periods[i]] = np.std(cv_xgb_bmc_per_period_filtered_auroc[periods[i]])


    ### by insurance metrics
    mean_xgb_by_insurance_auroc = {}
    mean_xgb_by_insurance_filtered_auroc = {}
    mean_xgb_bmc_by_insurance_auroc = {}
    mean_xgb_bmc_by_insurance_filtered_auroc = {}
    sd_xgb_by_insurance_auroc = {}
    sd_xgb_by_insurance_filtered_auroc = {}
    sd_xgb_bmc_by_insurance_auroc = {}
    sd_xgb_bmc_by_insurance_filtered_auroc = {}

    ### fill by insurance metric dictionaries with means + sd
    for i in range(3):
        mean_xgb_by_insurance_auroc[insurance[i]] = np.mean(cv_xgb_by_insurance_auroc[insurance[i]])
        mean_xgb_by_insurance_filtered_auroc[insurance[i]] = np.mean(cv_xgb_by_insurance_filtered_auroc[insurance[i]])
        mean_xgb_bmc_by_insurance_auroc[insurance[i]] = np.mean(cv_xgb_bmc_by_insurance_auroc[insurance[i]])
        mean_xgb_bmc_by_insurance_filtered_auroc[insurance[i]] = np.mean(cv_xgb_bmc_by_insurance_filtered_auroc[insurance[i]])
        sd_xgb_by_insurance_auroc[insurance[i]] = np.std(cv_xgb_by_insurance_auroc[insurance[i]])
        sd_xgb_by_insurance_filtered_auroc[insurance[i]] = np.std(cv_xgb_by_insurance_filtered_auroc[insurance[i]])
        sd_xgb_bmc_by_insurance_auroc[insurance[i]] = np.std(cv_xgb_bmc_by_insurance_auroc[insurance[i]])
        sd_xgb_bmc_by_insurance_filtered_auroc[insurance[i]] = np.std(cv_xgb_bmc_by_insurance_filtered_auroc[insurance[i]])

    ### calculate mean and sd for average metrics
    mean_xgb_overall_accuracy = np.mean(cv_xgb_overall_accuracy)
    mean_xgb_filtered_accuracy = np.mean(cv_xgb_filtered_accuracy)
    mean_xgb_bmc_overall_accuracy = np.mean(cv_xgb_bmc_overall_accuracy)
    mean_xgb_bmc_filtered_accuracy = np.mean(cv_xgb_bmc_filtered_accuracy)
    sd_xgb_overall_accuracy = np.std(cv_xgb_overall_accuracy)
    sd_xgb_filtered_accuracy = np.std(cv_xgb_filtered_accuracy)
    sd_xgb_bmc_overall_accuracy = np.std(cv_xgb_bmc_overall_accuracy)
    sd_xgb_bmc_filtered_accuracy = np.std(cv_xgb_bmc_filtered_accuracy)

    mean_xgb_overall_auroc = np.mean(cv_xgb_overall_auroc)
    mean_xgb_filtered_auroc = np.mean(cv_xgb_filtered_auroc)
    mean_xgb_bmc_overall_auroc = np.mean(cv_xgb_bmc_overall_auroc)
    mean_xgb_bmc_filtered_auroc = np.mean(cv_xgb_bmc_filtered_auroc)
    sd_xgb_overall_auroc = np.std(cv_xgb_overall_auroc)
    sd_xgb_filtered_auroc = np.std(cv_xgb_filtered_auroc)
    sd_xgb_bmc_overall_auroc = np.std(cv_xgb_bmc_overall_auroc)
    sd_xgb_bmc_filtered_auroc = np.std(cv_xgb_bmc_filtered_auroc)

    mean_xgb_overall_auprc = np.mean(cv_xgb_overall_auprc)
    mean_xgb_filtered_auprc = np.mean(cv_xgb_filtered_auprc)
    mean_xgb_bmc_overall_auprc = np.mean(cv_xgb_bmc_overall_auprc)
    mean_xgb_bmc_filtered_auprc = np.mean(cv_xgb_bmc_filtered_auprc)
    sd_xgb_overall_auprc = np.std(cv_xgb_overall_auprc)
    sd_xgb_filtered_auprc = np.std(cv_xgb_filtered_auprc)
    sd_xgb_bmc_overall_auprc = np.std(cv_xgb_bmc_overall_auprc)
    sd_xgb_bmc_filtered_auprc = np.std(cv_xgb_bmc_filtered_auprc)

    ### calculate mean + sd for curve plot data (use only cv 4)
    #### precision-recall curves
    mean_xgb_precision, sd_xgb_precision = average_across_lists([[d['precision'] for d in dict] for dict in cv_xgb_overall_prc_data][4])
    mean_xgb_recall, sd_xgb_recall = average_across_lists([[d['recall'] for d in dict] for dict in cv_xgb_overall_prc_data][4])
    mean_xgb_prc_auc, sd_xgb_prc_auc = np.mean([[d['auc'] for d in dict] for dict in cv_xgb_overall_prc_data][4]), np.std([[d['auc'] for d in dict] for dict in cv_xgb_overall_prc_data][4])

    mean_xgb_filtered_precision, sd_xgb_filtered_precision = average_across_lists([[d['precision'] for d in dict] for dict in cv_xgb_filtered_prc_data][4])
    mean_xgb_filtered_recall, sd_xgb_filtered_recall = average_across_lists([[d['recall'] for d in dict] for dict in cv_xgb_filtered_prc_data][4])
    mean_xgb_filtered_prc_auc, sd_xgb_filtered_prc_auc = np.mean([[d['auc'] for d in dict] for dict in cv_xgb_filtered_prc_data][4]), np.std([[d['auc'] for d in dict] for dict in cv_xgb_filtered_prc_data][4])

    mean_xgb_bmc_precision, sd_xgb_bmc_precision = average_across_lists([[d['precision'] for d in dict] for dict in cv_xgb_bmc_overall_prc_data][4])
    mean_xgb_bmc_recall, sd_xgb_bmc_recall = average_across_lists([[d['recall'] for d in dict] for dict in cv_xgb_bmc_overall_prc_data][4])
    mean_xgb_bmc_prc_auc, sd_xgb_bmc_prc_auc = np.mean([[d['auc'] for d in dict] for dict in cv_xgb_bmc_overall_prc_data][4]), np.std([[d['auc'] for d in dict] for dict in cv_xgb_bmc_overall_prc_data][4])

    mean_xgb_bmc_filtered_precision, sd_xgb_bmc_filtered_precision = average_across_lists([[d['precision'] for d in dict] for dict in cv_xgb_bmc_filtered_prc_data][4])
    mean_xgb_bmc_filtered_recall, sd_xgb_bmc_filtered_recall = average_across_lists([[d['recall'] for d in dict] for dict in cv_xgb_bmc_filtered_prc_data][4])
    mean_xgb_bmc_filtered_prc_auc, sd_xgb_bmc_filtered_prc_auc = np.mean([[d['auc'] for d in dict] for dict in cv_xgb_bmc_filtered_prc_data][4]), np.std([[d['auc'] for d in dict] for dict in cv_xgb_bmc_filtered_prc_data][4])

    #### receiver operating curves
    mean_xgb_fpr, sd_xgb_fpr = average_across_lists([[d['fpr'] for d in dict] for dict in cv_xgb_overall_roc_data][4])
    mean_xgb_tpr, sd_xgb_tpr = average_across_lists([[d['tpr'] for d in dict] for dict in cv_xgb_overall_roc_data][4])
    mean_xgb_roc_auc, sd_xgb_roc_auc = np.mean([[d['auc'] for d in dict] for dict in cv_xgb_overall_roc_data][4]), np.std([[d['auc'] for d in dict] for dict in cv_xgb_overall_roc_data][4])

    mean_xgb_filtered_fpr, sd_xgb_filtered_fpr = average_across_lists([[d['fpr'] for d in dict] for dict in cv_xgb_filtered_roc_data][4])
    mean_xgb_filtered_tpr, sd_xgb_filtered_tpr = average_across_lists([[d['tpr'] for d in dict] for dict in cv_xgb_filtered_roc_data][4])
    mean_xgb_filtered_roc_auc, sd_xgb_filtered_roc_auc = np.mean([[d['auc'] for d in dict] for dict in cv_xgb_filtered_roc_data][4]), np.std([[d['auc'] for d in dict] for dict in cv_xgb_filtered_roc_data][4])

    mean_xgb_bmc_fpr, sd_xgb_bmc_fpr = average_across_lists([[d['fpr'] for d in dict] for dict in cv_xgb_bmc_overall_roc_data][4])
    mean_xgb_bmc_tpr, sd_xgb_bmc_tpr = average_across_lists([[d['tpr'] for d in dict] for dict in cv_xgb_bmc_overall_roc_data][4])
    mean_xgb_bmc_roc_auc, sd_xgb_bmc_roc_auc = np.mean([[d['auc'] for d in dict] for dict in cv_xgb_bmc_overall_roc_data][4]), np.std([[d['auc'] for d in dict] for dict in cv_xgb_bmc_overall_roc_data][4])

    mean_xgb_bmc_filtered_fpr, sd_xgb_bmc_filtered_fpr = average_across_lists([[d['fpr'] for d in dict] for dict in cv_xgb_bmc_filtered_roc_data][4])
    mean_xgb_bmc_filtered_tpr, sd_xgb_bmc_filtered_tpr = average_across_lists([[d['tpr'] for d in dict] for dict in cv_xgb_bmc_filtered_roc_data][4])
    mean_xgb_bmc_filtered_roc_auc, sd_xgb_bmc_filtered_roc_auc = np.mean([[d['auc'] for d in dict] for dict in cv_xgb_bmc_filtered_roc_data][4]), np.std([[d['auc'] for d in dict] for dict in cv_xgb_bmc_filtered_roc_data][4])

    # Save computed metrics to wandb
    ## baseline metrics
    if not DEMO_MODE:
        save_metrics("baseline", "mgb", "", mean_baseline_overall_accuracy, sd_baseline_overall_accuracy, mean_baseline_overall_auroc, sd_baseline_overall_auroc, mean_baseline_overall_auprc, sd_baseline_overall_auprc)
        save_metrics("baseline", "mgb", "filtered ", mean_baseline_filtered_accuracy, sd_baseline_filtered_accuracy, mean_baseline_filtered_auroc, sd_baseline_filtered_auroc, mean_baseline_filtered_auprc, sd_baseline_filtered_auprc)
        save_metrics("baseline", "bmc", "", mean_baseline_bmc_overall_accuracy, sd_baseline_bmc_overall_accuracy, mean_baseline_bmc_overall_auroc, sd_baseline_bmc_overall_auroc, mean_baseline_bmc_overall_auprc, sd_baseline_bmc_overall_auprc)
        save_metrics("baseline", "bmc", "filtered ", mean_baseline_bmc_filtered_accuracy, sd_baseline_bmc_filtered_accuracy, mean_baseline_bmc_filtered_auroc, sd_baseline_bmc_filtered_auroc, mean_baseline_bmc_filtered_auprc, sd_baseline_bmc_filtered_auprc)

        for i in range(4):
            save_auroc_only("baseline", "mgb", periods[i], mean_baseline_per_period_auroc[periods[i]], sd_baseline_per_period_auroc[periods[i]])
            save_auroc_only("baseline", "mgb", "filtered " + periods[i], mean_baseline_per_period_filtered_auroc[periods[i]], sd_baseline_per_period_filtered_auroc[periods[i]])
            save_auroc_only("baseline", "bmc", periods[i], mean_baseline_bmc_per_period_auroc[periods[i]], sd_baseline_bmc_per_period_auroc[periods[i]])
            save_auroc_only("baseline", "bmc", "filtered " + periods[i], mean_baseline_bmc_per_period_filtered_auroc[periods[i]], sd_baseline_bmc_per_period_filtered_auroc[periods[i]])
            
        for i in range(3):
            save_auroc_only("baseline", "mgb", insurance[i], mean_baseline_by_insurance_auroc[insurance[i]], sd_baseline_by_insurance_auroc[insurance[i]])
            save_auroc_only("baseline", "mgb", "filtered " + insurance[i], mean_baseline_by_insurance_filtered_auroc[insurance[i]], sd_baseline_by_insurance_filtered_auroc[insurance[i]])
            save_auroc_only("baseline", "bmc", insurance[i], mean_baseline_bmc_by_insurance_auroc[insurance[i]], sd_baseline_bmc_by_insurance_auroc[insurance[i]])
            save_auroc_only("baseline", "bmc", "filtered " + insurance[i], mean_baseline_bmc_by_insurance_filtered_auroc[insurance[i]], sd_baseline_bmc_by_insurance_filtered_auroc[insurance[i]])

    save_metrics("xgb", "mgb", "", mean_xgb_overall_accuracy, sd_xgb_overall_accuracy, mean_xgb_overall_auroc, sd_xgb_overall_auroc, mean_xgb_overall_auprc, sd_xgb_overall_auprc)
    save_metrics("xgb", "mgb", "filtered ", mean_xgb_filtered_accuracy, sd_xgb_filtered_accuracy, mean_xgb_filtered_auroc, sd_xgb_filtered_auroc, mean_xgb_filtered_auprc, sd_xgb_filtered_auprc)
    save_metrics("xgb", "bmc", "", mean_xgb_bmc_overall_accuracy, sd_xgb_bmc_overall_accuracy, mean_xgb_bmc_overall_auroc, sd_xgb_bmc_overall_auroc, mean_xgb_bmc_overall_auprc, sd_xgb_bmc_overall_auprc)
    save_metrics("xgb", "bmc", "filtered ", mean_xgb_bmc_filtered_accuracy, sd_xgb_bmc_filtered_accuracy, mean_xgb_bmc_filtered_auroc, sd_xgb_bmc_filtered_auroc, mean_xgb_bmc_filtered_auprc, sd_xgb_bmc_filtered_auprc)

    if not DEMO_MODE:
        for i in range(4):
            save_auroc_only("xgb", "mgb", periods[i], mean_xgb_per_period_auroc[periods[i]], sd_xgb_per_period_auroc[periods[i]])
            save_auroc_only("xgb", "mgb", "filtered " + periods[i], mean_xgb_per_period_filtered_auroc[periods[i]], sd_xgb_per_period_filtered_auroc[periods[i]])
            save_auroc_only("xgb", "bmc", periods[i], mean_xgb_bmc_per_period_auroc[periods[i]], sd_xgb_bmc_per_period_auroc[periods[i]])
            save_auroc_only("xgb", "bmc", "filtered " + periods[i], mean_xgb_bmc_per_period_filtered_auroc[periods[i]], sd_xgb_bmc_per_period_filtered_auroc[periods[i]])
        
        for i in range(3):
            save_auroc_only("xgb", "mgb", insurance[i], mean_xgb_by_insurance_auroc[insurance[i]], sd_xgb_by_insurance_auroc[insurance[i]])
            save_auroc_only("xgb", "mgb", "filtered " + insurance[i], mean_xgb_by_insurance_filtered_auroc[insurance[i]], sd_xgb_by_insurance_filtered_auroc[insurance[i]])
            save_auroc_only("xgb", "bmc", insurance[i], mean_xgb_bmc_by_insurance_auroc[insurance[i]], sd_xgb_bmc_by_insurance_auroc[insurance[i]])
            save_auroc_only("xgb", "bmc", "filtered " + insurance[i], mean_xgb_bmc_by_insurance_filtered_auroc[insurance[i]], sd_xgb_bmc_by_insurance_filtered_auroc[insurance[i]])

    print("final metrics logged to wandb")

    # Generate and save combined curve plots
    ## restructure plot data 
    if not DEMO_MODE:
        prc_baseline_mgb = [[mean_baseline_recall, sd_baseline_recall], [mean_baseline_precision, sd_baseline_precision], [mean_baseline_prc_auc, sd_baseline_prc_auc]]
        prc_baseline_bmc = [[mean_baseline_bmc_recall, sd_baseline_bmc_recall], [mean_baseline_bmc_precision, sd_baseline_bmc_precision], [mean_baseline_bmc_prc_auc, sd_baseline_bmc_prc_auc]]

        roc_baseline_mgb = [[mean_baseline_fpr, sd_baseline_fpr], [mean_baseline_tpr, sd_baseline_tpr], [mean_baseline_roc_auc, sd_baseline_roc_auc]]
        roc_baseline_bmc = [[mean_baseline_bmc_fpr, sd_baseline_bmc_fpr], [mean_baseline_bmc_tpr, sd_baseline_bmc_tpr], [mean_baseline_bmc_roc_auc, sd_baseline_bmc_roc_auc]]

        filtered_prc_baseline_mgb = [[mean_baseline_filtered_recall, sd_baseline_filtered_recall], [mean_baseline_filtered_precision, sd_baseline_filtered_precision], [mean_baseline_filtered_prc_auc, sd_baseline_filtered_prc_auc]]
        filtered_prc_baseline_bmc = [[mean_baseline_bmc_filtered_recall, sd_baseline_bmc_filtered_recall], [mean_baseline_bmc_filtered_precision, sd_baseline_bmc_filtered_precision], [mean_baseline_bmc_filtered_prc_auc, sd_baseline_bmc_filtered_prc_auc]]

        filtered_roc_baseline_mgb = [[mean_baseline_filtered_fpr, sd_baseline_filtered_fpr], [mean_baseline_filtered_tpr, sd_baseline_filtered_tpr], [mean_baseline_filtered_roc_auc, sd_baseline_filtered_roc_auc]]
        filtered_roc_baseline_bmc = [[mean_baseline_bmc_filtered_fpr, sd_baseline_bmc_filtered_fpr], [mean_baseline_bmc_filtered_tpr, sd_baseline_bmc_filtered_tpr], [mean_baseline_bmc_filtered_roc_auc, sd_baseline_bmc_filtered_roc_auc]]

    filtered_roc_xgb_mgb = [[mean_xgb_filtered_fpr, sd_xgb_filtered_fpr], [mean_xgb_filtered_tpr, sd_xgb_filtered_tpr], [mean_xgb_filtered_roc_auc, sd_xgb_filtered_roc_auc]]
    filtered_roc_xgb_bmc = [[mean_xgb_bmc_filtered_fpr, sd_xgb_bmc_filtered_fpr], [mean_xgb_bmc_filtered_tpr, sd_xgb_bmc_filtered_tpr], [mean_xgb_bmc_filtered_roc_auc, sd_xgb_bmc_filtered_roc_auc]]
    filtered_prc_xgb_mgb = [[mean_xgb_filtered_recall, sd_xgb_filtered_recall], [mean_xgb_filtered_precision, sd_xgb_filtered_precision], [mean_xgb_filtered_prc_auc, sd_xgb_filtered_prc_auc]]
    filtered_prc_xgb_bmc = [[mean_xgb_bmc_filtered_recall, sd_xgb_bmc_filtered_recall], [mean_xgb_bmc_filtered_precision, sd_xgb_bmc_filtered_precision], [mean_xgb_bmc_filtered_prc_auc, sd_xgb_bmc_filtered_prc_auc]]
    prc_xgb_mgb = [[mean_xgb_recall, sd_xgb_recall], [mean_xgb_precision, sd_xgb_precision], [mean_xgb_prc_auc, sd_xgb_prc_auc]]
    prc_xgb_bmc = [[mean_xgb_bmc_recall, sd_xgb_bmc_recall], [mean_xgb_bmc_precision, sd_xgb_bmc_precision], [mean_xgb_bmc_prc_auc, sd_xgb_bmc_prc_auc]]
    roc_xgb_mgb = [[mean_xgb_fpr, sd_xgb_fpr], [mean_xgb_tpr, sd_xgb_tpr], [mean_xgb_roc_auc, sd_xgb_roc_auc]]
    roc_xgb_bmc = [[mean_xgb_bmc_fpr, sd_xgb_bmc_fpr], [mean_xgb_bmc_tpr, sd_xgb_bmc_tpr], [mean_xgb_bmc_roc_auc, sd_xgb_bmc_roc_auc]]

        ## save plot data to wandb
    if not DEMO_MODE:
        wandb.log({
            "baseline mgb prc data": prc_baseline_mgb,
            "baseline bmc prc data": prc_baseline_bmc,
            
            "baseline mgb filtered prc data": filtered_prc_baseline_mgb,
            "baseline bmc filtered prc data": filtered_prc_baseline_bmc,
            
            "baseline mgb roc data": roc_baseline_mgb,
            "baseline bmc roc data": roc_baseline_bmc,
        
            "baseline mgb filtered roc data": filtered_roc_baseline_mgb,
            "baseline bmc filtered roc data": filtered_roc_baseline_bmc,
        })
    
    wandb.log({
        "xgb mgb prc data": prc_xgb_mgb,
        "xgb bmc prc data": prc_xgb_bmc,
        "xgb mgb filtered prc data": filtered_prc_xgb_mgb,
        "xgb bmc filtered prc data": filtered_prc_xgb_bmc,
        "xgb mgb roc data": roc_xgb_mgb,
        "xgb bmc roc data": roc_xgb_bmc,
        "xgb mgb filtered roc data": filtered_roc_xgb_mgb,
        "xgb bmc filtered roc data": filtered_roc_xgb_bmc,
    })

        ## create and save combined plots 
    if not DEMO_MODE:
        create_combined_curve_plots(
            label = "", 
            prc_baseline_mgb = prc_baseline_mgb, 
            prc_baseline_bmc = prc_baseline_bmc, 
            prc_xgb_mgb = prc_xgb_mgb, 
            prc_xgb_bmc = prc_xgb_bmc, 
            roc_baseline_mgb = roc_baseline_mgb, 
            roc_baseline_bmc = roc_baseline_bmc, 
            roc_xgb_mgb = roc_xgb_mgb, 
            roc_xgb_bmc = roc_xgb_bmc
        )

        create_combined_curve_plots(
            label = "Filtered ", 
            prc_baseline_mgb = filtered_prc_baseline_mgb, 
            prc_baseline_bmc = filtered_prc_baseline_bmc, 
            prc_xgb_mgb = filtered_prc_xgb_mgb, 
            prc_xgb_bmc = filtered_prc_xgb_mgb, 
            roc_baseline_mgb = filtered_roc_baseline_mgb,
            roc_baseline_bmc = filtered_roc_baseline_bmc,
            roc_xgb_mgb = filtered_roc_xgb_mgb, 
            roc_xgb_bmc = filtered_roc_xgb_bmc
        )
        
        print("combined plots saved")


######################################################
# CODE EXECUTION

if __name__ == "__main__":
    
    # run model
    fire.Fire(train_xgb)
    print('Done with XGB training')