import fire
import pydantic
from pydantic import confloat, validate_arguments
import os
import sys
import numpy as np
from typing import Literal, List
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from wandb.integration.xgboost import WandbCallback
from sklearn.ensemble import RandomForestClassifier
import wandb
import pandas as pd
import pdb
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import tempfile
import torch
import cupy as cp

from xgboost import XGBClassifier

from helmet_mass_effect_pred.evaluation.cv_results import (
    get_results_per_cv,
    target_transform_partial,
)
from helmet_mass_effect_pred.data.splitting import drop_missing_impute_and_split
from helmet_mass_effect_pred.evaluation.visualisation import make_tsne_and_pca

from sklearn.model_selection import train_test_split

from torch import tensor as tnsr
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassAccuracy,
    MulticlassAveragePrecision,
)


@validate_arguments
def xgb_main(
    project: str = "bmc_data_exps",
    impute_method: Literal["simple", "iterative", "knn"] = "simple",
    time_NaN_fill: int = -10,
    lookback_hours=24,
    lookahead_hours=24,
    col_na_drop_fraction: confloat(ge=0.0, le=1.0) = 0.3,  # drop cols w/ % vals missing
    patient_na_drop_fraction: confloat(ge=0.0, le=1.0) = 0.2,  # drop rows w/ %  missing
    data_standard_scaling: bool = True,  # scale data to mean 0, std 1    learning_rate: confloat(ge=0.001, le=0.5) = 0.3,
    max_depth: int = 2,
    min_child_weight: int = 1,
    learning_rate: confloat(ge=0.001, le=0.5) = 0.058764,
    gamma: confloat(ge=0.0, le=5.0) = 4.7668,
    subsample: confloat(ge=0.5, le=1.0) = 0.9,
    reg_lambda: confloat(ge=1.0, le=10.0) = 8.0,
    reg_alpha: confloat(ge=0.0, le=5.0) = 4.061614,
    #no_transition_weighting: pydantic.Annotated[float, pydantic.Field(ge = 0, le = 1.0)] = 0.5, # map to power distribution, fix other hyperparams to best values from prior runs
    no_transition_weighting = 0.5,
    make_shap_plots: bool = True,
    create_pca_and_tsne: bool = False,
    remove_data_after_surgery_occurs: bool = True,
    use_avg_kernel_as_feature: bool = False,
    use_regression_as_feature: bool = False,
    remove_excess_features: bool = False,
):
    wandb.init(project=project, config=locals(), save_code=True)
    target_transform = lambda x: target_transform_partial(x, time_NaN_fill)
    # api = wandb.Api()
    # pdb.set_trace()

    if lookahead_hours == 24:
        OLTV = pd.read_json(os.getcwd() + "/data/processed/OLTV_24hr_LLM_features.json")
        bmc_OLTV = pd.read_json(os.getcwd() + "/data/processed/bmc_OLTV_24hr_LLM_features.json")
    elif lookahead_hours == 8:
        OLTV = pd.read_json(os.getcwd() + "/data/processed/OLTV_8hr_LLM_features.json")
        bmc_OLTV = pd.read_json(os.getcwd() + "/data/processed/bmc_OLTV_8hr_LLM_features.json")
    else: 
        print('Invalid Lookahead Hours')

    OTV = pd.read_json(os.getcwd() + "/data/processed/processed_OTV.json")
    bmc_OTV = pd.read_json(os.getcwd() + "/data/processed/processed_bmc_OTV.json")

    base_time = 'LSW'

    nddf = pd.read_csv("data/raw/OTV_BMC_4.csv")
    nddf[base_time] = pd.to_datetime(nddf[base_time])
    nddf["nddt"] = pd.to_datetime(nddf["nddt"])
    nddf["nddt"] = nddf["nddt"] - nddf[base_time]
    nddf["nddt"] = nddf["nddt"].dt.total_seconds() // 3600
    nddf["nddt"] = nddf["nddt"].fillna(int(time_NaN_fill))
    nddf = nddf[["CDW_MRN", "nddt"]]
    nddf.rename(columns={"CDW_MRN": "ptid"}, inplace=True)

    if any(col.startswith("ffill") for col in OLTV.columns):
        # remove ffill_ from column names
        OLTV.columns = [x.replace("ffill_", "") for x in OLTV.columns]
        bmc_OLTV.columns = [x.replace("ffill_", "") for x in bmc_OLTV.columns]

    if OLTV.columns.__contains__('rad_text'):
        OLTV.drop('rad_text', axis = 1, inplace = True)
        bmc_OLTV.drop('rad_text', axis = 1, inplace = True)

    if OLTV.columns.__contains__('target_mode'):
        OLTV.drop('target_mode', axis = 1, inplace = True)
        bmc_OLTV.drop('target_mode', axis = 1, inplace = True)

    targets = OLTV['targets'].copy()
    OLTV.drop('targets', axis = 1, inplace = True)
    bmc_targets = bmc_OLTV['targets'].copy()
    bmc_OLTV.drop('targets', axis = 1, inplace = True)

    (
        cv_train_ptids,
        cv_test_ptids,
        cv_train_OLTV,
        cv_test_OLTV,
        cv_train_targets,
        cv_test_targets,
        target_col_names,
        OLTV_columns,
    ) = drop_missing_impute_and_split(
        OLTV=OLTV,
        targets=targets,
        impute_method=impute_method,
        col_na_drop_fraction=col_na_drop_fraction,
        patient_na_drop_fraction=patient_na_drop_fraction,
        return_cross_validation=True,
    )

    (
        bmc_cv_train_ptids,
        bmc_cv_test_ptids,
        bmc_cv_train_OLTV,
        bmc_cv_test_OLTV,
        bmc_cv_train_targets,
        bmc_cv_test_targets,
        bmc_target_col_names,
        bmc_OLTV_columns,
    ) = drop_missing_impute_and_split(
        OLTV=bmc_OLTV,
        targets=bmc_targets,
        impute_method=impute_method,
        col_na_drop_fraction=col_na_drop_fraction,
        patient_na_drop_fraction=patient_na_drop_fraction,
        return_cross_validation=True,
    )

    cv_per_class_accuracy = {"0": [], "0_3": [], "3_8": [], "8_up": []}
    cv_per_class_auroc = {"0": [], "0_3": [], "3_8": [], "8_up": []}
    cv_per_class_auprc = {"0": [], "0_3": [], "3_8": [], "8_up": []}

    cv_per_period_auroc = {"<24": [], "24_48": [], "48_96": [], ">96": []}
    cv_by_insurance_auroc = {"medicaid": [], "uninsured": [], "other": []}

    cv_overall_accuracy = []
    cv_overall_auroc = []
    cv_overall_auprc = []

    cv_bmc_per_class_accuracy = {"0": [], "0_3": [], "3_8": [], "8_up": []}
    cv_bmc_per_class_auroc = {"0": [], "0_3": [], "3_8": [], "8_up": []}
    cv_bmc_per_class_auprc = {"0": [], "0_3": [], "3_8": [], "8_up": []}

    cv_bmc_per_period_auroc = {"<24": [], "24_48": [], "48_96": [], ">96": []}
    cv_bmc_by_insurance_auroc = {"medicaid": [], "uninsured": [], "other": []}

    cv_bmc_overall_accuracy = []
    cv_bmc_overall_auroc = []
    cv_bmc_overall_auprc = []

    filtered_cv_per_class_accuracy = {"0": [], "0_3": [], "3_8": [], "8_up": []}
    filtered_cv_per_class_auroc = {"0": [], "0_3": [], "3_8": [], "8_up": []}
    filtered_cv_per_class_auprc = {"0": [], "0_3": [], "3_8": [], "8_up": []}

    filtered_cv_per_period_auroc = {"<24": [], "24_48": [], "48_96": [], ">96": []}
    filtered_cv_by_insurance_auroc = {"medicaid": [], "uninsured": [], "other": []}

    filtered_cv_overall_accuracy = []
    filtered_cv_overall_auroc = []
    filtered_cv_overall_auprc = []

    filtered_cv_bmc_per_class_accuracy = {"0": [], "0_3": [], "3_8": [], "8_up": []}
    filtered_cv_bmc_per_class_auroc = {"0": [], "0_3": [], "3_8": [], "8_up": []}
    filtered_cv_bmc_per_class_auprc = {"0": [], "0_3": [], "3_8": [], "8_up": []}

    filtered_cv_bmc_per_period_auroc = {"<24": [], "24_48": [], "48_96": [], ">96": []}
    filtered_cv_bmc_by_insurance_auroc = {"medicaid": [], "uninsured": [], "other": []}
    
    filtered_cv_bmc_overall_accuracy = []
    filtered_cv_bmc_overall_auroc = []
    filtered_cv_bmc_overall_auprc = []


    cv_num = 0
    for (
        train_ptid,
        test_ptid,
        train_OLTV,
        test_OLTV,
        train_targets,
        test_targets,
        bmc_train_ptid,
        bmc_test_ptid,
        bmc_train_OLTV,
        bmc_test_OLTV,
        bmc_train_targets,
        bmc_test_targets,
    ) in zip(
        cv_train_ptids,
        cv_test_ptids,
        cv_train_OLTV,
        cv_test_OLTV,
        cv_train_targets,
        cv_test_targets,
        bmc_cv_train_ptids,
        bmc_cv_test_ptids,
        bmc_cv_train_OLTV,
        bmc_cv_test_OLTV,
        bmc_cv_train_targets,
        bmc_cv_test_targets,
    ):
        bmc_OLTV = np.concatenate([bmc_train_OLTV, bmc_test_OLTV])
        bmc_targets = np.concatenate([bmc_train_targets, bmc_test_targets])
        bmc_ptid = pd.concat([bmc_train_ptid, bmc_test_ptid])

        #concat test and train sets to make full mgh set 
        train_test_OLTV = np.concatenate([train_OLTV, test_OLTV])
        train_test_ptid = np.concatenate([train_ptid, test_ptid])

        def create_original_oltv(df, targets, ptid, columns):
            original_oltv = pd.DataFrame(df.copy(), columns=columns)
            original_oltv["targets"] = targets
            original_oltv["ptid"] = list(ptid)
            return original_oltv

        original_bmc_oltv = create_original_oltv(
            bmc_OLTV, bmc_targets, bmc_ptid, bmc_OLTV_columns
        )
        original_train_oltv = create_original_oltv(
            train_OLTV, train_targets, train_ptid, OLTV_columns
        )
        original_test_oltv = create_original_oltv(
            test_OLTV, test_targets, test_ptid, OLTV_columns
        )

        #transform targets
        train_targets = np.array(list(map(target_transform, train_targets)))
        test_targets = np.array(list(map(target_transform, test_targets)))
        train_test_targets = np.concatenate([train_targets, test_targets])
        bmc_targets = np.array(list(map(target_transform, bmc_targets)))

        if use_avg_kernel_as_feature:
            #make average transition kernels for both mgh and bmc
            from helmet_mass_effect_pred.data.transition_kernel import get_avg_transition_kernel
            mgh_avg_kernel = get_avg_transition_kernel(pd.DataFrame(train_test_OLTV, columns = OLTV_columns), train_test_targets, target_transform)
            bmc_avg_kernel = get_avg_transition_kernel(pd.DataFrame(bmc_OLTV, columns = bmc_OLTV_columns), bmc_targets, target_transform)

            #function to add avg_transition_kernels as new feature columns based on size_mls value
            def add_avg_kernel(df, df_cols, avg_kernel):

                size_mls_ind = df_cols.index("size_mls")

                avg_transition_0 = np.zeros(shape=df.shape[0])
                avg_transition_1 = np.zeros(shape=df.shape[0])
                avg_transition_2 = np.zeros(shape=df.shape[0])
                avg_transition_3 = np.zeros(shape=df.shape[0])

                for i, size in enumerate(df[:,size_mls_ind]):
                    if size == time_NaN_fill:
                        avg_transition_0[i] = time_NaN_fill
                        avg_transition_1[i] = time_NaN_fill
                        avg_transition_2[i] = time_NaN_fill
                        avg_transition_3[i] = time_NaN_fill
                    else:
                        size = target_transform(size)
                        avg_transition_0[i] = avg_kernel.iloc[size, 0]
                        avg_transition_1[i] = avg_kernel.iloc[size, 1]
                        avg_transition_2[i] = avg_kernel.iloc[size, 2]
                        avg_transition_3[i] = avg_kernel.iloc[size, 3]

                df = np.append(df, avg_transition_0.reshape(len(avg_transition_0), 1), axis = 1)
                df = np.append(df, avg_transition_1.reshape(len(avg_transition_1), 1), axis = 1)
                df = np.append(df, avg_transition_2.reshape(len(avg_transition_2), 1), axis = 1)
                df = np.append(df, avg_transition_3.reshape(len(avg_transition_3), 1), axis = 1)

                df_cols = df_cols.copy()

                for ii in range(4):
                    df_cols.append(f'avg_transition_{ii}')

                return df, df_cols
            
            #add avg transition kernel as feature columns
            train_OLTV, train_columns = add_avg_kernel(train_OLTV, OLTV_columns, mgh_avg_kernel)
            test_OLTV, test_columns = add_avg_kernel(test_OLTV, OLTV_columns, mgh_avg_kernel)
            bmc_OLTV, bmc_OLTV_columns_new = add_avg_kernel(bmc_OLTV, bmc_OLTV_columns, bmc_avg_kernel)
            train_test_OLTV, OLTV_columns_new = add_avg_kernel(train_test_OLTV, OLTV_columns, mgh_avg_kernel)
        else:
            OLTV_columns_new = OLTV_columns.copy()
            bmc_OLTV_columns_new = bmc_OLTV_columns.copy()


        prev_index = OLTV_columns_new.index('prev_mls')
        bmc_prev_index = bmc_OLTV_columns_new.index('prev_mls')

        train_OLTV[:,prev_index] = np.array(list(map(target_transform, train_OLTV[:,prev_index])))
        test_OLTV[:,prev_index] = np.array(list(map(target_transform, test_OLTV[:,prev_index])))
        train_test_OLTV[:, prev_index] = np.array(list(map(target_transform, train_test_OLTV[:, prev_index])))
        bmc_OLTV[:,bmc_prev_index] = np.array(list(map(target_transform, bmc_OLTV[:,bmc_prev_index])))
    
        #convert from np.array to pd.DataFrame
        train_test_OLTV = pd.DataFrame(train_test_OLTV, columns = OLTV_columns_new)
        train_OLTV = pd.DataFrame(train_OLTV, columns=OLTV_columns_new)
        test_OLTV = pd.DataFrame(test_OLTV, columns=OLTV_columns_new)
        bmc_OLTV = pd.DataFrame(bmc_OLTV, columns=bmc_OLTV_columns_new)

        #drop columns not found in the other data set
        bmc_extra_cols = list(set(bmc_OLTV.columns) - set(train_OLTV.columns))
        bmc_OLTV = bmc_OLTV.drop(bmc_extra_cols, axis = 1)

        mgh_extra_cols = list(set(train_OLTV.columns) - set(bmc_OLTV.columns))
        train_OLTV = train_OLTV.drop(mgh_extra_cols, axis = 1)
        test_OLTV = test_OLTV.drop(mgh_extra_cols, axis = 1)
        train_test_OLTV = train_test_OLTV.drop(mgh_extra_cols, axis = 1)

        OLTV_columns_new = list(train_OLTV.columns)
        bmc_OLTV_columns_new = list(bmc_OLTV.columns)

        #function to log mode baseline accuracy and prev mls baseline accuracy
        def create_mode_baseline(df, train_targets, ptids, type="train"):
            size_mls = df["size_mls"]
            size_mls = np.array(list(map(target_transform, size_mls)))
            size_mls_df = pd.DataFrame(size_mls, columns=["size_mls"])
            size_mls_df["targets"] = train_targets
            size_mls_df["ptid"] = ptids
            size_mls_df = size_mls_df[size_mls_df["size_mls"] != time_NaN_fill]
            # calculate the baseline accuracy

            total_accurate_mode = 0
            Y_mode = size_mls_df["targets"].mode().values.item()
            for i in range(4):
                Y_section = size_mls_df[size_mls_df["size_mls"] == i]
                total_accurate_mode += len(Y_section[Y_section["targets"] == Y_mode])

            overall_accuracy_mode = total_accurate_mode / len(size_mls_df)

            total_accurate_prev = 0
            skip = 0
            prev_pt = None
            for ii, val in enumerate(size_mls_df["targets"]):
                current_pt = size_mls_df.iloc[ii]["ptid"] 
                if prev_pt is None or current_pt != prev_pt:
                    skip += 1
                    prev_pt = current_pt
                    continue
                elif val == size_mls_df.iloc[ii-1]["size_mls"]:
                    total_accurate_prev += 1
            
            overall_accuracy_prev = total_accurate_prev / (size_mls_df.shape[0] - skip)

            wandb.log({f"{type} mode baseline accuracy": overall_accuracy_mode})
            wandb.log({f"{type} prev baseline accuracy": overall_accuracy_prev})

        #log mode and prev mls baseline accuracies
        create_mode_baseline(train_test_OLTV, train_test_targets, train_test_ptid, type="mgh")
        create_mode_baseline(bmc_OLTV, bmc_targets, bmc_ptid, type="bmc")

        #normalize data and create scalar dictionary
        if data_standard_scaling:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler().fit(train_OLTV)
            # save the scaler to a dictionary for each column name so we can use it later
            scaler_dict = {}
            for i, col in enumerate(OLTV_columns_new):
                scaler_dict[col] = [scaler.mean_[i], scaler.scale_[i]]
            train_OLTV = scaler.transform(train_OLTV)
            test_OLTV = scaler.transform(test_OLTV)
            bmc_OLTV = scaler.transform(bmc_OLTV)
            train_test_OLTV = scaler.transform(train_test_OLTV)

        #convert from np.array to pd.DataFrame
        train_test_OLTV = pd.DataFrame(train_test_OLTV, columns = OLTV_columns_new)
        train_OLTV = pd.DataFrame(train_OLTV, columns=OLTV_columns_new)
        test_OLTV = pd.DataFrame(test_OLTV, columns=OLTV_columns_new)
        bmc_OLTV = pd.DataFrame(bmc_OLTV, columns=bmc_OLTV_columns_new)

        #get total counts for each state in target space
        counts = np.unique(train_test_targets, return_counts=True)
        bmc_counts = np.unique(bmc_targets, return_counts=True)

        target_col_names = ["0", "0_3", "3_8", "8_up"]

        if create_pca_and_tsne:
            make_tsne_and_pca(
                train_targets, test_targets, train_OLTV, target_col_names, wandb
            )

        # print the number of samples in each category
        for i in range(np.unique(train_targets).shape[0]):
            print(
                f"train samples in category {target_col_names[i]}: {np.sum(train_targets == i)}"
            )
            print(
                f"test samples in category {target_col_names[i]}: {np.sum(test_targets == i)}"
            )

        #function to remove any data after surgery dt per pt
        def remove_data_after_surgery(data, original_data, targets, ptids):
            print("Removing data after surgery occurs")
            data["targets"] = targets
            data["ptid"] = ptids.values
            data = data[
                ~(
                    (original_data.dt >= original_data.surgdt_time_censored)
                    & (original_data.surgdt_time_censored > 0)
                )
            ]
            targets = data["targets"].values
            ptids = data["ptid"].values
            data = data.drop(columns=["targets"])
            data = data.drop(columns=["ptid"])
            return data, targets, ptids

        #remove all data per pt after surgery dt 
        if remove_data_after_surgery_occurs:
            train_OLTV, train_targets, train_ptid = remove_data_after_surgery(
                train_OLTV, original_train_oltv, train_targets, train_ptid
            )
            test_OLTV, test_targets, test_ptid = remove_data_after_surgery(
                test_OLTV, original_test_oltv, test_targets, test_ptid
            )
            bmc_OLTV, bmc_targets, bmc_ptid = remove_data_after_surgery(
                bmc_OLTV, original_bmc_oltv, bmc_targets, bmc_ptid
            )

        #function to denorm df using scaler_dict made earlier
        def de_normalise_data(data, scaler_dict):
            output = data.copy()
            for col in output.columns:
                output[col] = output[col] * scaler_dict[col][1] + scaler_dict[col][0]
            return output

        #create denorm version of datasets
        de_normalised_mgh_OLTV = de_normalise_data(train_test_OLTV, scaler_dict)
        de_normalised_test_OLTV = de_normalise_data(test_OLTV, scaler_dict)
        de_normalised_bmc_OLTV = de_normalise_data(bmc_OLTV, scaler_dict)

        #transform denorm mls to target space
        temp_size_mls_mgh = de_normalised_mgh_OLTV["size_mls"].apply(target_transform)
        temp_size_mls_test = de_normalised_test_OLTV["size_mls"].apply(target_transform)
        temp_size_mls_bmc = de_normalised_bmc_OLTV["size_mls"].apply(target_transform)
 
        #create filtered mgh dataset (only dt where transition occurs)
        filtered_mgh_OLTV = train_test_OLTV[temp_size_mls_mgh != train_test_targets]
        filtered_test_OLTV = test_OLTV[temp_size_mls_test != test_targets]
        filtered_mgh_targets = train_test_targets[temp_size_mls_mgh != train_test_targets]
        filtered_test_targets = test_targets[temp_size_mls_test != test_targets]
        filtered_mgh_ptid = train_test_ptid[temp_size_mls_mgh != train_test_targets]
        filtered_test_ptid = test_ptid[temp_size_mls_test != test_targets]

        #create filtered bmc dataset
        filtered_bmc_OLTV = bmc_OLTV[temp_size_mls_bmc != bmc_targets]
        filtered_bmc_targets = bmc_targets[temp_size_mls_bmc != bmc_targets]
        filtered_bmc_ptid = bmc_ptid[temp_size_mls_bmc != bmc_targets]

        #log filtered mode and prev mls baselines
        create_mode_baseline(filtered_bmc_OLTV, filtered_bmc_targets, filtered_bmc_ptid, type = "filtered bmc")
        create_mode_baseline(filtered_mgh_OLTV, filtered_mgh_targets, filtered_mgh_ptid, type = "filtered mgh")

        #function to log regression baseline accuracy
        def create_edema_score_baseline(OLTV_df, OTV_df, targets, ptids, type, cv_num):
            DF = OTV_df[["ptid", "gluc1", "stroke", "hba1c", "tpa", "mt"]]
            mls_components = pd.DataFrame(OLTV_df["size_mls"], columns=["size_mls"])
            mls_components["ptid"] = ptids
            mls_components["targets"] = targets
            DF = DF.merge(mls_components, on="ptid", how="inner")

            targets = DF["targets"]
            DF = DF.drop(columns=["targets", "ptid"], axis=1)

            x_train, x_test, y_train, y_test = train_test_split(
                DF, targets, test_size=0.2, shuffle=False
            )

            imputer = SimpleImputer()
            imputer.fit(x_train)
            x_train = imputer.transform(x_train)
            x_test = imputer.transform(x_test)

            edema_score_model = LogisticRegression().fit(x_train, y_train)
            baseline_accuracy = edema_score_model.score(x_test, y_test)
            proba = edema_score_model.predict_proba(x_test)

            if use_regression_as_feature:
                probs = edema_score_model.predict_proba(imputer.transform(DF))
                for i in range(4):
                    OLTV_df[f'regress_proba_{i}'] = probs[:,i]

            AUROC = MulticlassAUROC(num_classes=4, average="none")
            baseline_auroc = AUROC(tnsr(proba).float(), tnsr(y_test.values))

            AUPRC = MulticlassAveragePrecision(num_classes=4, average="none")
            baseline_auprc = AUPRC(tnsr(proba).float(), tnsr(y_test.values))

            mean_auroc = np.mean(baseline_auroc.numpy())
            sd_auroc = np.std(baseline_auroc.numpy())
            mean_auprc = np.mean(baseline_auprc.numpy())
            sd_auprc = np.std(baseline_auprc.numpy())

            wandb.log({f"{type} edema score baseline accuracy": baseline_accuracy})
            wandb.log({f"{type} edema score baseline mean AUROC": mean_auroc})
            wandb.log({f"{type} edema score baseline mean AUPRC": mean_auprc})

            wandb.log({f"{type} edema score baseline sd AUROC {cv_num}": sd_auroc})
            wandb.log({f"{type} edema score baseline sd AUPRC {cv_num}": sd_auprc})
                      

            wandb.log(
                {
                    f"{type} edema score baseline raw AUROC cv {cv_num}": baseline_auroc.numpy().tolist(),
                    f"{type} edema score baseline raw AUPRC cv {cv_num}": baseline_auprc.numpy().tolist(),
                }
            )

            print(f"{type} edema score baseline accuracy: ", baseline_accuracy)
            print(f"{type} edema score baseline AUROC: ", mean_auroc)

            return OLTV_df

        #log bmc regression baseline 
        bmc_OLTV = create_edema_score_baseline(bmc_OLTV, bmc_OTV, bmc_targets, bmc_ptid, "bmc", cv_num)
        train_test_OLTV = create_edema_score_baseline(train_test_OLTV, OTV, train_test_targets, train_test_ptid, "mgh", cv_num)
        train_OLTV = create_edema_score_baseline(train_OLTV, OTV, train_targets, train_ptid, "mgh (train)", cv_num)
        test_OLTV = create_edema_score_baseline(test_OLTV, OTV, test_targets, test_ptid, "mgh (test)", cv_num)

        

        #correct columns
        if use_regression_as_feature:
            for i in range(4):
                OLTV_columns_new.append(f'regress_proba_{i}')
                bmc_OLTV_columns_new.append(f'regress_proba_{i}')
        

        #log filtered regression baseline
        filtered_bmc_OLTV = create_edema_score_baseline(filtered_bmc_OLTV, bmc_OTV, filtered_bmc_targets, filtered_bmc_ptid, "filtered bmc", cv_num)
        filtered_mgh_OLTV = create_edema_score_baseline(filtered_mgh_OLTV, OTV, filtered_mgh_targets, filtered_mgh_ptid, "filtered mgh", cv_num)

        # apply dt scaler_dict transform to nddf
        nddf["nddt_transformed"] = (nddf["nddt"] - scaler_dict["dt"][1]) / scaler_dict[
            "dt"
        ][0]

        bmc_og_dt = de_normalised_bmc_OLTV.dt

        #instantiate xgb classifier
        device = "cpu"

        trained_model = XGBClassifier(
            callbacks=[
                WandbCallback(
                    log_model=True,
                    log_feature_importance=True,
                )
            ],
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            gamma=gamma,
            subsample=subsample,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            device = device,
            
        )

        #create sample weight array where transition = 1 and no_transition = no_transition_weighting variable
        sample_weights = np.ones(train_OLTV.shape[0])
        current_mls_denorm = train_OLTV["size_mls"] * scaler_dict["size_mls"][1] + scaler_dict["size_mls"][0]
        current_mls_denorm_in_target_space = current_mls_denorm.apply(target_transform)
        sample_weights[current_mls_denorm_in_target_space == train_targets] = no_transition_weighting
        
        reduced_cols_24 = [
                'aca_x', 'af', 'age_calc', 'aspects', 'aspects1_time_censored',
                'aspects1dt_time_censored', 'bun', 'bun1', 'cr', 'cr1', 'dbp', 'dbp1',
                'dt', 'firstmls_time_censored', 'firstmlsdt_time_censored', 'gluc', 'gluc1',
                'htn', 'LLM_24_0', 'LLM_24_1', 'LLM_24_2', 'LLM_24_3', 'LLM_36_0', 'LLM_36_1',
                'LLM_36_2', 'LLM_36_3', 'LLM_8_0', 'LLM_8_1', 'LLM_8_2', 'LLM_8_3', 'mannitol_x',
                'map', 'map1', 'mls7dt_time_censored', 'mtdt_time_censored', 'na', 'na1',
                'nihss', 'obsTime', 'osm', 'pres', 'prev_mls', 'pulse', 'pulse1', 'rolling_aca_y', 
                'rolling_aspects', 'rolling_bun', 'rolling_cr', 'rolling_dbp', 'rolling_gluc', 
                'rolling_hts23_y', 'rolling_hts3_y', 'rolling_mannitol_y', 'rolling_map', 'rolling_na', 
                'rolling_osm', 'rolling_osmotics_y', 'rolling_pulse', 'rolling_sbp', 'rolling_size_mls', 
                'rolling_size_pgs', 'rolling_temp', 'rolling_wbc', 'sbp', 'sbp1', 'size_mls', 
                'size_pgs', 'stroke_territory', 'temp', 'temp1', 'wbc', 'wbc1'
            ]

        reduced_cols_8 = [
            'age_calc', 'aspects', 'dbp', 'dt', 'firstmls_time_censored', 'firstmlsdt_time_censored',
            'hts23_x', 'LLM_24_0', 'LLM_24_1', 'LLM_24_2', 'LLM_24_3', 'LLM_8_0', 'LLM_8_1',
            'LLM_8_2', 'LLM_8_3', 'LLM_36_0', 'LLM_36_1', 'LLM_36_2', 'LLM_36_3', 'map', 
            'mls3_time_censored', 'mls3dt_time_censored', 'mls5_time_censored', 'nihss',
            'pgs2_time_censored', 'pgs2dt_time_censored', 'pres', 'prev_mls', 'pulse', 'pulse1',
            'rolling_cr', 'rolling_dbp', 'rolling_hts23_y', 'rolling_hts3_y', 'rolling_map', 
            'rolling_osmotics_y', 'rolling_pulse', 'rolling_sbp', 'rolling_size_mls', 
            'rolling_stroke_territory', 'rolling_temp', 'sbp', 'size_mls', 'size_pgs', 'temp1'
        ]

        #cut down to most useful features
        def reduce_df(df):
            if lookahead_hours == 24:
                df = df[reduced_cols_24]
            elif lookahead_hours == 8:
                df = df[reduced_cols_8]

            return df
        
        if remove_excess_features:
            train_OLTV = reduce_df(train_OLTV)
            test_OLTV = reduce_df(test_OLTV)
            train_test_OLTV = reduce_df(train_test_OLTV)
            OLTV_columns_new = train_test_OLTV.columns

            bmc_OLTV = reduce_df(bmc_OLTV)
            bmc_OLTV_columns_new = bmc_OLTV.columns

        #fit classifier to training set using sample weights
        if device == "cuda":
            train_OLTV = cp.array(train_OLTV)
            train_targets = cp.array(train_targets)

        trained_model.fit(train_OLTV, train_targets, sample_weight = sample_weights)

        bmc_nd_target = pd.DataFrame(
            {"bmc_og_dt": bmc_og_dt, "bmc_ptid": bmc_ptid.astype(int)}
        )
        bmc_nd_target["target"] = 0
        nddf["nddt_min"] = nddf["nddt"] - 10

        for index, row in bmc_nd_target.iterrows():
            current_ptid = row["bmc_ptid"].astype(int)
            min_max_row = nddf[nddf.ptid == current_ptid]
            max_nddt = min_max_row.nddt.values[0]
            min_nddt = min_max_row.nddt_min.values[0]
            current_dt = row["bmc_og_dt"]
            if min_nddt <= current_dt <= max_nddt:
                bmc_nd_target.at[index, "target"] = 1

        if device == "cuda":
            bmc_OLTV == cp.array(bmc_OLTV)

        bmc_class_predictions = trained_model.predict_proba(bmc_OLTV)
        mlscol = "size_mls"
        bmc_mls = bmc_OLTV[mlscol] * scaler_dict[mlscol][1] + scaler_dict[mlscol][0]
        bmc_in_target_space = bmc_mls.apply(target_transform)

        nd_target_df = pd.DataFrame(
            bmc_class_predictions, columns=["0", "0-3", "3-8", "8+"]
        )
        nd_target_df["current_mls"] = bmc_mls

        nd_target_df["prob_>_8"] = nd_target_df["8+"]
        nd_target_df["prob_>_3"] = nd_target_df["3-8"] + nd_target_df["8+"]
        nd_target_df["prob_>_0"] = nd_target_df["0-3"] + nd_target_df["prob_>_3"]

        for threshold, col_name in [
            (0, "deterioration from 0 predicted"),
            (3, "deterioration from 3 predicted"),
            (8, "deterioration from 8 predicted"),
        ]:
            nd_target_df[col_name] = 0
            nd_target_df.loc[
                (nd_target_df["current_mls"] <= threshold)
                & (nd_target_df[f"prob_>_{threshold}"] > 0.5),
                col_name,
            ] = 1

        nd_target_df["any_deterioration_predicted"] = 0
        nd_target_df.loc[
            (nd_target_df["deterioration from 0 predicted"] == 1)
            | (nd_target_df["deterioration from 3 predicted"] == 1)
            | (nd_target_df["deterioration from 8 predicted"] == 1),
            "any_deterioration_predicted",
        ] = 1

        nd_target_df["nonzero_deterioration_predicted"] = 0
        nd_target_df.loc[
            (nd_target_df["deterioration from 3 predicted"] == 1)
            | (nd_target_df["deterioration from 8 predicted"] == 1),
            "nonzero_deterioration_predicted",
        ] = 1

        nd_target_df["target"] = bmc_nd_target["target"].values

        for prediction_type in ["any_deterioration", "nonzero_deterioration"]:
            pred_col = f"{prediction_type}_predicted"
            accuracy = accuracy_score(nd_target_df[pred_col], nd_target_df["target"])
            auroc = roc_auc_score(nd_target_df["target"], nd_target_df[pred_col])

            wandb.log(
                {
                    f"{prediction_type}_Accuracy": accuracy,
                    f"{prediction_type}_AUROC": auroc,
                }
            )
            print(f"{prediction_type} accuracy: {accuracy}")
            print(f"{prediction_type} AUROC: {auroc}")

        #get performance metrics for mgh test set

        if device == "cuda":
            test_OLTV = cp.array(test_OLTV)

        (
            accuracy,
            per_class_acc,
            per_class_auroc,
            per_class_auprc,
            mgh_plot_data,
            filtered_accuracy,
            filtered_per_class_auroc,
            filtered_per_class_auprc,
        ) = get_results_per_cv(
            trained_model,
            test_OLTV,
            test_targets,
            test_ptid,
            original_test_oltv,
            make_shap_plots,
            OTV,
            cv_num,
            time_NaN_fill,
            scaler_dict,
        )

        categories = ["0", "0_3", "3_8", "8_up"]
        for cat, acc, auroc, auprc, filtered_auroc, filtered_auprc in zip(
            categories, per_class_acc, per_class_auroc, per_class_auprc, filtered_per_class_auroc, filtered_per_class_auprc
        ):
            cv_per_class_accuracy[cat].append(float(acc))
            cv_per_class_auprc[cat].append(float(auprc))
            cv_per_class_auroc[cat].append(float(auroc))
            filtered_cv_per_class_auroc[cat].append(float(filtered_auroc))
            filtered_cv_per_class_auprc[cat].append(float(filtered_auprc))

        cv_overall_accuracy.append(accuracy)
        cv_overall_auroc.append(float(per_class_auroc.mean()))
        cv_overall_auprc.append(float(per_class_auprc.mean()))
        filtered_cv_overall_accuracy.append(filtered_accuracy)
        filtered_cv_overall_auroc.append(float(filtered_per_class_auroc.mean()))
        filtered_cv_overall_auprc.append(float(filtered_per_class_auprc.mean()))

        # get performance metrics for bmc dataset
        (
            accuracy,
            per_class_acc,
            per_class_auroc,
            per_class_auprc,
            bmc_plot_data,
            filtered_accuracy,
            filtered_per_class_auroc,
            filtered_per_class_auprc,
        ) = get_results_per_cv(
            trained_model,
            bmc_OLTV,
            bmc_targets,
            bmc_ptid,
            original_bmc_oltv,
            make_shap_plots,
            bmc_OTV,
            "bmc" + str(cv_num),
            time_NaN_fill,
            scaler_dict,
        )

        # wandb.log({"bmc accuracy": accuracy})
        # wandb.log({"bmc AUROC": per_class_auroc.mean()})
        # wandb.log({"bmc AUPRC": per_class_auprc.mean()})

        categories = ["0", "0_3", "3_8", "8_up"]
        for cat, acc, auroc, auprc, filtered_auroc, filtered_auprc in zip(
            categories, per_class_acc, per_class_auroc, per_class_auprc, filtered_per_class_auroc, filtered_per_class_auprc
        ):
            cv_bmc_per_class_accuracy[cat].append(float(acc))
            cv_bmc_per_class_auprc[cat].append(float(auprc))
            cv_bmc_per_class_auroc[cat].append(float(auroc))
            filtered_cv_bmc_per_class_auroc[cat].append(float(filtered_auroc))
            filtered_cv_bmc_per_class_auprc[cat].append(float(filtered_auprc))

        cv_bmc_overall_accuracy.append(accuracy)
        cv_bmc_overall_auroc.append(float(per_class_auroc.mean()))
        cv_bmc_overall_auprc.append(float(per_class_auprc.mean()))
        filtered_cv_bmc_overall_accuracy.append(filtered_accuracy)
        filtered_cv_bmc_overall_auroc.append(float(filtered_per_class_auroc.mean()))
        filtered_cv_bmc_overall_auprc.append(float(filtered_per_class_auprc.mean()))
        

        for mgh_plt, bmc_plt in zip(mgh_plot_data, bmc_plot_data):
            assert mgh_plt["type"] == bmc_plt["type"]
            plt.figure()

            if mgh_plt["type"] == "precision_recall":
                plt.plot(mgh_plt["x"], mgh_plt["y"], label="mgh")
                plt.plot(bmc_plt["x"], bmc_plt["y"], label="bmc")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title(f"Precision-Recall Curve for Class {mgh_plt['class']}")
                plt.legend()
                wandb.log(
                    {
                        f"combined_precision_recall_class_{mgh_plt['class']}_cv{cv_num}": wandb.Image(
                            plt
                        )
                    }
                )
            elif mgh_plt["type"] == "roc":
                plt.plot(
                    mgh_plt["x"], mgh_plt["y"], label=f"mgh (area={mgh_plt['auc']:.2f})"
                )
                plt.plot(
                    bmc_plt["x"], bmc_plt["y"], label=f"bmc (area={bmc_plt['auc']:.2f}"
                )
                plt.plot([0, 1], [0, 1], "k--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(
                    f"Receiver Operating Characteristic for Class {mgh_plt['class']}"
                )
                plt.legend(loc="lower right")
                wandb.log(
                    {
                        f"combined_roc_class_{mgh_plt['class']}_cv{cv_num}": wandb.Image(
                            plt
                        )
                    }
                )
            plt.close()

        cv_num += 1

    metrics = ["accuracy", "AUROC", "AUPRC"]
    prefixes = ["", "bmc_"]
    preprefixes = ["", "filtered_"]

    for cat in categories:
        for preprefix in preprefixes:
            for prefix in prefixes:
                for metric in metrics:
                    mean_key = f"{preprefix} CV {metric} for {prefix}target {cat}"
                    sd_key = f"{preprefix} CV sd {metric} for {prefix}target {cat}"

                    mean_value = np.mean(
                        locals()[f"{preprefix}cv_{prefix}per_class_{metric.lower()}"][cat]
                    )
                    sd_value = np.std(
                        locals()[f"{preprefix}cv_{prefix}per_class_{metric.lower()}"][cat]
                    )

                    wandb.log({mean_key: mean_value})
                    wandb.log({sd_key: sd_value})

    def log_metrics(prefix, accuracy, auroc, auprc):
        wandb.log({f"{prefix} mean accuracy": np.mean(accuracy)})
        wandb.log({f"{prefix} sd accuracy": np.std(accuracy)})
        wandb.log({f"{prefix} mean AUROC": np.mean(auroc)})
        wandb.log({f"{prefix} sd AUROC": np.std(auroc)})
        wandb.log({f"{prefix} mean AUPRC": np.mean(auprc)})
        wandb.log({f"{prefix} sd AUPRC": np.std(auprc)})
        wandb.log({f"raw {prefix} accuracy": accuracy})
        wandb.log({f"raw {prefix} AUROC": auroc})
        wandb.log({f"raw {prefix} AUPRC": auprc})

    log_metrics("CV", cv_overall_accuracy, cv_overall_auroc, cv_overall_auprc)
    log_metrics(
        "bmc CV", cv_bmc_overall_accuracy, cv_bmc_overall_auroc, cv_bmc_overall_auprc
    )
    log_metrics(
        "filtered CV",
        filtered_cv_overall_accuracy,
        filtered_cv_overall_auroc,
        filtered_cv_overall_auprc,
    )
    log_metrics(
        "filtered bmc CV",
        filtered_cv_bmc_overall_accuracy,
        filtered_cv_bmc_overall_auroc,
        filtered_cv_bmc_overall_auprc,
    )

    print(f"CV mean accuracy: {np.mean(cv_overall_accuracy)}")
    print(f"CV mean AUROC: {np.mean(cv_overall_auroc)}")
    print(f"CV mean bmc accuracy: {np.mean(cv_bmc_overall_accuracy)}")
    print(f"CV mean bmc AUROC: {np.mean(cv_bmc_overall_auroc)}")
    print(f"CV mean filtered accuracy: {np.mean(filtered_cv_overall_accuracy)}")
    print(f"CV mean filtered AUROC: {np.mean(filtered_cv_overall_auroc)}")
    print(f"CV mean filtered bmc accuracy: {np.mean(filtered_cv_bmc_overall_accuracy)}")
    print(f"CV mean filtered bmc AUROC: {np.mean(filtered_cv_bmc_overall_auroc)}")

# for cat, acc, auroc, auprc in zip(categories, per_class_acc, per_class_auroc, per_class_auprc):


# %%
if __name__ == "__main__":
    fire.Fire(xgb_main)
    print("done")
