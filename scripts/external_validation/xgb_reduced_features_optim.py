import fire
import pydantic
from pydantic import confloat, validate_arguments
import os
import sys
import numpy as np
from typing import Literal, List
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from wandb.xgboost import WandbCallback
from sklearn.ensemble import RandomForestClassifier
import wandb
import pandas as pd
import pdb
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import tempfile

from skopt import gp_minimize, space, BayesSearchCV

from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier


from helmet_mass_effect_pred.evaluation.cv_results import (
    get_results_per_cv,
    target_transform_partial,
    get_results_simple
)
from helmet_mass_effect_pred.data.splitting import drop_missing_impute_and_split
from helmet_mass_effect_pred.data.external_cohort import create_MGH_and_BMC
from helmet_mass_effect_pred.evaluation.scoring import OTV_scorer
from helmet_mass_effect_pred.evaluation.visualisation import make_tsne_and_pca

from sklearn.model_selection import train_test_split

from torch import tensor as tnsr
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassAccuracy,
    MulticlassAveragePrecision,
)

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

def get_scaler_dict(df):
    from sklearn.preprocessing import StandardScaler

    df_columns = df.columns

    scaler = StandardScaler().fit(df)
    # save the scaler to a dictionary for each column name so we can use it later
    scaler_dict = {}
    for i, col in enumerate(df_columns):
        scaler_dict[col] = [scaler.mean_[i], scaler.scale_[i]]
    
    return scaler_dict

def create_original_oltv(df, targets, ptid, columns):
    original_oltv = pd.DataFrame(df.copy(), columns=columns)
    original_oltv["targets"] = targets
    original_oltv["ptid"] = list(ptid)
    return original_oltv

def de_normalise_data(data, scaler_dict):
    output = data.copy()
    for col in output.columns:
        output[col] = output[col] * scaler_dict[col][1] + scaler_dict[col][0]
    return output

def get_sample_weights(df, targets, no_transition_weight, target_transform, scaler_dict):
    sample_weights = np.ones(df.shape[0])
    current_mls_denorm = df["size_mls"] * scaler_dict["size_mls"][1] + scaler_dict["size_mls"][0]
    current_mls_denorm_in_target_space = current_mls_denorm.apply(target_transform)
    sample_weights[current_mls_denorm_in_target_space == targets] = no_transition_weight
    return sample_weights

from helmet_mass_effect_pred.evaluation.cv_results import (
    get_results_per_cv,
    target_transform_partial,
)
from helmet_mass_effect_pred.data.splitting import drop_missing_impute_and_split
from helmet_mass_effect_pred.data.external_cohort import create_MGH_and_BMC
from helmet_mass_effect_pred.evaluation.scoring import OTV_scorer
from helmet_mass_effect_pred.evaluation.visualisation import make_tsne_and_pca

time_NaN_fill = -10

def reduce_df(df):
    reduced_cols = ['size_mls', 'rolling_firstmls_time_censored', 
        'age_calc',  'firstmlsdt_time_censored',
        'prev_mls', 'firstmls_time_censored', 'rolling_pulse', 'rolling_firstmlsdt_time_censored',
        'pulse', 'rolling_size_mls', 'temp1', 'rolling_hts23_y', 'rolling_map', 'osm1', 
        'rolling_aca_y', 'lbs', 'rolling_gluc', 'hts23_x', 'osm', 'pulse1', 
        'rolling_sbp', 'rolling_dbp', 'cr', 'dt', 'rolling_bun', 
        'rolling_sex', 'tici', 'rolling_temp', 'aspects', 'na1', 'rolling_na', 
        'rolling_size_pgs', 'size_pgs', 'mannitol_y', 'map1', 'dbp', 'rolling_cr',
        'rolling_wbc', 'map', 'nihss', 'rolling_pres', 'dbp1', 'tpadt_time_censored', 'af', 'rolling_af', 'sbp1', 
        'rolling_aca_x', 'sbp', 'bun1', 'bun', 'wbc', 'wbc1', 'rolling_hts23_x', 'cr1', 'gluc', 
        'aca1', 'aca_y', 'rolling_hts3_y', 'rolling_mannitol_y', 'na', 'stroke_territory', 'rolling_htn'
    ]
    df = df[reduced_cols]

    return df, reduced_cols

@validate_arguments
def get_cleaned_data(
    project: str = "bmc_data_exps",
    OTV_version: Literal["1.1", "2.0", "2.1"] = "2.1",
    LTV_version: Literal["1.1", "2.0"] = "2.0",
    disable_one_hot_encoding: bool = True, 
    impute_method: Literal["simple", "iterative", "knn"] = "simple",
    time_NaN_fill: int = -10,
    use_only_rc_values = True,
    lookback_hours = 12,
    lookahead_hours = 8,
    col_na_drop_fraction: confloat(ge=0.0, le = 1.0) = 0.3,
    patient_na_drop_fraction: confloat(ge=0.0, le=1.0) = 0.2,
    data_standard_scaling: bool = True,
    use_lsw_as_base_time_instead_of_pres: bool = True,
    create_new_mls_and_pgs_vals: bool = True,
    create_pca_and_tsne: bool = False,
    make_shap_plots: bool = False,
    use_ffil_and_rolling: bool = True,
    remove_data_after_surgery_occurs: bool = True,
    add_min_values: int = 0,
    use_mode_as_feature: bool = True,
    use_prev_as_feature: bool = True,
    remove_excess_features: bool = True,
):

    target_transform = lambda x: target_transform_partial(x, time_NaN_fill)

    if add_min_values > 0:
        add_min_values = True
    else:
        add_min_values = False
    
    OLTV, targets, bmc_OLTV, bmc_targets, OTV, bmc_OTV = create_MGH_and_BMC(
        filter_out_less_than_24_hours=True,
        disable_one_hot_encoding=disable_one_hot_encoding,
        time_NaN_fill=time_NaN_fill,
        use_only_rc_values=use_only_rc_values,
        lookback_hours=lookback_hours,
        lookahead_hours=lookahead_hours,
        use_lsw_as_base_time_instead_of_pres=use_lsw_as_base_time_instead_of_pres,
        create_new_mls_and_pgs_vals=create_new_mls_and_pgs_vals,
        use_ffil_and_rolling=use_ffil_and_rolling,
        add_min_values=add_min_values,
        use_mode_as_feature=use_mode_as_feature,
        use_prev_as_feature=use_prev_as_feature,
    )

    if add_min_values:
        OLTV = OLTV[bmc_OLTV.columns]

    base_time = "LSW" if use_lsw_as_base_time_instead_of_pres else "first_arrival_dt"

    nddf = pd.read_csv("data/raw/OTV_BMC_4.csv")
    nddf[base_time] = pd.to_datetime(nddf[base_time])
    nddf["nddt"] = pd.to_datetime(nddf["nddt"])
    nddf["nddt"] = nddf["nddt"] - nddf[base_time]
    nddf["nddt"] = nddf["nddt"].dt.total_seconds() // 3600
    nddf["nddt"] = nddf["nddt"].fillna(int(time_NaN_fill))
    nddf = nddf[["CDW_MRN", "nddt"]]
    nddf.rename(columns={"CDW_MRN": "ptid"}, inplace=True)

    if use_ffil_and_rolling:
        # remove ffill_ from column names
        OLTV.columns = [x.replace("ffill_", "") for x in OLTV.columns]
        bmc_OLTV.columns = [x.replace("ffill_", "") for x in bmc_OLTV.columns]

    OLTV_columns = OLTV.columns
    bmc_OLTV_columns = bmc_OLTV.columns

    targets = targets.apply(target_transform)
    bmc_targets = bmc_targets.apply(target_transform)

    test_frac = 0.2 

    def split_and_impute(OLTV, targets, test_frac):
        OLTV = pd.concat([OLTV, targets], axis=1)
        num_rows, num_cols = OLTV.shape

        OLTV = OLTV.reset_index(drop=True)
        unique = OLTV.ptid.unique()
        ptid_train, ptid_test = train_test_split(unique, test_size=test_frac, shuffle=False)
        train_test_list = [[ptid_train, ptid_test]]

        train_OLTV = OLTV[OLTV.ptid.isin(ptid_train)].copy()
        test_OLTV = OLTV[OLTV.ptid.isin(ptid_test)].copy()

        train_targets = train_OLTV['target'].copy()
        test_targets = test_OLTV['target'].copy()

        train_OLTV = train_OLTV.drop(columns=['target'], axis=1)
        test_OLTV = test_OLTV.drop(columns=['target'], axis=1)

        train_ptid = train_OLTV["ptid"].copy()
        test_ptid = test_OLTV["ptid"].copy()

        train_OLTV = train_OLTV.drop(columns=["ptid"], axis=1)
        test_OLTV = test_OLTV.drop(columns=["ptid"], axis=1)

        cols = train_OLTV.columns

        imputer = SimpleImputer().fit(train_OLTV)

        train_OLTV = imputer.transform(train_OLTV)
        test_OLTV = imputer.transform(test_OLTV)

        train_OLTV = pd.DataFrame(train_OLTV, columns = cols)
        test_OLTV = pd.DataFrame(test_OLTV, columns = cols)

        return train_ptid, test_ptid, train_OLTV, test_OLTV, train_targets, test_targets

    train_ptid, test_ptid, train_OLTV, test_OLTV, train_targets, test_targets = split_and_impute(OLTV=OLTV, targets=targets, test_frac = test_frac)
    bmc_train_ptid, bmc_test_ptid, bmc_train_OLTV, bmc_test_OLTV, bmc_train_targets, bmc_test_targets = split_and_impute(OLTV=bmc_OLTV, targets = bmc_targets, test_frac=test_frac)
    
    bmc_ptid = pd.concat([bmc_train_ptid, bmc_test_ptid])
    bmc_OLTV = pd.concat([bmc_train_OLTV, bmc_test_OLTV])
    bmc_targets = pd.concat([bmc_train_targets, bmc_test_targets])

    #transform mode baseline feature to target space
    if use_mode_as_feature:
        train_OLTV['target_mode'] = np.array(list(map(target_transform, train_OLTV['target_mode'])))
        test_OLTV['target_mode'] = np.array(list(map(target_transform, test_OLTV['target_mode'])))
        bmc_OLTV['target_mode'] = np.array(list(map(target_transform, bmc_OLTV['target_mode'])))

    #transform prev mls baseline feature to target space
    if use_prev_as_feature:
        train_OLTV['prev_mls'] = np.array(list(map(target_transform, train_OLTV['prev_mls'])))
        test_OLTV['prev_mls'] = np.array(list(map(target_transform, test_OLTV['prev_mls'])))
        bmc_OLTV['prev_mls'] = np.array(list(map(target_transform, bmc_OLTV['prev_mls'])))
    
    train_OLTV = pd.DataFrame(train_OLTV, columns = OLTV_columns)
    test_OLTV = pd.DataFrame(test_OLTV, columns = OLTV_columns)

    #drop columns not found in the other data set
    bmc_extra_cols = list(set(bmc_OLTV.columns) - set(train_OLTV.columns))
    bmc_OLTV = bmc_OLTV.drop(bmc_extra_cols, axis = 1)

    mgh_extra_cols = list(set(train_OLTV.columns) - set(bmc_OLTV.columns))
    train_OLTV = train_OLTV.drop(mgh_extra_cols, axis = 1)
    test_OLTV = test_OLTV.drop(mgh_extra_cols, axis = 1)
  
    OLTV_columns_new = list(train_OLTV.columns)
    bmc_OLTV_columns_new = list(bmc_OLTV.columns)

    if remove_excess_features:
        train_OLTV, train_cols = reduce_df(train_OLTV)
        test_OLTV, test_cols = reduce_df(test_OLTV)
        bmc_OLTV, bmc_cols = reduce_df(bmc_OLTV)
    
        OLTV_columns_new = train_cols
        bmc_OLTV_columns_new = bmc_cols

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
    else:
        scaler_dict = None
    
    #convert from np.array to pd.DataFrame
    train_OLTV = pd.DataFrame(train_OLTV, columns=OLTV_columns_new)
    test_OLTV = pd.DataFrame(test_OLTV, columns=OLTV_columns_new)
    bmc_OLTV = pd.DataFrame(bmc_OLTV, columns=bmc_OLTV_columns_new)

    return train_OLTV, train_targets, test_OLTV, test_targets, bmc_OLTV, bmc_targets, scaler_dict

def xgb_main(
    project: str = "bmc_data_exps",
    OTV_version: Literal["1.1", "2.0", "2.1"] = "2.1",
    LTV_version: Literal["1.1", "2.0"] = "2.0",
    disable_one_hot_encoding: bool = True,
    impute_method: Literal["simple", "iterative", "knn"] = "simple",
    time_NaN_fill: int = -10,
    use_only_rc_values=True,
    lookback_hours=24,
    lookahead_hours=24,
    col_na_drop_fraction: confloat(ge=0.0, le=1.0) = 0.3,  # drop cols w/ % vals missing
    patient_na_drop_fraction: confloat(ge=0.0, le=1.0) = 0.2,  # drop rows w/ %  missing
    data_standard_scaling: bool = True,  # scale data to mean 0, std 1    learning_rate: confloat(ge=0.001, le=0.5) = 0.3,
    max_depth: int = 6,
    min_child_weight: int = 1,
    learning_rate: confloat(ge=0.001, le=0.5) = 0.3,
    gamma: confloat(ge=0.0, le=5.0) = 0.0,
    subsample: confloat(ge=0.5, le=1.0) = 1.0,
    reg_lambda: confloat(ge=1.0, le=10.0) = 1.0,
    reg_alpha: confloat(ge=0.0, le=5.0) = 0.0,
    #no_transition_weighting: pydantic.Annotated[float, pydantic.Field(ge = 0, le = 1.0)] = 0.5, # map to power distribution, fix other hyperparams to best values from prior runs
    no_transition_weighting = 0.5,
    use_lsw_as_base_time_instead_of_pres: bool = True,
    create_new_mls_and_pgs_vals: bool = True,
    create_pca_and_tsne: bool = False,
    make_shap_plots: bool = True,
    use_ffil_and_rolling: bool = True,
    remove_data_after_surgery_occurs: bool = True,
    add_min_values: int = 0,
):
    wandb.init(project=project, config=locals(), save_code=True)
    target_transform = lambda x: target_transform_partial(x, time_NaN_fill)
    # api = wandb.Api()
    # pdb.set_trace()

    if add_min_values > 0:
        add_min_values = True
    else:
        add_min_values = False

    OLTV, targets, bmc_OLTV, bmc_targets, OTV, bmc_OTV = create_MGH_and_BMC(
        filter_out_less_than_24_hours=True,
        disable_one_hot_encoding=disable_one_hot_encoding,
        time_NaN_fill=time_NaN_fill,
        use_only_rc_values=use_only_rc_values,
        lookback_hours=lookback_hours,
        lookahead_hours=lookahead_hours,
        use_lsw_as_base_time_instead_of_pres=use_lsw_as_base_time_instead_of_pres,
        create_new_mls_and_pgs_vals=create_new_mls_and_pgs_vals,
        use_ffil_and_rolling=use_ffil_and_rolling,
        add_min_values=add_min_values,
    )

    if add_min_values:
        # drop the columns in OLTV that are not in bmc_OLTV
        OLTV = OLTV[bmc_OLTV.columns]

    base_time = "LSW" if use_lsw_as_base_time_instead_of_pres else "first_arrival_dt"

    nddf = pd.read_csv("data/raw/OTV_BMC_4.csv")
    nddf[base_time] = pd.to_datetime(nddf[base_time])
    nddf["nddt"] = pd.to_datetime(nddf["nddt"])
    nddf["nddt"] = nddf["nddt"] - nddf[base_time]
    nddf["nddt"] = nddf["nddt"].dt.total_seconds() // 3600
    nddf["nddt"] = nddf["nddt"].fillna(int(time_NaN_fill))
    nddf = nddf[["CDW_MRN", "nddt"]]
    nddf.rename(columns={"CDW_MRN": "ptid"}, inplace=True)

    if use_ffil_and_rolling:
        # remove ffill_ from column names
        OLTV.columns = [x.replace("ffill_", "") for x in OLTV.columns]
        bmc_OLTV.columns = [x.replace("ffill_", "") for x in bmc_OLTV.columns]

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

    cv_overall_accuracy = []
    cv_overall_auroc = []
    cv_overall_auprc = []

    cv_bmc_per_class_accuracy = {"0": [], "0_3": [], "3_8": [], "8_up": []}
    cv_bmc_per_class_auroc = {"0": [], "0_3": [], "3_8": [], "8_up": []}
    cv_bmc_per_class_auprc = {"0": [], "0_3": [], "3_8": [], "8_up": []}

    cv_bmc_overall_accuracy = []
    cv_bmc_overall_auroc = []
    cv_bmc_overall_auprc = []

    filtered_cv_per_class_accuracy = {"0": [], "0_3": [], "3_8": [], "8_up": []}
    filtered_cv_per_class_auroc = {"0": [], "0_3": [], "3_8": [], "8_up": []}
    filtered_cv_per_class_auprc = {"0": [], "0_3": [], "3_8": [], "8_up": []}

    filtered_cv_overall_accuracy = []
    filtered_cv_overall_auroc = []
    filtered_cv_overall_auprc = []

    filtered_cv_bmc_per_class_accuracy = {"0": [], "0_3": [], "3_8": [], "8_up": []}
    filtered_cv_bmc_per_class_auroc = {"0": [], "0_3": [], "3_8": [], "8_up": []}
    filtered_cv_bmc_per_class_auprc = {"0": [], "0_3": [], "3_8": [], "8_up": []}

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

        original_bmc_oltv = create_original_oltv(
            bmc_OLTV, bmc_targets, bmc_ptid, OLTV_columns
        )
        original_train_oltv = create_original_oltv(
            train_OLTV, train_targets, train_ptid, OLTV_columns
        )
        original_test_oltv = create_original_oltv(
            test_OLTV, test_targets, test_ptid, OLTV_columns
        )

        original_mgh_oltv = pd.concat([original_train_oltv, original_test_oltv])

        train_test_OLTV = np.concatenate([train_OLTV, test_OLTV])
        train_test_OLTV = pd.DataFrame(train_test_OLTV, columns=OLTV_columns)
        train_test_ptid = np.concatenate([train_ptid, test_ptid])

        train_targets = np.array(list(map(target_transform, train_targets)))
        test_targets = np.array(list(map(target_transform, test_targets)))
        train_test_targets = np.concatenate([train_targets, test_targets])

        bmc_targets = np.array(list(map(target_transform, bmc_targets)))

        def create_mode_baseline(df, train_targets, ptids, type="train"):
            size_mls = df["size_mls"]
            size_mls = np.array(list(map(target_transform, size_mls)))
            size_mls_df = pd.DataFrame(size_mls, columns=["size_mls"])
            size_mls_df["targets"] = train_targets
            size_mls_df["ptid"] = ptids
            size_mls_df = size_mls_df[size_mls_df["size_mls"] != time_NaN_fill]
            # calculate the baseline accuracy

            total_accurate_mode = 0
            for i in range(4):
                Y_section = size_mls_df[size_mls_df["size_mls"] == i]
                Y_mode = size_mls_df["targets"].mode().values.item()
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


        create_mode_baseline(train_test_OLTV, train_test_targets, train_test_ptid, type="mgh")
        create_mode_baseline(pd.DataFrame(bmc_OLTV, columns=bmc_OLTV_columns), bmc_targets, bmc_ptid, type="bmc")

        if data_standard_scaling:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler().fit(train_OLTV)
            # save the scaler to a dictionary for each column name so we can use it later
            scaler_dict = {}
            for i, col in enumerate(OLTV_columns):
                scaler_dict[col] = [scaler.mean_[i], scaler.scale_[i]]
            train_OLTV = scaler.transform(train_OLTV)
            test_OLTV = scaler.transform(test_OLTV)
            bmc_OLTV = scaler.transform(bmc_OLTV)

        counts = np.concatenate([train_targets, test_targets])
        counts = np.unique(counts, return_counts=True)

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

        train_OLTV = pd.DataFrame(train_OLTV, columns=OLTV_columns)
        test_OLTV = pd.DataFrame(test_OLTV, columns=OLTV_columns)
        bmc_OLTV = pd.DataFrame(bmc_OLTV, columns=OLTV_columns)

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

            AUROC = MulticlassAUROC(num_classes=4, average="none")
            baseline_auroc = AUROC(tnsr(proba).float(), tnsr(y_test.values))

            AUPRC = MulticlassAveragePrecision(num_classes=4, average="none")
            baseline_auprc = AUPRC(tnsr(proba).float(), tnsr(y_test.values))

            mean_auroc = np.mean(baseline_auroc.numpy())
            mean_auprc = np.mean(baseline_auprc.numpy())

            wandb.log({f"{type} edema score baseline accuracy": baseline_accuracy})
            wandb.log({f"{type} edema score baseline mean AUROC": mean_auroc})
            wandb.log({f"{type} edema score baseline mean AUPRC": mean_auprc})

            wandb.log(
                {
                    f"{type} edema score baseline raw AUROC cv {cv_num}": baseline_auroc.numpy().tolist(),
                    f"{type} edema score baseline raw AUPRC cv {cv_num}": baseline_auprc.numpy().tolist(),
                }
            )

            print(f"{type} edema score baseline accuracy: ", baseline_accuracy)
            print(f"{type} edema score baseline AUROC: ", mean_auroc)

        create_edema_score_baseline(bmc_OLTV, bmc_OTV, bmc_targets, bmc_ptid, "bmc", cv_num)

        all_mgh_OLTV = pd.concat([train_OLTV, test_OLTV])
        all_mgh_targets = np.concatenate([train_targets, test_targets])
        all_mgh_ptid = np.concatenate([train_ptid, test_ptid])

        create_edema_score_baseline(all_mgh_OLTV, OTV, all_mgh_targets, all_mgh_ptid, "mgh", cv_num)

        def de_normalise_data(data, scaler_dict):
            output = data.copy()
            for col in output.columns:
                output[col] = output[col] * scaler_dict[col][1] + scaler_dict[col][0]
            return output

        de_normalised_mgh_OLTV = de_normalise_data(all_mgh_OLTV, scaler_dict)
        de_normalised_bmc_OLTV = de_normalise_data(bmc_OLTV, scaler_dict)
        
        temp_size_mls_mgh = de_normalised_mgh_OLTV["size_mls"].apply(target_transform)
        temp_size_mls_bmc = de_normalised_bmc_OLTV["size_mls"].apply(target_transform)

        filtered_mgh_OLTV = all_mgh_OLTV[temp_size_mls_mgh != all_mgh_targets]
        filtered_mgh_targets = all_mgh_targets[temp_size_mls_mgh != all_mgh_targets]
        filtered_mgh_ptid = all_mgh_ptid[temp_size_mls_mgh != all_mgh_targets]

        filtered_bmc_OLTV = bmc_OLTV[temp_size_mls_bmc != bmc_targets]
        filtered_bmc_targets = bmc_targets[temp_size_mls_bmc != bmc_targets]
        filtered_bmc_ptid = bmc_ptid[temp_size_mls_bmc != bmc_targets]

        create_mode_baseline(filtered_bmc_OLTV, filtered_bmc_targets, filtered_bmc_ptid, type = "filtered bmc")
        create_mode_baseline(filtered_mgh_OLTV, filtered_mgh_targets, filtered_mgh_ptid, type = "filtered mgh")

        create_edema_score_baseline(filtered_bmc_OLTV, bmc_OTV, filtered_bmc_targets, filtered_bmc_ptid, "filtered bmc", cv_num)
        create_edema_score_baseline(filtered_mgh_OLTV, OTV, filtered_mgh_targets, filtered_mgh_ptid, "filtered mgh", cv_num)

        # TODO! make version of variables that are not normalised!

        # apply dt scaler_dict transform to nddf
        nddf["nddt_transformed"] = (nddf["nddt"] - scaler_dict["dt"][1]) / scaler_dict[
            "dt"
        ][0]

        bmc_og_dt = de_normalised_bmc_OLTV.dt

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
            
        )

        sample_weights = get_sample_weights(train_OLTV, train_targets, no_transition_weighting, target_transform, scaler_dict)

        trained_model.fit(train_OLTV, train_targets, sample_weight = sample_weights)

        new_bmc_targets = "ass"

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

from skopt.utils import use_named_args

#@use_named_args(search_space)
def objective(**args):
    train_df = args['train_df']
    train_targets = args['train_targets']
    test_df = args['test_df']
    test_targets = args['test_targets']
    bmc_df = args['bmc_df']
    bmc_targets = args['bmc_targets']
    scaler_dict = args['scaler_dict']
    learning_rate = args['learning_rate']
    max_depth = args['max_depth']
    min_child_weight = args['max_child_weight']
    gamma = args['gamma']
    subsample = args['subsample']
    reg_lambda = args['reg_lambda']
    reg_alpha = args['reg_alpha']
    no_transition_weight = args['no_transition_weight']

    model = XGBClassifier(
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
        reg_alpha=reg_alpha
    )

    time_NaN_fill = -10
    target_transform = lambda x: target_transform_partial(x, time_NaN_fill)

    sample_weights = get_sample_weights(train_df, train_targets, no_transition_weight, target_transform, scaler_dict)

    model.fit(train_df, train_targets, sample_weight = sample_weights)

    accuracy, filtered_accuracy = get_results_simple(trained_model=model, OLTV_df = test_df, target_df = test_targets, scaler_dict = scaler_dict)
    bmc_accuracy, filtered_bmc_accuracy = get_results_simple(trained_model = model, OLTV_df = bmc_df, target_df = bmc_targets, sclaer_dict = scaler_dict)

    print(f'accuracy: {accuracy: .3f}')
    print(f'filtered_accuracy: {filtered_accuracy: .3f}')
    print(f'bmc accuracy: {bmc_accuracy: .3f}')
    print(f'bmc filtered accuracy: {filtered_bmc_accuracy: .3f}')
    
    return -accuracy

def scorer(model, X, y):
    scaler_dict = get_scaler_dict(X)
    accuracy, filtered_accuracy = get_results_simple(trained_model = model, OLTV_df = X, target_df = y, scaler_dict = scaler_dict)
    return accuracy

from skopt import space

# %%
if __name__ == "__main__":
    train_df, train_targets, test_df, test_targets, bmc_df, bmc_targets, scaler_dict = get_cleaned_data()
    
    no_transition_weight = 0.5

    sample_weights = get_sample_weights(train_df, train_targets, no_transition_weight = no_transition_weight, target_transform = lambda x: target_transform_partial(x, time_NaN_fill = -10), scaler_dict = scaler_dict)

    search_space = {
            'learning_rate': space.Real(0.001, 0.3, "uniform", name = "learning_rate"),
            'max_depth': space.Integer(2, 10, name = "max_depth"),
            'min_child_weight': space.Integer(1, 10, name = "min_child_weight"),
            'gamma': space.Real(0.0, 5.0, "uniform", name = "gamma"),
            'subsample': space.Real(0.3, 1.0, "uniform", name = "subsample"), 
            'reg_lambda': space.Real(1.0, 10.0, "uniform", name = "reg_lambda"),
            'reg_alpha': space.Real(0.0, 5.0, "uniform", name = "reg_alpha"),
    }

    all_mgh_df = pd.concat([train_df, test_df])
    all_mgh_targets = pd.concat([train_targets, test_targets])

    opt = BayesSearchCV(
        estimator = XGBClassifier(), 
        search_spaces = search_space, 
        scoring = scorer, 
        fit_params={
            'sample_weight': sample_weights,
        }, 
        n_iter = 100, 
        n_jobs = -1, 
        cv = 3, 
        verbose = 1
    )

    opt.fit(all_mgh_df, all_mgh_targets)

    print(f'Best MGH accuracy: {opt.best_score_: .3f}')
    print(f'BMC accuracy: {opt.score(bmc_df, bmc_targets): .3f}')
    print(f'Best params: {str(opt.best_params_)}')

    import json

    with open("configs/hyperparameters/reduce_feature_opt_hyperparams_24hr.json", "w") as outfile:
        json.dump(opt.best_params_, outfile)

    print("done")