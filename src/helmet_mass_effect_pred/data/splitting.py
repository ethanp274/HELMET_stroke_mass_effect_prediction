from typing import Literal
from pydantic import confloat, validate_arguments
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import sys



@validate_arguments
def drop_missing_impute_and_split(
    OLTV,
    targets,
    impute_method: Literal["simple", "iterative", "knn"] = "simple",
    col_na_drop_fraction: confloat(ge=0.0, le=1.0) = 0.3,  # drop cols w/ % vals missing
    patient_na_drop_fraction: confloat(ge=0.0, le=1.0) = 0.2,  # drop rows w/ %  missing
    test_split: confloat(ge=0.1, le=0.5) = 0.2,  # fraction of data to use for testing
    return_cross_validation: bool = False,
    create_deep_learning_dataset: bool = False,
    create_panos_data: bool = False,
):
    if isinstance(targets, pd.Series):
        targets = pd.DataFrame(targets)
    target_col_names = targets.columns.copy()

    OLTV = pd.concat([OLTV, targets], axis=1)
    num_rows, num_cols = OLTV.shape

    if create_deep_learning_dataset or create_panos_data:
        print("cannot drop rows when creating deep learning dataset")
    else:
        OLTV = OLTV.dropna(axis=0, thresh=OLTV.shape[1] * patient_na_drop_fraction)
        print(f"Number of rows dropped: {num_rows - OLTV.shape[0]}")

    OLTV = OLTV.dropna(axis=1, thresh=OLTV.shape[0] * col_na_drop_fraction)
    print(f"Number of columns dropped: {num_cols - OLTV.shape[1]}")

    # reset index
    OLTV = OLTV.reset_index(drop=True)
    unique = OLTV.ptid.unique()
    if return_cross_validation:
        KFold_splits = 5
        kf = KFold(n_splits=KFold_splits, shuffle=True)
        train_test_list = []
        for train_indices, test_indices in kf.split(unique):
            train_test_list.append([unique[train_indices], unique[test_indices]])
    else:
        ptid_train, ptid_test = train_test_split(
            unique, test_size=test_split, shuffle=False
        )
        train_test_list = [[ptid_train, ptid_test]]

    cv_train_ptids = []
    cv_test_ptids = []
    cv_train_OLTV = []
    cv_test_OLTV = []
    cv_train_targets = []
    cv_test_targets = []

    if create_deep_learning_dataset:
        cv_train_instances = []
        cv_test_instances = []

    for ptid_train, ptid_test in train_test_list:
        train_OLTV = OLTV[OLTV.ptid.isin(ptid_train)].copy()
        test_OLTV = OLTV[OLTV.ptid.isin(ptid_test)].copy()

        train_targets = train_OLTV[target_col_names].copy()
        test_targets = test_OLTV[target_col_names].copy()

        train_OLTV = train_OLTV.drop(columns=target_col_names, axis=1)
        test_OLTV = test_OLTV.drop(columns=target_col_names, axis=1)

        train_ptid = train_OLTV["ptid"].copy()
        test_ptid = test_OLTV["ptid"].copy()
        train_OLTV = train_OLTV.drop(columns=["ptid"], axis=1)
        test_OLTV = test_OLTV.drop(columns=["ptid"], axis=1)
        if create_deep_learning_dataset:
            train_indices = train_OLTV["instance"].copy()
            test_indices = test_OLTV["instance"].copy()
            train_OLTV = train_OLTV.drop(columns=["instance"], axis=1)
            test_OLTV = test_OLTV.drop(columns=["instance"], axis=1)

        OLTV_columns = list(train_OLTV.columns.copy())

        if impute_method == "iterative":
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer

            imputer = IterativeImputer().fit(train_OLTV)
        elif impute_method == "simple":
            from sklearn.impute import SimpleImputer

            imputer = SimpleImputer().fit(train_OLTV)
        elif impute_method == "knn":
            from sklearn.impute import KNNImputer

            imputer = KNNImputer().fit(train_OLTV)
        else:
            print("warning - no imputation method specified. May cause errors.")
            OLTV = OLTV.to_numpy()

        train_OLTV = imputer.transform(train_OLTV)
        test_OLTV = imputer.transform(test_OLTV)

        train_targets = train_targets.to_numpy()
        test_targets = test_targets.to_numpy()
        if train_targets.shape[1] < 1:
            train_targets = train_targets.argmax(axis=1)
            train_targets = test_targets.argmax(axis=1)
        else:
            train_targets = np.squeeze(train_targets)
            test_targets = np.squeeze(test_targets)

        cv_train_ptids.append(train_ptid)
        cv_test_ptids.append(test_ptid)
        cv_train_OLTV.append(train_OLTV)
        cv_test_OLTV.append(test_OLTV)
        cv_train_targets.append(train_targets)
        cv_test_targets.append(test_targets)
        if create_deep_learning_dataset:
            cv_train_instances.append(train_indices)
            cv_test_instances.append(test_indices)

    if create_deep_learning_dataset:
        return (
            cv_train_ptids,
            cv_test_ptids,
            cv_train_OLTV,
            cv_test_OLTV,
            cv_train_targets,
            cv_test_targets,
            target_col_names,
            OLTV_columns,
            cv_train_instances,
            cv_test_instances,
        )
    else:
        return (
            cv_train_ptids,
            cv_test_ptids,
            cv_train_OLTV,
            cv_test_OLTV,
            cv_train_targets,
            cv_test_targets,
            target_col_names,
            OLTV_columns,
        )
