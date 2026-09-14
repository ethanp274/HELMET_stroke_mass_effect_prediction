from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import shap
import wandb
from torch import tensor as tnsr
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassAccuracy,
    MulticlassAveragePrecision,
)
from sklearn.metrics import precision_recall_curve as pr_curve
from sklearn.metrics import roc_curve, roc_auc_score


def swelling_time_fn(x):
    if x <= 24:
        return "hyperacute swelling"
    if x <= 48:
        return "acute swelling"
    if x <= 96:
        return "average swelling"
    return "subacute swelling"


def target_transform_partial(x, time_NaN_fill):
    if x == 0:
        return 0
    elif 0 < x <= 3:
        return 1
    elif 3 < x <= 8:
        return 2
    elif 8 < x:
        return 3
    else:
        return time_NaN_fill

        # bmc_mls = bmc_OLTV[mlscol] * scaler_dict[mlscol][1] + scaler_dict[mlscol][0]
        # bmc_in_target_space = bmc_mls.apply(target_transform)

def get_results_per_cv(
    trained_model,
    OLTV_df,
    target_df,
    OLTV_ptid,
    original_OLTV_df,
    make_shap_plots,
    OTV_df,
    cv_num,
    time_NaN_fill,
    scaler_dict,
    make_auc_plots,
):
    reverse_map = {time_NaN_fill: "no scan", 0: "0", 1: "0_3", 2: "3_8", 3: "8_up"}
    target_col_names = ["0", "0_3", "3_8", "8_up"]
    target_transform = lambda x: target_transform_partial(x, time_NaN_fill)

    # predict on the test data
    predictions = trained_model.predict(OLTV_df)
    proba = trained_model.predict_proba(OLTV_df)

    current_mls = OLTV_df["size_mls"] * scaler_dict["size_mls"][1] + scaler_dict["size_mls"][0]
    OLTV_df_to_filter = OLTV_df.copy()
    OLTV_df_to_filter['predicted'] = predictions
    
    current_mls_in_target_space = current_mls.apply(target_transform)
    OLTV_df_to_filter["size_mls_in_target_space"] = current_mls_in_target_space
    OLTV_df_to_filter["target"] = target_df
    OLTV_df_to_filter["proba"] = proba.tolist()
    
    # Filter out the rows where the target is NaN 
    OLTV_df_to_filter = OLTV_df_to_filter[OLTV_df_to_filter["size_mls_in_target_space"] != OLTV_df_to_filter["target"]]

    # the accuracy in the filtered data
    filtered_correct_predictions = OLTV_df_to_filter[OLTV_df_to_filter["predicted"] == OLTV_df_to_filter["target"]].shape[0]
    if OLTV_df_to_filter.shape[0] == 0:
        filtered_accuracy = 0
    else:
        filtered_accuracy = filtered_correct_predictions / OLTV_df_to_filter.shape[0]

    try:
        filtered_proba = np.array(OLTV_df_to_filter["proba"].tolist())
        AUROC = MulticlassAUROC(num_classes=4, average=None)
        filtered_per_class_auroc = AUROC(tnsr(filtered_proba).float(), tnsr(OLTV_df_to_filter["target"].to_numpy()))
        AUPRC = MulticlassAveragePrecision(num_classes=4, average=None)
        filtered_per_class_auprc = AUPRC(tnsr(filtered_proba).float(), tnsr(OLTV_df_to_filter["target"].to_numpy()))
    except:
        filtered_per_class_auroc = np.array([0, 0, 0, 0])
        filtered_per_class_auprc = np.array([0, 0, 0, 0])
                                    

    num_classes = len(trained_model.classes_)

    # reverse map predictions and targets
    str_predictions = np.array(list(map(lambda x: reverse_map[x], predictions)))
    str_test_targets = np.array(list(map(lambda x: reverse_map[x], target_df)))

    original_OLTV_subframe = original_OLTV_df[["ptid", "dt", "size_mls"]].copy()
    original_OLTV_subframe["ptid"] = original_OLTV_subframe["ptid"].astype(int)
    OTV_subframe = OTV_df[["ptid", "surg", "surgdt", "maxmlsdt"]].copy()
    PTID_frame = OTV_subframe.merge(original_OLTV_subframe, on="ptid", how="inner")

    PTID_frame["day of data"] = PTID_frame["dt"] // 24
    PTID_frame["mls_values"] = PTID_frame["size_mls"]
    PTID_frame["mls_categories"] = (
        PTID_frame["mls_values"]
        .apply(lambda x: target_transform(x))
        .apply(lambda x: reverse_map[x])
    )
    PTID_frame["swelling_time"] = PTID_frame["maxmlsdt"].apply(swelling_time_fn)
    PTID_frame = PTID_frame[PTID_frame.index.isin(OLTV_df.index)]

    if make_shap_plots:
        explainer = shap.TreeExplainer(trained_model)

        # Note: Using plt.figure() here to ensure a new figure is created for the SHAP plot
        def make_shap(explainer, data, image_path):
            plt.figure()
            shap_values = explainer.shap_values(data)
            shap_values_list = [shap_values[:,:,i] for i in range(shap_values.shape[2])]
            shap.summary_plot(
                shap_values=shap_values_list,
                features=data,
                feature_names=data.columns.tolist(),
                plot_type="bar",
                class_names=target_col_names,
                show=False,
            )
            plt.savefig(image_path)
            plt.close()
            wandb.log({image_path: wandb.Image(image_path)})

        
        make_shap(explainer, OLTV_df, f"shap_plot_cv_{cv_num}.png")

        for cat in PTID_frame.mls_categories.unique():
            category_subset = OLTV_df[PTID_frame.mls_categories == cat]
            make_shap(explainer, category_subset, f"shap_plot_cv{cv_num}_{cat}.png")

    wandb.log(
        {
            "predictions_and_features_v7": wandb.Table(
                columns=[
                    "targets",
                    "mls_values",
                    "mls_categories",
                    "correct",
                    "predictions",
                    "dhc (surg)",
                    "maxmlsdt",
                    "swelling_time",
                    "day of data",
                    "ptid",
                    "0_prediction_probability",
                    "0_3_prediction_probability",
                    "3_8_prediction_probability",
                    "8_up_prediction_probability",
                ],
                data=list(
                    zip(
                        list(str_test_targets),
                        list(PTID_frame.mls_values.astype(str)),
                        list(PTID_frame.mls_categories.astype(str)),
                        [str(x) for x in list(str_predictions == str_test_targets)],
                        list(str_predictions),
                        list(PTID_frame.surg.astype(str)),
                        list(PTID_frame.maxmlsdt.astype(str)),
                        list(PTID_frame.swelling_time.astype(str)),
                        list(PTID_frame["day of data"].astype(str)),
                        list(PTID_frame.ptid.astype(str)),
                        list(proba[:, 0]),
                        list(proba[:, 1]),
                        list(proba[:, 2]),
                        list(proba[:, 3]),
                    )
                ),
            )
        }
    )


    accuracy = (predictions == target_df).mean()

    try:
        ACC = MulticlassAccuracy(num_classes=num_classes, average=None)
        per_class_acc = ACC(tnsr(predictions), tnsr(target_df))
        AUROC = MulticlassAUROC(num_classes=num_classes, average=None)
        per_class_auroc = AUROC(tnsr(proba).float(), tnsr(target_df))
        AUPRC = MulticlassAveragePrecision(num_classes=num_classes, average=None)
        per_class_auprc = AUPRC(tnsr(proba).float(), tnsr(target_df))
    except:
        per_class_acc = np.array([0, 0, 0, 0])
        per_class_auroc = np.array([0, 0, 0, 0])
        per_class_auprc = np.array([0, 0, 0, 0])

    def get_plot_data(num_classes, target_col_names, target_df, proba):
        plot_data = []

        for i in range(num_classes):
            t_nom = target_col_names[i]
            one_hot_t = np.eye(num_classes)[target_df]

            # Get the Precision-Recall curve data
            precision, recall, _ = pr_curve(one_hot_t[:, i], proba[:, i])
            plot_data.append(
                {
                    "type": "precision_recall",
                    "class": t_nom,
                    "x": recall,
                    "y": precision,
                }
            )

            # Get the ROC curve data
            fpr, tpr, _ = roc_curve(one_hot_t[:, i], proba[:, i])
            roc_auc = roc_auc_score(one_hot_t[:, i], proba[:, i])
            plot_data.append(
                {"type": "roc", "class": t_nom, "x": fpr, "y": tpr, "auc": roc_auc}
            )

        return plot_data

    if make_auc_plots:
        plot_data = get_plot_data(num_classes, target_col_names, target_df, proba)

        for i in range(num_classes):
            t_nom = target_col_names[i]
            one_hot_t = np.eye(num_classes)[target_df]

            # Plot the Precision-Recall curve
            precision, recall, _ = pr_curve(one_hot_t[:, i], proba[:, i])
            plt.figure()
            plt.plot(recall, precision, label=f"Class {t_nom}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve for Class {t_nom}")
            plt.legend()
            wandb.log({f"precision_recall_class_{t_nom}": wandb.Image(plt)})
            plt.close()

            # Plot the ROC curve
            fpr, tpr, _ = roc_curve(one_hot_t[:, i], proba[:, i])
            roc_auc = roc_auc_score(one_hot_t[:, i], proba[:, i])
            plt.figure()
            plt.plot(fpr, tpr, label=f"Class {t_nom} (area={roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], "k--")  # Dashed diagonal
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"Receiver Operating Characteristic for Class {t_nom}")
            plt.legend(loc="lower right")
            wandb.log({f"roc_class_{t_nom}": wandb.Image(plt)})
            plt.close()

        return accuracy, per_class_acc, per_class_auroc, per_class_auprc, plot_data, filtered_accuracy, filtered_per_class_auroc, filtered_per_class_auprc
    else:
        return accuracy, per_class_acc, per_class_auroc, per_class_auprc, filtered_accuracy, filtered_per_class_auroc, filtered_per_class_auprc

def get_results_simple(
    trained_model,
    OLTV_df,
    target_df,
    scaler_dict,
    time_NaN_fill = -10
):
    target_transform = lambda x: target_transform_partial(x, time_NaN_fill)

    # predict on the test data
    predictions = trained_model.predict(OLTV_df)
    proba = trained_model.predict_proba(OLTV_df)

    current_mls = OLTV_df["size_mls"] * scaler_dict["size_mls"][1] + scaler_dict["size_mls"][0]
    OLTV_df_to_filter = OLTV_df.copy()
    OLTV_df_to_filter['predicted'] = predictions
    
    current_mls_in_target_space = current_mls.apply(target_transform)
    OLTV_df_to_filter["size_mls_in_target_space"] = current_mls_in_target_space
    OLTV_df_to_filter["target"] = target_df
    OLTV_df_to_filter["proba"] = proba.tolist()

    # Filter out the rows where the target is NaN 
    OLTV_df_to_filter = OLTV_df_to_filter[OLTV_df_to_filter["size_mls_in_target_space"] != OLTV_df_to_filter["target"]]

    # the accuracy in the filtered data
    filtered_correct_predictions = OLTV_df_to_filter[OLTV_df_to_filter["predicted"] == OLTV_df_to_filter["target"]].shape[0]
    filtered_accuracy = filtered_correct_predictions / OLTV_df_to_filter.shape[0]

    accuracy = (predictions == target_df).mean()

    return accuracy, filtered_accuracy