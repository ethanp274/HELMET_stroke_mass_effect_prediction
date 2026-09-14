import sys
import os
import pandas as pd
import numpy as np
import json
import pickle
import csv
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve as pr_curve
from sklearn.metrics import roc_curve, auc
from torch import tensor as tnsr
import matplotlib.pyplot as plt
import matplotlib as mpl

# Output directories for figures (git-ignored, so absent in a fresh clone).
for _output_dir in ('results/figures/shap',):
    os.makedirs(_output_dir, exist_ok=True)

mpl.rc('font', family='Times New Roman')

PRE_SAVED = True  #Set to True if you want to load pre-saved EDEMA splits and models

######
# For Reference:
# sensitivity = recall = TP / (TP + FN)
# specificity = TN / (TN + FP)
# precision = TP / (TP + FP)
######


def restructure_output(raw_output):
    probas_list = []
    target_list = []

    for i in range(5): 
        probas = raw_output[raw_output['cv'] == i][['proba_0', 'proba_1', 'proba_2', 'proba_3']]
        probas = [[p0, p1, p2, p3] for p0, p1, p2, p3 in zip(probas['proba_0'], probas['proba_1'], probas['proba_2'], probas['proba_3'])]

        targets = raw_output[raw_output['cv'] == i]['targets'].to_list()

        probas_list.append(probas)
        target_list.append(targets)
    
    return probas_list, target_list

def restructure_for_plotting(avg_metricx_by_thresh, avg_metricy_by_thresh):
    x_axis = []
    y_axis = []
    # lower_y = []
    # upper_y = []
    y_std = []

    if len(avg_metricx_by_thresh) != len(avg_metricy_by_thresh):
        print('Error: Different number of thresholds')
        return

    for thresh in avg_metricx_by_thresh.keys():
        x_axis.append(avg_metricx_by_thresh[thresh][0])
    
    for thresh in avg_metricy_by_thresh.keys():
        y_axis.append(avg_metricy_by_thresh[thresh][0])
        # lower_y.append(avg_metricy_by_thresh[thresh][1])
        # upper_y.append(avg_metricy_by_thresh[thresh][2])
        y_std.append(avg_metricy_by_thresh[thresh][1])

    # return [x_axis, y_axis, lower_y, upper_y]
    return [x_axis, y_axis, y_std]


def get_metrics_from_probas(probas_list, targets_list, threshold_list):
    
    sensitivities_by_thresh = {}
    specificities_by_thresh = {}
    roc_x_by_thresh = {}
    precisions_by_thresh = {}

    for index, thresh in enumerate(threshold_list):

        sensitivities_by_split = []
        specificities_by_split = []
        roc_x_by_split = []
        precision_by_split = []

        weights_by_split = []

        for probas, targets in zip(probas_list, targets_list):
            
            sensitivities_by_class = []
            specificities_by_class = []
            roc_x_by_class = []
            precision_by_class = []

            targets_by_class = np.eye(4)[targets]
            targets_0 = targets_by_class[:, 0]
            targets_1 = targets_by_class[:, 1]
            targets_2 = targets_by_class[:, 2]
            targets_3 = targets_by_class[:, 3]

            targets_0 = np.array(targets_0)
            targets_1 = np.array(targets_1)
            targets_2 = np.array(targets_2)
            targets_3 = np.array(targets_3)

            targets_by_class = [targets_0, targets_1, targets_2, targets_3]

            if index == 0:
                weights_by_class = [sum(targets_0) / len(targets), sum(targets_1) / len(targets), sum(targets_2) / len(targets), sum(targets_3) / len(targets)]
                weights_by_split.append(weights_by_class)

            pred_by_class = [[], [], [], []]

            for i in range(len(probas)):
                if probas[i][0] >= thresh:
                    pred_by_class[0].append(1)
                else:
                    pred_by_class[0].append(0)

                if probas[i][1] >= thresh:
                    pred_by_class[1].append(1)
                else:
                    pred_by_class[1].append(0)

                if probas[i][2] >= thresh:
                    pred_by_class[2].append(1)
                else:
                    pred_by_class[2].append(0)

                if probas[i][3] >= thresh:
                    pred_by_class[3].append(1)
                else:
                    pred_by_class[3].append(0)
                
            pred_by_class = [np.array(pred_by_class[0]), np.array(pred_by_class[1]), np.array(pred_by_class[2]), np.array(pred_by_class[3])]

            for j in range(4):
                tp = sum([1 for i in range(len(targets_by_class[j])) if pred_by_class[j][i] == 1 and targets_by_class[j][i] == 1])
                tn = sum([1 for i in range(len(targets_by_class[j])) if pred_by_class[j][i] == 0 and targets_by_class[j][i] == 0])
                fp = sum([1 for i in range(len(targets_by_class[j])) if pred_by_class[j][i] == 1 and targets_by_class[j][i] == 0])
                fn = sum([1 for i in range(len(targets_by_class[j])) if pred_by_class[j][i] == 0 and targets_by_class[j][i] == 1])

                try:
                    sensitivity = tp / (tp + fn)
                except ZeroDivisionError:
                    sensitivity = 0
                
                try:
                    specificity = tn / (tn + fp)
                except ZeroDivisionError:
                    specificity = 0

                roc_x = 1 - specificity
                
                try:
                    precision = tp / (tp + fp)
                except ZeroDivisionError:
                    precision = 0

                sensitivities_by_class.append(sensitivity)
                specificities_by_class.append(specificity)
                roc_x_by_class.append(roc_x)
                precision_by_class.append(precision)

            sensitivities_by_split.append(sensitivities_by_class)
            specificities_by_split.append(specificities_by_class)
            roc_x_by_split.append(roc_x_by_class)
            precision_by_split.append(precision_by_class)

        if index == 0:
            weights = weights_by_split

        sensitivities_by_thresh[thresh] = sensitivities_by_split
        specificities_by_thresh[thresh] = specificities_by_split
        roc_x_by_thresh[thresh] = roc_x_by_split
        precisions_by_thresh[thresh] = precision_by_split

    # Calculate average metrics
    avg_AUROC, avg_AUPRC = get_avg_AUCs(sensitivities_by_thresh, roc_x_by_thresh, precisions_by_thresh, weights)

    # Create plot data
    avg_sens_data = get_avg_metrics_for_plotting(sensitivities_by_thresh, weights)
    avg_spec_data = get_avg_metrics_for_plotting(specificities_by_thresh, weights)
    avg_roc_x_data = get_avg_metrics_for_plotting(roc_x_by_thresh, weights)
    avg_prec_data = get_avg_metrics_for_plotting(precisions_by_thresh, weights)

    roc_data = restructure_for_plotting(avg_roc_x_data, avg_sens_data)
    prc_data = restructure_for_plotting(avg_sens_data, avg_prec_data)

    return avg_AUROC, avg_AUPRC, roc_data, prc_data

def get_avg_AUCs(sens_by_thresh, roc_x_by_thresh, prec_by_thresh, weights):
    AUROC_by_split = []
    AUPRC_by_split = []

    for s in range(5):
        AUROC_by_class = []
        AUPRC_by_class = []

        for i in range(4):
            roc_x = []
            roc_y = []
            prc_x = []
            prc_y = []

            for thresh in sens_by_thresh.keys():
                roc_x.append(roc_x_by_thresh[thresh][s][i])
                roc_y.append(sens_by_thresh[thresh][s][i])
                prc_x.append(sens_by_thresh[thresh][s][i])
                prc_y.append(prec_by_thresh[thresh][s][i])

            auroc = auc(roc_x, roc_y)
            auprc = auc(prc_x, prc_y)

            AUROC_by_class.append(auroc)
            AUPRC_by_class.append(auprc)
        
        avg_AUROC, std_AUROC = weighted_std(AUROC_by_class, weights[s])
        avg_AUPRC, std_AUPRC = weighted_std(AUPRC_by_class, weights[s])

        AUROC_by_split.append((avg_AUROC, std_AUROC))
        AUPRC_by_split.append((avg_AUPRC, std_AUPRC))

    avg_AUROC = np.mean([auroc[0] for auroc in AUROC_by_split])
    avg_AUPRC = np.mean([auprc[0] for auprc in AUPRC_by_split])

    std_AUROC = np.sqrt(sum([auroc[1]**2 for auroc in AUROC_by_split]) / len(AUROC_by_split))
    std_AUPRC = np.sqrt(sum([auprc[1]**2 for auprc in AUPRC_by_split]) / len(AUPRC_by_split))

    lower_AUROC = avg_AUROC - 1.96 * std_AUROC
    upper_AUROC = avg_AUROC + 1.96 * std_AUROC
    lower_AUPRC = avg_AUPRC - 1.96 * std_AUPRC
    upper_AUPRC = avg_AUPRC + 1.96 * std_AUPRC

    return [avg_AUROC, std_AUROC, lower_AUROC, upper_AUROC], [avg_AUPRC, std_AUPRC, lower_AUPRC, upper_AUPRC]

def get_avg_metrics_for_plotting(metric_by_thresh, class_weights_by_cv):
    plot_metrics = {}

    for thresh in metric_by_thresh.keys():
        avg_metrics_by_split = []

        for index, cv in enumerate(metric_by_thresh[thresh]):
            class_weights = class_weights_by_cv[index]
            avg_metric, std_metric = weighted_std(cv, class_weights)

            avg_metrics_by_split.append((avg_metric, std_metric))

        avg_avg_metric = np.mean([metric[0] for metric in avg_metrics_by_split])
        avg_std_metric = np.sqrt(sum([metric[1]**2 for metric in avg_metrics_by_split]) / len(avg_metrics_by_split)) 
        # avg_upper_metric = avg_avg_metric + 1.96 * avg_std_metric
        # avg_lower_metric = avg_avg_metric - 1.96 * avg_std_metric

        # plot_metrics[thresh] = (avg_avg_metric, avg_upper_metric, avg_lower_metric)

        plot_metrics[thresh] = (avg_avg_metric, avg_std_metric)
    
    return plot_metrics

def get_AUCs_from_output(output):
    threshold_list = list(np.linspace(0, 1, 101))
    probas, targets = restructure_output(output)

    try:
        auroc, auprc, roc_data, prc_data = get_metrics_from_probas(probas, targets, threshold_list)
    except:
        print('Error in AUC calculation')
        return [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], None, None

    return auroc, auprc, roc_data, prc_data
                

def get_data_from_raw_output(file):
    with open(file) as f:
        data = json.load(f)
        df = pd.DataFrame(data['data'], columns = data['columns'])
    
    return df

def target_transform(num):
    if num == 0:
        return 0
    elif num > 0 and num <= 3:
        return 1
    elif num > 3 and num <= 8:
        return 2
    elif num > 8:
        return 3
    else:
        return int(-10)
    
def back_transform_classes(str):
    if str == '0':
        return 0
    elif str == '0_3':
        return 1
    elif str == '3_8':
        return 2
    elif str == '8_up':
        return 3
    else:
        return int(-10)

def weighted_std(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    std = np.sqrt(variance) / np.sqrt(len(values))
    return average, std

def weighted_conf_int(values, weights):
    average, std =  weighted_std(values, weights)
    return average - 1.96 * std, average + 1.96 * std

def get_acc_sens_spec_AGNI(df): # DEPRICATED
    result = []
    filt_result = []

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for obs in range(df.shape[0]):
        if df.targets[obs] == 0 and df.predictions[obs] == 0:
            tn += 1
        elif df.targets[obs] == df.predictions[obs]:
            tp += 1
        elif df.targets[obs] > df.predictions[obs]:
            fn += 1
        elif df.targets[obs] < df.predictions[obs]:
            fp += 1
        else:
            print('Error in AGNI calculation')
            return

    acc = (tp + tn) / df.shape[0]
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)

    result.append(acc)
    result.append(sens)
    result.append(spec)

    df_filt = df[df.targets != df.mls_categories].reset_index(drop = True)

    tp_filt = 0
    tn_filt = 0
    fp_filt = 0
    fn_filt = 0

    for obs_filt in range(df_filt.shape[0]):
        if df_filt.targets[obs_filt] == 0 and df_filt.predictions[obs_filt] == 0:
            tn_filt += 1
        elif df_filt.targets[obs_filt] == df_filt.predictions[obs_filt]:
            tp_filt += 1
        elif df_filt.targets[obs_filt] > df_filt.predictions[obs_filt]:
            fn_filt += 1
        elif df_filt.targets[obs_filt] < df_filt.predictions[obs_filt]:
            fp_filt += 1
        else:
            print('Error in AGNI filtered calculation')
            return
    
    acc_filt = (tp_filt + tn_filt) / df_filt.shape[0]
    sens_filt = tp_filt / (tp_filt + fn_filt)
    spec_filt = tn_filt / (tn_filt + fp_filt)

    filt_result.append(acc_filt)
    filt_result.append(sens_filt)
    filt_result.append(spec_filt)

    return result, filt_result

def get_acc_sens_spec_WORSEN(df):
    result = []
    result_filt = []

    acc = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for obs in range(df.shape[0]):
        if df.targets[obs] == df.predictions[obs]:
            acc += 1

        true = df.targets[obs] > df.mls_categories[obs]
        pred = df.predictions[obs] > df.mls_categories[obs]

        if true and pred:
            tp += 1
        elif not true and not pred:
            tn += 1
        elif not true and pred:
            fp += 1
        elif true and not pred:
            fn += 1
        else:
            print('Error in WORSEN calculation')
            return
    
    accuracy = acc / df.shape[0]
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)

    result.append(accuracy)
    result.append(sens)
    result.append(spec)

    df_filt = df[df.targets != df.mls_categories].reset_index(drop = True)
    
    acc_filt = 0
    tp_filt = 0
    tn_filt = 0
    fp_filt = 0
    fn_filt = 0

    for obs_filt in range(df_filt.shape[0]):
        if df_filt.targets[obs_filt] == df_filt.predictions[obs_filt]:
            acc_filt += 1
        
        true_filt = df_filt.targets[obs_filt] > df_filt.mls_categories[obs_filt]
        pred_filt = df_filt.predictions[obs_filt] > df_filt.mls_categories[obs_filt]

        if true_filt and pred_filt:
            tp_filt += 1
        elif not true_filt and not pred_filt:
            tn_filt += 1
        elif not true_filt and pred_filt:
            fp_filt += 1
        elif true_filt and not pred_filt:
            fn_filt += 1
        else:
            print('Error in WORSEN filtered calculation')
            return
        
    accuracy_filt = acc_filt / df_filt.shape[0]
    sens_filt = tp_filt / (tp_filt + fn_filt)
    spec_filt = tn_filt / (tn_filt + fp_filt)

    result_filt.append(accuracy_filt)
    result_filt.append(sens_filt)
    result_filt.append(spec_filt)

    return result, result_filt


def print_results(results, filt_results):
    print(f"Accuracy: {results[0][0]*100:.1f}% (95% CI [{results[0][2]*100:.1f}%, {results[0][3]*100:.1f}%])")
    print(f"Filtered Accuracy: {filt_results[0][0]*100:.1f}% (95% CI [{filt_results[0][2]*100:.1f}%, {filt_results[0][3]*100:.1f}]%)")
    print(f"Sensitivity: {results[1][0]*100:.1f}% (95% CI [{results[1][2]*100:.1f}%, {results[1][3]*100:.1f}%])")
    print(f"Filtered Sensitivity: {filt_results[1][0]*100:.1f}% (95% CI [{filt_results[1][2]*100:.1f}%, {filt_results[1][3]*100:.1f}%])")
    print(f"Specificity: {results[2][0]*100:.1f}% (95% CI [{results[2][2]*100:.1f}%, {results[2][3]*100:.1f}%])")
    print(f"Filtered Specificity: {filt_results[2][0]*100:.1f}% (95% CI [{filt_results[2][2]*100:.1f}%, {filt_results[2][3]*100:.1f}%])")

def print_AUCs(AUROC, filt_AUROC, AUPRC, filt_AUPRC):
    print(f'AUROC: {AUROC[0]*100:.1f}% (95% CI [{AUROC[2]*100:.1f}%, {AUROC[3]*100:.1f}%])')
    print(f'Filtered AUROC: {filt_AUROC[0]*100:.1f}% (95% CI [{filt_AUROC[2]*100:.1f}%, {filt_AUROC[3]*100:.1f}%])')
    print(f'AUPRC: {AUPRC[0]*100:.1f}% (95% CI [{AUPRC[2]*100:.1f}%, {AUPRC[3]*100:.1f}%])')
    print(f'Filtered AUPRC: {filt_AUPRC[0]*100:.1f}% (95% CI [{filt_AUPRC[2]*100:.1f}%, {filt_AUPRC[3]*100:.1f}%])')

def print_AUROC(AUROC, filt_AUROC):
    print(f'AUROC: {AUROC[0]*100:.1f}% (95% CI [{AUROC[2]*100:.1f}%, {AUROC[3]*100:.1f}%])')
    print(f'Filtered AUROC: {filt_AUROC[0]*100:.1f}% (95% CI [{filt_AUROC[2]*100:.1f}%, {filt_AUROC[3]*100:.1f}%])')

def pull_HELMET_output(lookahead_hours):
    # Load data
    cv0_mgb = get_data_from_raw_output(f'results/raw_model_output/{lookahead_hours}hr/predictions_and_features_v7_cv0_mgb.json')
    cv0_bmc = get_data_from_raw_output(f'results/raw_model_output/{lookahead_hours}hr/predictions_and_features_v7_cv0_bmc.json')

    cv1_mgb = get_data_from_raw_output(f'results/raw_model_output/{lookahead_hours}hr/predictions_and_features_v7_cv1_mgb.json')
    cv1_bmc = get_data_from_raw_output(f'results/raw_model_output/{lookahead_hours}hr/predictions_and_features_v7_cv1_bmc.json')

    cv2_mgb = get_data_from_raw_output(f'results/raw_model_output/{lookahead_hours}hr/predictions_and_features_v7_cv2_mgb.json')
    cv2_bmc = get_data_from_raw_output(f'results/raw_model_output/{lookahead_hours}hr/predictions_and_features_v7_cv2_bmc.json')

    cv3_mgb = get_data_from_raw_output(f'results/raw_model_output/{lookahead_hours}hr/predictions_and_features_v7_cv3_mgb.json')
    cv3_bmc = get_data_from_raw_output(f'results/raw_model_output/{lookahead_hours}hr/predictions_and_features_v7_cv3_bmc.json')

    cv4_mgb = get_data_from_raw_output(f'results/raw_model_output/{lookahead_hours}hr/predictions_and_features_v7_cv4_mgb.json')
    cv4_bmc = get_data_from_raw_output(f'results/raw_model_output/{lookahead_hours}hr/predictions_and_features_v7_cv4_bmc.json')

    mgb_insurance = pd.read_csv('data/raw/mgb_insurance.csv')
    bmc_insurance = pd.read_csv('data/raw/bmc_insurance.csv')

    mgb_uninsured = list(mgb_insurance[mgb_insurance['uninsured'] == 1]['ptid'])
    bmc_uninsured = list(bmc_insurance[bmc_insurance['uninsured'] == 1]['ptid'])

    mgb_medicaid = list(mgb_insurance[mgb_insurance['medicaid'] == 1]['ptid'])
    bmc_medicaid = list(bmc_insurance[bmc_insurance['medicaid'] == 1]['ptid'])

    mgb_other = list(mgb_insurance['ptid'][~np.isin(mgb_insurance['ptid'], mgb_uninsured + mgb_medicaid)])
    bmc_other = list(bmc_insurance['ptid'][~np.isin(bmc_insurance['ptid'], bmc_uninsured + bmc_medicaid)])

    #mgb_other = list(mgb_insurance[~((mgb_insurance['uninsured'] == 1) | (mgb_insurance['medicaid'] == 1))]['ptid'])
    #bmc_other = list(bmc_insurance[~((mgb_insurance['uninsured'] == 1) | (mgb_insurance['medicaid'] == 1))]['ptid'])

    # get raw output data in useable format
    cv_mgb_output = pd.DataFrame(columns = ['cv', 'ptid', 'day', 'mls_categories', 'targets', 'predictions', 'proba_0', 'proba_1', 'proba_2', 'proba_3'])
    cv_bmc_output = pd.DataFrame(columns = ['cv', 'mls_categories', 'targets', 'predictions', 'proba_0', 'proba_1', 'proba_2', 'proba_3'])

    for i, df in enumerate([cv0_mgb, cv1_mgb, cv2_mgb, cv3_mgb, cv4_mgb]):
        mgb_output = pd.DataFrame({
            'ptid': df['ptid'].astype(int), 
            'day': df['day of data'].astype(float),
            'mls_categories': df['mls_categories'].apply(back_transform_classes), 
            'targets': df['targets'].apply(back_transform_classes), 
            'predictions': df['predictions'].apply(back_transform_classes), 
            'proba_0': df['0_prediction_probability'].astype(float), 
            'proba_1': df['0_3_prediction_probability'].astype(float), 
            'proba_2': df['3_8_prediction_probability'].astype(float), 
            'proba_3': df['8_up_prediction_probability'].astype(float)
        })
        mgb_output['cv'] = i

        cv_mgb_output = pd.concat([cv_mgb_output, mgb_output], ignore_index=True)
    
    for i, df in enumerate([cv0_bmc, cv1_bmc, cv2_bmc, cv3_bmc, cv4_bmc]):
        bmc_output = pd.DataFrame({
            'ptid': df['ptid'].astype(int),
            'day': df['day of data'].astype(float),
            'mls_categories': df['mls_categories'].apply(back_transform_classes),
            'targets': df['targets'].apply(back_transform_classes),
            'predictions': df['predictions'].apply(back_transform_classes),
            'proba_0': df['0_prediction_probability'].astype(float),
            'proba_1': df['0_3_prediction_probability'].astype(float),
            'proba_2': df['3_8_prediction_probability'].astype(float),
            'proba_3': df['8_up_prediction_probability'].astype(float)
        })
        bmc_output['cv'] = i

        cv_bmc_output = pd.concat([cv_bmc_output, bmc_output], ignore_index=True)

    cv_mgb_filt_output = cv_mgb_output[cv_mgb_output['targets'] != cv_mgb_output['mls_categories']].reset_index(drop = True)
    cv_bmc_filt_output = cv_bmc_output[cv_bmc_output['targets'] != cv_bmc_output['mls_categories']].reset_index(drop = True)

    cv_mgb_output_24 = cv_mgb_output[cv_mgb_output['day'] == 0.0].reset_index(drop = True)
    cv_mgb_output_24_48 = cv_mgb_output[cv_mgb_output['day'] == 1.0].reset_index(drop = True)
    cv_mgb_output_48_96 = cv_mgb_output[(cv_mgb_output['day'] == 2.0) | (cv_mgb_output['day'] == 3.0) ].reset_index(drop = True)
    cv_mgb_output_96 = cv_mgb_output[cv_mgb_output['day'] > 3.0].reset_index(drop = True)

    cv_mgb_filt_output_24 = cv_mgb_filt_output[cv_mgb_filt_output['day'] == 0.0].reset_index(drop = True)
    cv_mgb_filt_output_24_48 = cv_mgb_filt_output[cv_mgb_filt_output['day'] == 1.0].reset_index(drop = True)
    cv_mgb_filt_output_48_96 = cv_mgb_filt_output[(cv_mgb_filt_output['day'] == 2.0) | (cv_mgb_filt_output['day'] == 3.0) ].reset_index(drop = True)
    cv_mgb_filt_output_96 = cv_mgb_filt_output[cv_mgb_filt_output['day'] > 3.0].reset_index(drop = True)

    cv_bmc_output_24 = cv_bmc_output[cv_bmc_output['day'] == 0.0].reset_index(drop = True)
    cv_bmc_output_24_48 = cv_bmc_output[cv_bmc_output['day'] == 1.0].reset_index(drop = True)
    cv_bmc_output_48_96 = cv_bmc_output[(cv_bmc_output['day'] == 2.0) | (cv_bmc_output['day'] == 3.0) ].reset_index(drop = True)
    cv_bmc_output_96 = cv_bmc_output[cv_bmc_output['day'] > 3.0].reset_index(drop = True)

    cv_bmc_filt_output_24 = cv_bmc_filt_output[cv_bmc_filt_output['day'] == 0.0].reset_index(drop = True)
    cv_bmc_filt_output_24_48 = cv_bmc_filt_output[cv_bmc_filt_output['day'] == 1.0].reset_index(drop = True)
    cv_bmc_filt_output_48_96 = cv_bmc_filt_output[(cv_bmc_filt_output['day'] == 2.0) | (cv_bmc_filt_output['day'] == 3.0) ].reset_index(drop = True)
    cv_bmc_filt_output_96 = cv_bmc_filt_output[cv_bmc_filt_output['day'] > 3.0].reset_index(drop = True)

    cv_mgb_output_uninsured = cv_mgb_output[cv_mgb_output['ptid'].isin(mgb_uninsured)].reset_index(drop = True)
    cv_mgb_output_medicaid = cv_mgb_output[cv_mgb_output['ptid'].isin(mgb_medicaid)].reset_index(drop = True)
    cv_mgb_output_other = cv_mgb_output[cv_mgb_output['ptid'].isin(mgb_other)].reset_index(drop = True)

    cv_bmc_output_uninsured = cv_bmc_output[cv_bmc_output['ptid'].isin(bmc_uninsured)].reset_index(drop = True)
    cv_bmc_output_medicaid = cv_bmc_output[cv_bmc_output['ptid'].isin(bmc_medicaid)].reset_index(drop = True)
    cv_bmc_output_other = cv_bmc_output[cv_bmc_output['ptid'].isin(bmc_other)].reset_index(drop = True)

    cv_mgb_filt_output_uninsured = cv_mgb_filt_output[cv_mgb_filt_output['ptid'].isin(mgb_uninsured)].reset_index(drop = True)
    cv_mgb_filt_output_medicaid = cv_mgb_filt_output[cv_mgb_filt_output['ptid'].isin(mgb_medicaid)].reset_index(drop = True)
    cv_mgb_filt_output_other = cv_mgb_filt_output[cv_mgb_filt_output['ptid'].isin(mgb_other)].reset_index(drop = True)

    cv_bmc_filt_output_uninsured = cv_bmc_filt_output[cv_bmc_filt_output['ptid'].isin(bmc_uninsured)].reset_index(drop = True)
    cv_bmc_filt_output_medicaid = cv_bmc_filt_output[cv_bmc_filt_output['ptid'].isin(bmc_medicaid)].reset_index(drop = True)
    cv_bmc_filt_output_other = cv_bmc_filt_output[cv_bmc_filt_output['ptid'].isin(bmc_other)].reset_index(drop = True)

    mgb_sets = [cv_mgb_output, cv_mgb_filt_output, cv_mgb_output_24, cv_mgb_filt_output_24, cv_mgb_output_24_48, cv_mgb_filt_output_24_48, cv_mgb_output_48_96, cv_mgb_filt_output_48_96, cv_mgb_output_96, cv_mgb_filt_output_96, cv_mgb_output_uninsured, cv_mgb_filt_output_uninsured, cv_mgb_output_medicaid, cv_mgb_filt_output_medicaid, cv_mgb_output_other, cv_mgb_filt_output_other]
    bmc_sets = [cv_bmc_output, cv_bmc_filt_output, cv_bmc_output_24, cv_bmc_filt_output_24, cv_bmc_output_24_48, cv_bmc_filt_output_24_48, cv_bmc_output_48_96, cv_bmc_filt_output_48_96, cv_bmc_output_96, cv_bmc_filt_output_96, cv_bmc_output_uninsured, cv_bmc_filt_output_uninsured, cv_bmc_output_medicaid, cv_bmc_filt_output_medicaid, cv_bmc_output_other, cv_bmc_filt_output_other]

    return mgb_sets, bmc_sets

#_______________________________________________________________________________________________________________________
def make_edema_dataset(OLTV_df, OTV_df):
    df = OTV_df[["ptid", "gluc1", "stroke", "hba1c", "tpa", "mt"]]
    mls_components = pd.DataFrame(OLTV_df[["size_mls", "gluc", "bce"]], columns = ["size_mls", "gluc", "bce"])
    mls_components["dt"] = OLTV_df["dt"]
    mls_components["ptid"] = OLTV_df["ptid"]
    mls_components["targets"] = OLTV_df["targets"]
    df = df.merge(mls_components, on="ptid", how="inner")

    targets = df["targets"]
    df = df.drop(columns=["targets"], axis=1)

    df['day'] = df['dt'].apply(lambda x: x // 24)
    df.drop(columns = ['dt'], inplace=True)

    return df, targets

def get_scalers(df, test_ptids, lookahead_hours):
    
    scaler_list = []

    for i in range(5):
        train_df = df[~df.ptid.isin(test_ptids[i])]
        train_df.drop(columns = ['ptid', 'targets'], inplace=True)

        if not PRE_SAVED:
            scaler = StandardScaler().fit(train_df)
            scaler_list.append(scaler)

            pkl_filename = f"results/models/scaler_{lookahead_hours}hrs_{i}.pkl"
            with open(pkl_filename, 'wb') as file:
                pickle.dump(scaler, file)
        else:
            pkl_filename = f"results/models/scaler_{lookahead_hours}hrs_{i}.pkl"
            with open(pkl_filename, 'rb') as file:
                scaler = pickle.load(file)
                scaler_list.append(scaler)

    return scaler_list

def normalise_data(df, scaler):
    preserve_ptids = df.ptid.copy()
    preserve_targets = df.targets.copy()
    df.drop(columns = ['ptid', 'targets'], inplace=True)
    cols = df.columns
    
    scaler_cols = scaler.feature_names_in_
    df = df[scaler_cols]

    df = scaler.transform(df)
    df = pd.DataFrame(df, columns = cols)
    df['ptid'] = preserve_ptids.values
    df['targets'] = preserve_targets.values
    return df


def create_edema_baseline(OLTV_df, OTV_df, test_ptids, scaler_list, lookahead_hours, hosp):

    cv_output = pd.DataFrame(columns = ['cv', 'ptid', 'day', 'mls_categories', 'targets', 'predictions', 'proba_0', 'proba_1', 'proba_2', 'proba_3'])

    for cv_num in range(5):

        # Split data into train and test sets
        train_df = OLTV_df[~OLTV_df.ptid.isin(test_ptids[cv_num])]
        test_df = OLTV_df[OLTV_df.ptid.isin(test_ptids[cv_num])]

        OTV_train = OTV_df[~OTV_df.ptid.isin(test_ptids[cv_num])]
        OTV_test = OTV_df[OTV_df.ptid.isin(test_ptids[cv_num])]

        preserve_mls_categories = test_df.size_mls.copy().apply(target_transform)
        preserve_targets = test_df.targets.copy()

        # Apply normalisation
        #train_df = normalise_data(train_df, scaler_list[cv_num])
        #test_df = normalise_data(test_df, scaler_list[cv_num])

        # make new dataframes for edema score model
        edema_train_df, edema_train_targets = make_edema_dataset(train_df, OTV_train)
        edema_test_df, edema_test_targets = make_edema_dataset(test_df, OTV_test)

        preserve_ptids = edema_test_df.ptid.copy()
        preserve_days = edema_test_df.day.copy()

        # Drop ptid and day columns
        edema_train_df.drop(columns = ['ptid', 'day'], inplace=True)
        edema_test_df.drop(columns = ['ptid', 'day'], inplace=True)

        edema_df_columns = edema_train_df.columns

        # Create or load imputer
        if not PRE_SAVED:
            imputer = SimpleImputer()
            imputer.fit(edema_train_df)

            with open(f"results/models/imputer_{hosp}_{lookahead_hours}hrs_{cv_num}.pkl", 'wb') as file:
                pickle.dump(imputer, file)

        else:
            with open(f"results/models/imputer_{hosp}_{lookahead_hours}hrs_{cv_num}.pkl", 'rb') as file:
                imputer = pickle.load(file)

        # Impute missing values
        edema_train_df = imputer.transform(edema_train_df)
        edema_test_df = imputer.transform(edema_test_df)

        edema_train_df = pd.DataFrame(edema_train_df, columns = edema_df_columns)
        edema_test_df = pd.DataFrame(edema_test_df, columns = edema_df_columns)

        # Fit or load model
        if not PRE_SAVED:
            edema_score_model = LogisticRegression(max_iter = 3000).fit(edema_train_df, edema_train_targets)
            print(f'saving new model: {hosp}_{lookahead_hours}hrs_{cv_num}')
            pkl_filename = f"results/models/EDEMA_model_{hosp}_{lookahead_hours}hrs_{cv_num}.pkl"
            with open(pkl_filename, 'wb') as file:
                pickle.dump(edema_score_model, file)

        else:
            pkl_filename = f"results/models/EDEMA_model_{hosp}_{lookahead_hours}hrs_{cv_num}.pkl"
            print(f'loading model: {hosp}_{lookahead_hours}hrs_{cv_num}')
            with open(pkl_filename, 'rb') as file:
                edema_score_model = pickle.load(file)

        model_cols = edema_score_model.feature_names_in_
        edema_test_df = edema_test_df[model_cols]
        
        predictions = edema_score_model.predict(edema_test_df)
        proba = edema_score_model.predict_proba(edema_test_df)
        proba_0 = [p[0] for p in proba]
        proba_1 = [p[1] for p in proba]
        proba_2 = [p[2] for p in proba]
        proba_3 = [p[3] for p in proba]

        output = pd.DataFrame(
            {
                'ptid': preserve_ptids.values,
                'day': preserve_days.values,
                'mls_categories': preserve_mls_categories.values, 
                'targets': preserve_targets.values, 
                'predictions': predictions, 
                'proba_0': proba_0, 
                'proba_1': proba_1, 
                'proba_2': proba_2, 
                'proba_3': proba_3
            }
        )
        output['cv'] = cv_num
        cv_output = pd.concat([cv_output, output], ignore_index=True)
        
    return cv_output



def generate_EDEMA_output(lookahead_hours):
    # Load data
    if lookahead_hours == 8:
        cv0_mgb = get_data_from_raw_output('results/raw_model_output/8hr/predictions_and_features_v7_cv0_mgb.json')
        cv1_mgb = get_data_from_raw_output('results/raw_model_output/8hr/predictions_and_features_v7_cv1_mgb.json')
        cv2_mgb = get_data_from_raw_output('results/raw_model_output/8hr/predictions_and_features_v7_cv2_mgb.json')
        cv3_mgb = get_data_from_raw_output('results/raw_model_output/8hr/predictions_and_features_v7_cv3_mgb.json')
        cv4_mgb = get_data_from_raw_output('results/raw_model_output/8hr/predictions_and_features_v7_cv4_mgb.json')

        cv_test_ptids = [cv0_mgb.ptid.unique(), cv1_mgb.ptid.unique(), cv2_mgb.ptid.unique(), cv3_mgb.ptid.unique(), cv4_mgb.ptid.unique()]

        OLTV = pd.read_json('data/processed/OLTV_8_FINAL.json')
        targets = pd.read_json('data/processed/targets_8_FINAL.json')
        bmc_OLTV = pd.read_json('data/processed/bmc_OLTV_8_FINAL.json')
        bmc_targets = pd.read_json('data/processed/bmc_targets_8_FINAL.json')

    elif lookahead_hours == 24:
        cv0_mgb = get_data_from_raw_output('results/raw_model_output/24hr/predictions_and_features_v7_cv0_mgb.json')
        cv1_mgb = get_data_from_raw_output('results/raw_model_output/24hr/predictions_and_features_v7_cv1_mgb.json')
        cv2_mgb = get_data_from_raw_output('results/raw_model_output/24hr/predictions_and_features_v7_cv2_mgb.json')
        cv3_mgb = get_data_from_raw_output('results/raw_model_output/24hr/predictions_and_features_v7_cv3_mgb.json')
        cv4_mgb = get_data_from_raw_output('results/raw_model_output/24hr/predictions_and_features_v7_cv4_mgb.json')

        cv_test_ptids = [cv0_mgb.ptid.unique(), cv1_mgb.ptid.unique(), cv2_mgb.ptid.unique(), cv3_mgb.ptid.unique(), cv4_mgb.ptid.unique()]

        OLTV = pd.read_json('data/processed/OLTV_24_FINAL.json')
        targets = pd.read_json('data/processed/targets_24_FINAL.json')
        bmc_OLTV = pd.read_json('data/processed/bmc_OLTV_24_FINAL.json')
        bmc_targets = pd.read_json('data/processed/bmc_targets_24_FINAL.json')

    else:
        print('Invalid lookahead hours')
        return
    
    mgb_insurance = pd.read_csv('data/raw/mgb_insurance.csv')
    bmc_insurance = pd.read_csv('data/raw/bmc_insurance.csv')

    mgb_uninsured = list(mgb_insurance[mgb_insurance['uninsured'] == 1]['ptid'])
    bmc_uninsured = list(bmc_insurance[bmc_insurance['uninsured'] == 1]['ptid'])

    mgb_medicaid = list(mgb_insurance[mgb_insurance['medicaid'] == 1]['ptid'])
    bmc_medicaid = list(bmc_insurance[bmc_insurance['medicaid'] == 1]['ptid'])

    mgb_other = list(mgb_insurance['ptid'][~np.isin(mgb_insurance['ptid'], mgb_uninsured + mgb_medicaid)])
    bmc_other = list(bmc_insurance['ptid'][~np.isin(bmc_insurance['ptid'], bmc_uninsured + bmc_medicaid)])

    #mgb_other = list(mgb_insurance[~((mgb_insurance['uninsured'] == 1) | (mgb_insurance['medicaid'] == 1))]['ptid'])
    #bmc_other = list(bmc_insurance[~((mgb_insurance['uninsured'] == 1) | (mgb_insurance['medicaid'] == 1))]['ptid'])

    # Preprocess data
    cv_test_ptids = [[int(x) for x in array] for array in cv_test_ptids]

    OLTV['targets'] = targets['target']
    bmc_OLTV['targets'] = bmc_targets['target']

    OTV = pd.read_json('data/processed/OTV_FINAL.json')
    bmc_OTV = pd.read_json('data/processed/bmc_OTV_FINAL.json')

    OLTV.columns = [col.replace('ffill_', '') for col in OLTV.columns]
    bmc_OLTV.columns = [col.replace('ffill_', '') for col in bmc_OLTV.columns]

    OLTV_cols = list(set(OLTV.columns).intersection(set(bmc_OLTV.columns)))
    OLTV = OLTV[OLTV_cols]
    bmc_OLTV = bmc_OLTV[OLTV_cols]

    OLTV['targets'] = OLTV['targets'].apply(target_transform)
    OLTV = OLTV[OLTV['targets'] != -10]

    bmc_OLTV['targets'] = bmc_OLTV['targets'].apply(target_transform)
    bmc_OLTV = bmc_OLTV[bmc_OLTV['targets'] != -10]

    # save or load bmc test ptids
    if not PRE_SAVED:
        # split bmc patients into train and test sets
        bmc_ptids = bmc_OTV.ptid.unique()

        cv_bmc_test_ptids = []

        for i in range(5):
            train_ptids, test_ptids = train_test_split(bmc_ptids, test_size = 0.2)
            cv_bmc_test_ptids.append(test_ptids)

        with open(f'results/models/bmc_test_ptids_{lookahead_hours}hrs.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerows(cv_bmc_test_ptids)

    else:    
        with open(f'results/models/bmc_test_ptids_{lookahead_hours}hrs.csv', 'r') as file:
            reader = csv.reader(file)
            cv_bmc_test_ptids = list(reader)
            cv_bmc_test_ptids = [x for x in cv_bmc_test_ptids if x != []]
            cv_bmc_test_ptids = [[int(x) for x in array] for array in cv_bmc_test_ptids]

    # get list of scaler objects for each cv split
    scaler_list = get_scalers(OLTV, cv_test_ptids, lookahead_hours)

    mgb_output = create_edema_baseline(OLTV, OTV, cv_test_ptids, scaler_list, lookahead_hours, 'mgb')
    bmc_output = create_edema_baseline(bmc_OLTV, bmc_OTV, cv_bmc_test_ptids, scaler_list, lookahead_hours, 'bmc')

    mgb_filt_output = mgb_output[mgb_output['targets'] != mgb_output['mls_categories']].reset_index(drop = True)
    bmc_filt_output = bmc_output[bmc_output['targets'] != bmc_output['mls_categories']].reset_index(drop = True)

    mgb_insurance = pd.read_csv('data/raw/mgb_insurance.csv')
    bmc_insurance = pd.read_csv('data/raw/bmc_insurance.csv')

    mgb_uninsured = list(mgb_insurance[mgb_insurance['uninsured'] == 1]['ptid'])
    bmc_uninsured = list(bmc_insurance[bmc_insurance['uninsured'] == 1]['ptid'])

    mgb_medicaid = list(mgb_insurance[mgb_insurance['medicaid'] == 1]['ptid'])
    bmc_medicaid = list(bmc_insurance[bmc_insurance['medicaid'] == 1]['ptid'])

    mgb_other = list(mgb_insurance['ptid'][~np.isin(mgb_insurance['ptid'], mgb_uninsured + mgb_medicaid)])
    bmc_other = list(bmc_insurance['ptid'][~np.isin(bmc_insurance['ptid'], bmc_uninsured + bmc_medicaid)])

    #mgb_other = list(mgb_insurance[~((mgb_insurance['uninsured'] == 1) | (mgb_insurance['medicaid'] == 1))]['ptid'])
    #bmc_other = list(bmc_insurance[~((mgb_insurance['uninsured'] == 1) | (mgb_insurance['medicaid'] == 1))]['ptid'])

    mgb_output_24 = mgb_output[mgb_output['day'] == 0.0].reset_index(drop = True)
    mgb_output_24_48 = mgb_output[mgb_output['day'] == 1.0].reset_index(drop = True)
    mgb_output_48_96 = mgb_output[(mgb_output['day'] == 2.0) | (mgb_output['day'] == 3.0) ].reset_index(drop = True)
    mgb_output_96 = mgb_output[mgb_output['day'] > 3.0].reset_index(drop = True)

    mgb_filt_output_24 = mgb_filt_output[mgb_filt_output['day'] == 0.0].reset_index(drop = True)
    mgb_filt_output_24_48 = mgb_filt_output[mgb_filt_output['day'] == 1.0].reset_index(drop = True)
    mgb_filt_output_48_96 = mgb_filt_output[(mgb_filt_output['day'] == 2.0) | (mgb_filt_output['day'] == 3.0) ].reset_index(drop = True)
    mgb_filt_output_96 = mgb_filt_output[mgb_filt_output['day'] > 3.0].reset_index(drop = True)

    bmc_output_24 = bmc_output[bmc_output['day'] == 0.0].reset_index(drop = True)
    bmc_output_24_48 = bmc_output[bmc_output['day'] == 1.0].reset_index(drop = True)
    bmc_output_48_96 = bmc_output[(bmc_output['day'] == 2.0) | (bmc_output['day'] == 3.0) ].reset_index(drop = True)
    bmc_output_96 = bmc_output[bmc_output['day'] > 3.0].reset_index(drop = True)

    bmc_filt_output_24 = bmc_filt_output[bmc_filt_output['day'] == 0.0].reset_index(drop = True)
    bmc_filt_output_24_48 = bmc_filt_output[bmc_filt_output['day'] == 1.0].reset_index(drop = True)
    bmc_filt_output_48_96 = bmc_filt_output[(bmc_filt_output['day'] == 2.0) | (bmc_filt_output['day'] == 3.0) ].reset_index(drop = True)
    bmc_filt_output_96 = bmc_filt_output[bmc_filt_output['day'] > 3.0].reset_index(drop = True)

    mgb_output_uninsured = mgb_output[mgb_output['ptid'].isin(mgb_uninsured)].reset_index(drop = True)
    mgb_output_medicaid = mgb_output[mgb_output['ptid'].isin(mgb_medicaid)].reset_index(drop = True)
    mgb_output_other = mgb_output[mgb_output['ptid'].isin(mgb_other)].reset_index(drop = True)

    bmc_output_uninsured = bmc_output[bmc_output['ptid'].isin(bmc_uninsured)].reset_index(drop = True)
    bmc_output_medicaid = bmc_output[bmc_output['ptid'].isin(bmc_medicaid)].reset_index(drop = True)
    bmc_output_other = bmc_output[bmc_output['ptid'].isin(bmc_other)].reset_index(drop = True)

    mgb_filt_output_uninsured = mgb_filt_output[mgb_filt_output['ptid'].isin(mgb_uninsured)].reset_index(drop = True)
    mgb_filt_output_medicaid = mgb_filt_output[mgb_filt_output['ptid'].isin(mgb_medicaid)].reset_index(drop = True)
    mgb_filt_output_other = mgb_filt_output[mgb_filt_output['ptid'].isin(mgb_other)].reset_index(drop = True)

    bmc_filt_output_uninsured = bmc_filt_output[bmc_filt_output['ptid'].isin(bmc_uninsured)].reset_index(drop = True)
    bmc_filt_output_medicaid = bmc_filt_output[bmc_filt_output['ptid'].isin(bmc_medicaid)].reset_index(drop = True)
    bmc_filt_output_other = bmc_filt_output[bmc_filt_output['ptid'].isin(bmc_other)].reset_index(drop = True)

    mgb_list = [mgb_output, mgb_filt_output, mgb_output_24, mgb_filt_output_24, mgb_output_24_48, mgb_filt_output_24_48, mgb_output_48_96, mgb_filt_output_48_96, mgb_output_96, mgb_filt_output_96, mgb_output_uninsured, mgb_filt_output_uninsured, mgb_output_medicaid, mgb_filt_output_medicaid, mgb_output_other, mgb_filt_output_other]
    bmc_list = [bmc_output, bmc_filt_output, bmc_output_24, bmc_filt_output_24, bmc_output_24_48, bmc_filt_output_24_48, bmc_output_48_96, bmc_filt_output_48_96, bmc_output_96, bmc_filt_output_96, bmc_output_uninsured, bmc_filt_output_uninsured, bmc_output_medicaid, bmc_filt_output_medicaid, bmc_output_other, bmc_filt_output_other]

    return mgb_list, bmc_list




#_______________________________________________________________________________________________________________________

def process_average_metrics(cv_results):
    avg_results = cv_results.mean(axis = 0).to_list()
    avg_results = list(map(lambda x: [x], avg_results))

    std_results = cv_results.std(axis = 0).to_list()
    _ = [x.append(y) for x, y in zip(avg_results, std_results)]

    lower_results = [avg_results[i][0] - 1.96 * avg_results[i][1] for i in range(3)]
    upper_results = [avg_results[i][0] + 1.96 * avg_results[i][1] for i in range(3)]

    _ = [x.append(y) for x, y in zip(avg_results, lower_results)]
    _ = [x.append(y) for x, y in zip(avg_results, upper_results)]

    return avg_results


def calc_metrics(cv_mgb_output, cv_mgb_filt_output, cv_bmc_output, cv_bmc_filt_output, lookahead_hours, total_obs_mgb=1, total_obs_bmc=1):

    # Calculate AUCs
    mgb_AUROC, mgb_AUPRC, mgb_ROC_data, mgb_PRC_data = get_AUCs_from_output(cv_mgb_output)
    mgb_filt_AUROC, mgb_filt_AUPRC, mgb_ROC_data_filt, mgb_PRC_data_filt = get_AUCs_from_output(cv_mgb_filt_output)
    bmc_AUROC, bmc_AUPRC, bmc_ROC_data, bmc_PRC_data = get_AUCs_from_output(cv_bmc_output)
    bmc_filt_AUROC, bmc_filt_AUPRC, bmc_ROC_data_filt, bmc_PRC_data_filt = get_AUCs_from_output(cv_bmc_filt_output)

    # Print results
    print(f"Results for MGB (n={len(np.unique(cv_mgb_output.ptid))}) (%obs={len(cv_mgb_output)/total_obs_mgb:.3f}): ")
    print_AUROC(mgb_AUROC, mgb_filt_AUROC)

    print(f"Results for BMC (n={len(np.unique(cv_bmc_output.ptid))}) (%obs={len(cv_bmc_output)/total_obs_bmc:.3f}): ")
    print_AUROC(bmc_AUROC, bmc_filt_AUROC)

    
def create_plots(xgb_overall, xgb_filt, edema_overall, edema_filt, type, hosp, hours):
    plt.figure()
    plt.plot(xgb_overall[0], xgb_overall[1], label=f'HELMET-{hours}', color = 'blue')
    #plt.fill_between(xgb_overall[0], xgb_overall[2], xgb_overall[3], alpha=0.2, facecolor = 'blue')
    plt.fill_between(xgb_overall[0], [avg - std for avg, std in zip(xgb_overall[1], xgb_overall[2])], [avg + std for avg, std in zip(xgb_overall[1], xgb_overall[2])], alpha=0.2, facecolor = 'blue')

    plt.plot(xgb_filt[0], xgb_filt[1], label=f'HELMET-{hours} (Filtered)', color = 'green')
    #plt.fill_between(xgb_filt[0], xgb_filt[2], xgb_filt[3], alpha=0.2, facecolor = 'green')
    plt.fill_between(xgb_filt[0], [avg - std for avg, std in zip(xgb_filt[1], xgb_filt[2])], [avg + std for avg, std in zip(xgb_filt[1], xgb_filt[2])], alpha=0.2, facecolor = 'green')

    plt.plot(edema_overall[0], edema_overall[1], label=f'EDEMA-{hours}', color = 'red')
    #plt.fill_between(edema_overall[0], edema_overall[2], edema_overall[3], alpha=0.2, facecolor = 'red')
    plt.fill_between(edema_overall[0], [avg - std for avg, std in zip(edema_overall[1], edema_overall[2])], [avg + std for avg, std in zip(edema_overall[1], edema_overall[2])], alpha=0.2, facecolor = 'red')

    plt.plot(edema_filt[0], edema_filt[1], label=f'EDEMA-{hours} (Filtered)', color = 'orange')
    #plt.fill_between(edema_filt[0], edema_filt[2], edema_filt[3], alpha=0.2, facecolor = 'orange')
    plt.fill_between(edema_filt[0], [avg - std for avg, std in zip(edema_filt[1], edema_filt[2])], [avg + std for avg, std in zip(edema_filt[1], edema_filt[2])], alpha=0.2, facecolor = 'orange')

    if type == 'roc':
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        if hosp == 'MGB':
            plt.title(f'ROC Curve for {hours}-Hour Prediction Task\n(Mass General Brigham Cohort)')
        else:
            plt.title(f'ROC Curve for {hours}-Hour Prediction Task\n(Boston Medical Center Cohort)')
    elif type == 'prc':
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        if hosp == 'MGB':
            plt.title(f'Precision-Recall Curve for {hours}-Hour Prediction Task\n(Mass General Brigham Cohort)')
        else:
            plt.title(f'Precision-Recall Curve for {hours}-Hour Prediction Task\n(Boston Medical Center Cohort)')
    
    ax = plt.gca()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.legend(loc = 'lower right')
    
    plt.savefig(f'results/figures/shap/{hosp}_{hours}hr_{type}_plot.png')
    print(f'Saved plot: {hosp}_{hours}hr_{type}_plot.png')
    plt.close()

if __name__ == "__main__":

    # Get data
    HELMET_mgb_list_8, HELMET_bmc_list_8 = pull_HELMET_output(8)
    HELMET_mgb_list_24, HELMET_bmc_list_24 = pull_HELMET_output(24)

    EDEMA_mgb_list_8, EDEMA_bmc_list_8 = generate_EDEMA_output(8)
    EDEMA_mgb_list_24, EDEMA_bmc_list_24 = generate_EDEMA_output(24)

    # Unpack data
    HELMET_mgb_8, HELMET_mgb_filt_8, HELMET_mgb_8_24, HELMET_mgb_filt_8_24, HELMET_mgb_8_24_48, HELMET_mgb_filt_8_24_48, HELMET_mgb_8_48_96, HELMET_mgb_filt_8_48_96, HELMET_mgb_8_96, HELMET_mgb_filt_8_96, HELMET_mgb_8_uninsured, HELMET_mgb_filt_8_uninsured, HELMET_mgb_8_medicaid, HELMET_mgb_filt_8_medicaid, HELMET_mgb_8_other, HELMET_mgb_filt_8_other = HELMET_mgb_list_8
    HELMET_bmc_8, HELMET_bmc_filt_8, HELMET_bmc_8_24, HELMET_bmc_filt_8_24, HELMET_bmc_8_24_48, HELMET_bmc_filt_8_24_48, HELMET_bmc_8_48_96, HELMET_bmc_filt_8_48_96, HELMET_bmc_8_96, HELMET_bmc_filt_8_96, HELMET_bmc_8_uninsured, HELMET_bmc_filt_8_uninsured, HELMET_bmc_8_medicaid, HELMET_bmc_filt_8_medicaid, HELMET_bmc_8_other, HELMET_bmc_filt_8_other = HELMET_bmc_list_8

    HELMET_mgb_24, HELMET_mgb_filt_24, HELMET_mgb_24_24, HELMET_mgb_filt_24_24, HELMET_mgb_24_24_48, HELMET_mgb_filt_24_24_48, HELMET_mgb_24_48_96, HELMET_mgb_filt_24_48_96, HELMET_mgb_24_96, HELMET_mgb_filt_24_96, HELMET_mgb_24_uninsured, HELMET_mgb_filt_24_uninsured, HELMET_mgb_24_medicaid, HELMET_mgb_filt_24_medicaid, HELMET_mgb_24_other, HELMET_mgb_filt_24_other = HELMET_mgb_list_24
    HELMET_bmc_24, HELMET_bmc_filt_24, HELMET_bmc_24_24, HELMET_bmc_filt_24_24, HELMET_bmc_24_24_48, HELMET_bmc_filt_24_24_48, HELMET_bmc_24_48_96, HELMET_bmc_filt_24_48_96, HELMET_bmc_24_96, HELMET_bmc_filt_24_96, HELMET_bmc_24_uninsured, HELMET_bmc_filt_24_uninsured, HELMET_bmc_24_medicaid, HELMET_bmc_filt_24_medicaid, HELMET_bmc_24_other, HELMET_bmc_filt_24_other = HELMET_bmc_list_24

    EDEMA_mgb_8, EDEMA_mgb_filt_8, EDEMA_mgb_8_24, EDEMA_mgb_filt_8_24, EDEMA_mgb_8_24_48, EDEMA_mgb_filt_8_24_48, EDEMA_mgb_8_48_96, EDEMA_mgb_filt_8_48_96, EDEMA_mgb_8_96, EDEMA_mgb_filt_8_96, EDEMA_mgb_8_uninsured, EDEMA_mgb_filt_8_uninsured, EDEMA_mgb_8_medicaid, EDEMA_mgb_filt_8_medicaid, EDEMA_mgb_8_other, EDEMA_mgb_filt_8_other = EDEMA_mgb_list_8
    EDEMA_bmc_8, EDEMA_bmc_filt_8, EDEMA_bmc_8_24, EDEMA_bmc_filt_8_24, EDEMA_bmc_8_24_48, EDEMA_bmc_filt_8_24_48, EDEMA_bmc_8_48_96, EDEMA_bmc_filt_8_48_96, EDEMA_bmc_8_96, EDEMA_bmc_filt_8_96, EDEMA_bmc_8_uninsured, EDEMA_bmc_filt_8_uninsured, EDEMA_bmc_8_medicaid, EDEMA_bmc_filt_8_medicaid, EDEMA_bmc_8_other, EDEMA_bmc_filt_8_other = EDEMA_bmc_list_8

    EDEMA_mgb_24, EDEMA_mgb_filt_24, EDEMA_mgb_24_24, EDEMA_mgb_filt_24_24, EDEMA_mgb_24_24_48, EDEMA_mgb_filt_24_24_48, EDEMA_mgb_24_48_96, EDEMA_mgb_filt_24_48_96, EDEMA_mgb_24_96, EDEMA_mgb_filt_24_96, EDEMA_mgb_24_uninsured, EDEMA_mgb_filt_24_uninsured, EDEMA_mgb_24_medicaid, EDEMA_mgb_filt_24_medicaid, EDEMA_mgb_24_other, EDEMA_mgb_filt_24_other = EDEMA_mgb_list_24
    EDEMA_bmc_24, EDEMA_bmc_filt_24, EDEMA_bmc_24_24, EDEMA_bmc_filt_24_24, EDEMA_bmc_24_24_48, EDEMA_bmc_filt_24_24_48, EDEMA_bmc_24_48_96, EDEMA_bmc_filt_24_48_96, EDEMA_bmc_24_96, EDEMA_bmc_filt_24_96, EDEMA_bmc_24_uninsured, EDEMA_bmc_filt_24_uninsured, EDEMA_bmc_24_medicaid, EDEMA_bmc_filt_24_medicaid, EDEMA_bmc_24_other, EDEMA_bmc_filt_24_other = EDEMA_bmc_list_24

    # Calculate metrics, print metrics, and get plot data

    total_obs_e24 = len(EDEMA_mgb_24)
    total_obs_e8 = len(EDEMA_mgb_8)
    total_obs_h24 = len(HELMET_mgb_24)
    total_obs_h8 = len(HELMET_mgb_8)

    total_obs_bmc_e24 = len(EDEMA_bmc_24)
    total_obs_bmc_e8 = len(EDEMA_bmc_8)
    total_obs_bmc_h24 = len(HELMET_bmc_24)
    total_obs_bmc_h8 = len(HELMET_bmc_8)




    print('EDEMA-24 (Overall)')
    calc_metrics(EDEMA_mgb_24, EDEMA_mgb_filt_24, EDEMA_bmc_24, EDEMA_bmc_filt_24, 24, total_obs_e24, total_obs_bmc_e24)

    print('EDEMA-24 (<24h)')
    calc_metrics(EDEMA_mgb_24_24, EDEMA_mgb_filt_24_24, EDEMA_bmc_24_24, EDEMA_bmc_filt_24_24, 24, total_obs_e24, total_obs_bmc_e24)

    print('EDEMA-24 (24-48h)')
    calc_metrics(EDEMA_mgb_24_24_48, EDEMA_mgb_filt_24_24_48, EDEMA_bmc_24_24_48, EDEMA_bmc_filt_24_24_48, 24, total_obs_e24, total_obs_bmc_e24)

    print('EDEMA-24 (48-96h)')
    calc_metrics(EDEMA_mgb_24_48_96, EDEMA_mgb_filt_24_48_96, EDEMA_bmc_24_48_96, EDEMA_bmc_filt_24_48_96, 24, total_obs_e24, total_obs_bmc_e24)

    print('EDEMA-24 (96h+)')
    calc_metrics(EDEMA_mgb_24_96, EDEMA_mgb_filt_24_96, EDEMA_bmc_24_96, EDEMA_bmc_filt_24_96, 24, total_obs_e24, total_obs_bmc_e24)

    print('EDEMA-24 (Other Insurance)')
    calc_metrics(EDEMA_mgb_24_other, EDEMA_mgb_filt_24_other, EDEMA_bmc_24_other, EDEMA_bmc_filt_24_other, 24)

    print('EDEMA-24 (Medicaid)')
    calc_metrics(EDEMA_mgb_24_medicaid, EDEMA_mgb_filt_24_medicaid, EDEMA_bmc_24_medicaid, EDEMA_bmc_filt_24_medicaid, 24)

    print('EDEMA-24 (Uninsured)')
    calc_metrics(EDEMA_mgb_24_uninsured, EDEMA_mgb_filt_24_uninsured, EDEMA_bmc_24_uninsured, EDEMA_bmc_filt_24_uninsured, 24)




    print ('HELMT-24 (Overall)')
    calc_metrics(HELMET_mgb_24, HELMET_mgb_filt_24, HELMET_bmc_24, HELMET_bmc_filt_24, 24, total_obs_h24, total_obs_bmc_h24)

    print ('HELMT-24 (<24h)')
    calc_metrics(HELMET_mgb_24_24, HELMET_mgb_filt_24_24, HELMET_bmc_24_24, HELMET_bmc_filt_24_24, 24, total_obs_h24, total_obs_bmc_h24)

    print ('HELMT-24 (24-48h)')
    calc_metrics(HELMET_mgb_24_24_48, HELMET_mgb_filt_24_24_48, HELMET_bmc_24_24_48, HELMET_bmc_filt_24_24_48, 24, total_obs_h24, total_obs_bmc_h24)

    print ('HELMT-24 (48-96h)')
    calc_metrics(HELMET_mgb_24_48_96, HELMET_mgb_filt_24_48_96, HELMET_bmc_24_48_96, HELMET_bmc_filt_24_48_96, 24, total_obs_h24, total_obs_bmc_h24)

    print ('HELMT-24 (96h+)')
    calc_metrics(HELMET_mgb_24_96, HELMET_mgb_filt_24_96, HELMET_bmc_24_96, HELMET_bmc_filt_24_96, 24, total_obs_h24, total_obs_bmc_h24)

    print ('HELMT-24 (Other Insurance)')
    calc_metrics(HELMET_mgb_24_other, HELMET_mgb_filt_24_other, HELMET_bmc_24_other, HELMET_bmc_filt_24_other, 24)

    print ('HELMT-24 (Medicaid)')
    calc_metrics(HELMET_mgb_24_medicaid, HELMET_mgb_filt_24_medicaid, HELMET_bmc_24_medicaid, HELMET_bmc_filt_24_medicaid, 24)

    print ('HELMT-24 (Uninsured)')
    calc_metrics(HELMET_mgb_24_uninsured, HELMET_mgb_filt_24_uninsured, HELMET_bmc_24_uninsured, HELMET_bmc_filt_24_uninsured, 24)




    print('EDEMA-8 (Overall)')
    calc_metrics(EDEMA_mgb_8, EDEMA_mgb_filt_8, EDEMA_bmc_8, EDEMA_bmc_filt_8, 8, total_obs_e8, total_obs_bmc_e8)

    print('EDEMA-8 (<24h)')
    calc_metrics(EDEMA_mgb_8_24, EDEMA_mgb_filt_8_24, EDEMA_bmc_8_24, EDEMA_bmc_filt_8_24, 8, total_obs_e8, total_obs_bmc_e8)

    print('EDEMA-8 (24-48h)')
    calc_metrics(EDEMA_mgb_8_24_48, EDEMA_mgb_filt_8_24_48, EDEMA_bmc_8_24_48, EDEMA_bmc_filt_8_24_48, 8, total_obs_e8, total_obs_bmc_e8)

    print('EDEMA-8 (48-96h)')
    calc_metrics(EDEMA_mgb_8_48_96, EDEMA_mgb_filt_8_48_96, EDEMA_bmc_8_48_96, EDEMA_bmc_filt_8_48_96, 8, total_obs_e8, total_obs_bmc_e8)

    print('EDEMA-8 (96h+)')
    calc_metrics(EDEMA_mgb_8_96, EDEMA_mgb_filt_8_96, EDEMA_bmc_8_96, EDEMA_bmc_filt_8_96, 8, total_obs_e8, total_obs_bmc_e8)

    print('EDEMA-8 (Other Insurance)')
    calc_metrics(EDEMA_mgb_8_other, EDEMA_mgb_filt_8_other, EDEMA_bmc_8_other, EDEMA_bmc_filt_8_other, 8)

    print('EDEMA-8 (Medicaid)')
    calc_metrics(EDEMA_mgb_8_medicaid, EDEMA_mgb_filt_8_medicaid, EDEMA_bmc_8_medicaid, EDEMA_bmc_filt_8_medicaid, 8)

    print('EDEMA-8 (Uninsured)')
    calc_metrics(EDEMA_mgb_8_uninsured, EDEMA_mgb_filt_8_uninsured, EDEMA_bmc_8_uninsured, EDEMA_bmc_filt_8_uninsured, 8)




    print('HELMET-8 (Overall)')
    calc_metrics(HELMET_mgb_8, HELMET_mgb_filt_8, HELMET_bmc_8, HELMET_bmc_filt_8, 8, total_obs_h8, total_obs_bmc_h8)

    print('HELMET-8 (<24h)')
    calc_metrics(HELMET_mgb_8_24, HELMET_mgb_filt_8_24, HELMET_bmc_8_24, HELMET_bmc_filt_8_24, 8, total_obs_h8, total_obs_bmc_h8)

    print('HELMET-8 (24-48h)')
    calc_metrics(HELMET_mgb_8_24_48, HELMET_mgb_filt_8_24_48, HELMET_bmc_8_24_48, HELMET_bmc_filt_8_24_48, 8, total_obs_h8, total_obs_bmc_h8)

    print('HELMET-8 (48-96h)')
    calc_metrics(HELMET_mgb_8_48_96, HELMET_mgb_filt_8_48_96, HELMET_bmc_8_48_96, HELMET_bmc_filt_8_48_96, 8, total_obs_h8, total_obs_bmc_h8)

    print('HELMET-8 (96h+)')
    calc_metrics(HELMET_mgb_8_96, HELMET_mgb_filt_8_96, HELMET_bmc_8_96, HELMET_bmc_filt_8_96, 8, total_obs_h8, total_obs_bmc_h8)

    print('HELMET-8 (Other Insurance)')
    calc_metrics(HELMET_mgb_8_other, HELMET_mgb_filt_8_other, HELMET_bmc_8_other, HELMET_bmc_filt_8_other, 8)

    print('HELMET-8 (Medicaid)')
    calc_metrics(HELMET_mgb_8_medicaid, HELMET_mgb_filt_8_medicaid, HELMET_bmc_8_medicaid, HELMET_bmc_filt_8_medicaid, 8)

    print('HELMET-8 (Uninsured)')
    calc_metrics(HELMET_mgb_8_uninsured, HELMET_mgb_filt_8_uninsured, HELMET_bmc_8_uninsured, HELMET_bmc_filt_8_uninsured, 8)


    print('done')