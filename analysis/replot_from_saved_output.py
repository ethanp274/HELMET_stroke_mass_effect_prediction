import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import wandb
import xgboost
import shap

# Output directories for figures (git-ignored, so absent in a fresh clone).
for _output_dir in ('results/figures/shap',):
    os.makedirs(_output_dir, exist_ok=True)

mpl.rc('font', family = 'Times New Roman')


def create_SHAP_plots(model, data, order, labels, model_name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    shap_values_list = [shap_values[:,:,i] for i in order]

    plt.figure()
    shap.summary_plot(
         shap_values = shap_values_list, 
         features = data, 
         feature_names = labels,
         plot_type="bar", 
         class_names = ['No MLS', '0-3mm', '3-8mm', '8mm or greater'],
         show=False
    )
    plt.xlabel('SHAP Value')
    plt.savefig(f"results/figures/shap/{model_name}_shap_plot.png")
    plt.close()


if __name__ == "__main__":

    # LOAD MODELS
    #initialize wandb run and load json artifacts from chosen models
    run = wandb.init()
    artifact_8 = run.use_artifact('edema_pred_ml/july_8_model_bmc_xgb_LLM_8hr/6nt23vmm_model.json:latest', type='model')
    artifact_dir_8 = artifact_8.download()

    artifact_24 = run.use_artifact('edema_pred_ml/july_5_model_bmc_xgb_LLM_24hr/z3hfbt1e_model.json:latest', type = 'model')
    artifact_dir_24 = artifact_24.download()

    #load XGB classifier models from downloaded json artifacts
    model_8 = xgboost.XGBClassifier()
    model_8.load_model(artifact_dir_8 + '/6nt23vmm_model.json')
    model_8_features = model_8.get_booster().feature_names

    model_24 = xgboost.XGBClassifier()
    model_24.load_model(artifact_dir_24 + '/z3hfbt1e_model.json')
    model_24_features = model_24.get_booster().feature_names

    # LOAD DATA
    OLTV_8 = pd.read_json('data/processed/OLTV_8hr_LLM_features.json')
    OLTV_8.columns = [col.replace('ffill_', '') for col in OLTV_8.columns]
    OLTV_8 = OLTV_8[model_8_features]

    OLTV_24 = pd.read_json('data/processed/OLTV_24hr_LLM_features.json')
    OLTV_24.columns = [col.replace('ffill_', '') for col in OLTV_24.columns]
    OLTV_24 = OLTV_24[model_24_features]

        # define feature names
    model_8_feat_labs = {
        'prev_mls': 'Previous MLS value',
        'size_mls': 'Most recent MLS value',
        'aspects1dt_time_censored': 'Time of first ASPECTS score',
        'firstmlsdt_time_censored': 'Time of first non-zero MLS',
        'LLM_8_3': '8-Hour LLM probability of >8mm MLS',
        'LLM_24_1': '24-Hour LLM probability of 0-3mm MLS',
        'LLM_8_2': '8-Hour LLM probability of 3-8mm MLS',
        'firstmls_time_censored': 'First non-zero MLS value',
        'LLM_8_0': '8-Hour LLM probability of no MLS',
        'mls3dt_time_censored': 'Time of first MLS value >3mm',
        'LLM_8_1': '8-Hour LLM probability of 0-3mm MLS',
        'LLM_36_0': '36-Hour LLM probability of no MLS',
        'rolling_size_mls': 'Max MLS value over past 24 hours',
        'nihss': 'NIHSS score at admission',
        'pres': 'Time of admission',
        'gluc1': 'Blood glucose at admission',
        'rolling_wbc': 'Max white blood cell count over past 24 hours',
        'LLM_36_3': '36-Hour LLM probability of >8mm MLS',
        'LLM_24_0': '24-Hour LLM probability of no MLS',
        'cr1': 'Creatinine at admission'
    }

    model_24_feat_labs = {
        'LLM_24_0': '24-Hour LLM probability of no MLS',
        'LLM_24_1': '24-Hour LLM probability of 0-3mm MLS',
        'firstmls_time_censored': 'First non-zero MLS value',
        'LLM_36_0': '36-Hour LLM probability of no MLS',
        'LLM_24_2': '24-Hour LLM probability of 3-8mm MLS',
        'gluc1': 'Blood glucose at admission',
        'prev_mls': 'Previous MLS value',
        'LLM_8_0': '8-Hour LLM probability of no MLS',
        'LLM_24_3': '24-Hour LLM probability of >8mm MLS',
        'LLM_8_2': '8-Hour LLM probability of 3-8mm MLS',
        'LLM_8_3': '8-Hour LLM probability of >8mm MLS',
        'LLM_36_1': '36-Hour LLM probability of 0-3mm MLS',
        'size_mls': 'Most recent MLS value',
        'nihss': 'NIHSS score at admission',
        'pres': 'Time of admission',
        'map1': 'Mean arterial pressure at admission',
        'dbp1': 'Diastolic blood pressure at admission',
        'rolling_hts23_y': '23% Hypertonic saline in past 24 hours',
        'age_calc': 'Age',
        'bun1': 'Blood urea nitrogen at admission'
    }

    cols_8 = OLTV_8.columns.tolist()
    cols_24 = OLTV_24.columns.tolist()

    for col in cols_8:
        if col in model_8_feat_labs.keys():
            cols_8[cols_8.index(col)] = model_8_feat_labs[col]
    
    for col in cols_24:
        if col in model_24_feat_labs.keys():
            cols_24[cols_24.index(col)] = model_24_feat_labs[col]


    create_SHAP_plots(model_8, OLTV_8, order_8, cols_8, '8hr')
    create_SHAP_plots(model_24, OLTV_24, order_24, cols_24, '24hr')

    print('done')




