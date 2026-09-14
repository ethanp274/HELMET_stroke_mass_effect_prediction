import pandas as pd
import numpy as np
import wandb
import xgboost
import os
from sklearn.preprocessing import StandardScaler
from pandas.api.types import is_numeric_dtype

def load_processed_data(lookahead_hours):
    if lookahead_hours == 24:
        OLTV = pd.read_json(os.getcwd() + "/data/processed/OLTV_24hr_LLM_features.json")
        bmc_OLTV = pd.read_json(os.getcwd() + "/data/processed/bmc_OLTV_24hr_LLM_features.json")
    elif lookahead_hours == 8:
        OLTV = pd.read_json(os.getcwd() + "/data/processed/OLTV_8hr_LLM_features.json")
        bmc_OLTV = pd.read_json(os.getcwd() + "/data/processed/bmc_OLTV_8hr_LLM_features.json")
    else: 
        OLTV = None
        bmc_OLTV = None
        print('Invalid Lookahead Hours')

    OLTV.columns = [col.replace('ffill_', '') for col in OLTV.columns]
    bmc_OLTV.columns = [col.replace('ffill_', '') for col in bmc_OLTV.columns]

    return OLTV, bmc_OLTV

def separate_targets(df):
    if df.columns.__contains__('targets'):
        targets = df['targets'].copy()
        df.drop(columns = ['targets'], inplace = True)
    else: 
        targets = None
        print('DataFrame does not contain targets')
        
    return df, targets

def probas_to_class(probas):
    probas_df = pd.DataFrame({
        '0': probas[0],
        '1': probas[1],
        '2': probas[2],
        '3': probas[3]
    })

    classes = probas_df.idxmax(axis = 1)
    return [int(val) for val in classes]

def target_transform(val):
    if val == 0.0:
        return 0
    elif val > 0.0 and val <= 3.0:
        return 1
    elif val > 3.0 and val <=8.0:
        return 2
    elif val > 8.0:
        return 3
    else:
        return -10
    
def scale_data(df):
    scaler = StandardScaler().fit(df)
    # save the scaler to a dictionary for each column name so we can use it later
    scaler_dict = {}
    for i, col in enumerate(df.columns):
        scaler_dict[col] = [scaler.mean_[i], scaler.scale_[i]]
    df = scaler.transform(df)
    return df

def prepare_data(df):
    #transform targets and prev_mls to target space
    df, targets = separate_targets(df)
    targets = list(map(target_transform, targets))

    cols = df.columns
    df['prev_mls'] = list(map(target_transform, df['prev_mls']))
    df = pd.DataFrame(df, columns = cols)

    #remove text-based cols 
    df.drop(columns = [col for col in df.columns if not is_numeric_dtype(df[col])], inplace = True)
    cols = df.columns

    #apply scaler
    df = scale_data(df)
    df = pd.DataFrame(df, columns = cols)

    #add the targets back
    df['targets'] = targets

    return df


if __name__ == "__main__":
    
    #load pre-processed data
    OLTV_8, bmc_OLTV_8 = load_processed_data(8)
    OLTV_24, bmc_OLTV_24 = load_processed_data(24)

    OLTV_8 = prepare_data(OLTV_8)
    bmc_OLTV_8 = prepare_data(bmc_OLTV_8)
    OLTV_24 = prepare_data(OLTV_24)
    bmc_OLTV_24 = prepare_data(bmc_OLTV_24)

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

    iter_n = 2



    #iterate for desired number of bootstrap samples
    for i in range(iter_n):

        sample_size = 100

        #bootstrap sample datasets
        boot_OLTV_8 = OLTV_8.sample(n = sample_size, replace = True)
        boot_bmc_OLTV_8 = bmc_OLTV_8.sample(n = sample_size, replace = True)
        boot_OLTV_24 = OLTV_24.sample(n = sample_size, replace = True)
        boot_bmc_OLTV_24 = bmc_OLTV_24.sample(n = sample_size, replace = True)

        #extract targets from dataset
        boot_OLTV_8, targets_8 = separate_targets(boot_OLTV_8)
        boot_bmc_OLTV_8, bmc_targets_8 = separate_targets(boot_bmc_OLTV_8)
        boot_OLTV_24, targets_24 = separate_targets(boot_OLTV_24)
        boot_bmc_OLTV_24, bmc_targets_24 = separate_targets(boot_bmc_OLTV_24)

        #remove unused features
        boot_OLTV_8 = boot_OLTV_8[model_8_features]
        boot_bmc_OLTV_8 = boot_bmc_OLTV_8[model_8_features]
        boot_OLTV_24 = boot_OLTV_24[model_24_features]
        boot_bmc_OLTV_24 = boot_bmc_OLTV_24[model_24_features]

        #make class predictions for each patient-hour
        pred_8 = probas_to_class(model_8.predict_proba(boot_OLTV_8))
        bmc_pred_8 = probas_to_class(model_8.predict_proba(boot_bmc_OLTV_8))

        #compare predicted classes to target classes
        print(f'Iteration {i}')
        print(f'MGH accuracy: {sum([pred == true for pred, true in zip(pred_8, targets_8)])/sample_size}')
        print(f'BMC accuracy: {sum([pred == true for pred, true in zip(bmc_pred_8, bmc_targets_8)])/sample_size}')
        print([pred for pred, true in zip(pred_8, targets_8) if pred == true])


