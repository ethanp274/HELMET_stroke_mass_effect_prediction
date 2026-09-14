# Final data processing pipeline for edema prediction using LLM and XGB
# Compiled by Ethan Phillips (Univ of Oxford) on 26 July 2024

#################################################################
# CONFIGURE PACKAGES AND WORKING DIRECTORY
import os
import sys
import pandas as pd
import numpy as np
import re
from pydantic import confloat, validate_call
from datetime import datetime
import dateparser
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

#####################################################################
# DEFINE METHODS NEEDED FOR DATA PROCESSING

# Method to flexibly parse dates
def parse_dates(string):
    string = str(string)
    # check for case of missing value (np.nan)
    if not string == string:
        return string
    
    if "UTC" in string:
        string = string.split("UTC")[0]
    
    if bool(re.search(string, "%Y-%m-%d %H:%M:%S")):
        return datetime.strptime(string, "%Y-%m-%d %H:%M:%S")
    elif bool(re.search(string, "%Y-%m-%d")):
        return datetime.strptime(string, "%Y-%m-%d")
    else:
        return dateparser.parse(string)

# Method to select 'rc_' (clinician-graded) radiology values
def get_rc_columns(df):
    rc_cols = [col for col in df.columns if "rc_" in col]
    non_rc_cols = [col.replace('rc_', '') for col in rc_cols]
    non_rc_cols = [col for col in non_rc_cols if col in df.columns]
    df = df.drop(non_rc_cols, axis = 1)
    df.columns = [col.replace('rc_', '') for col in df.columns]
    return df

# Method to crop radiology texts to standardized sections, regardless of source hospital
def reduce_rad_texts(text):
    text = str(text)
    if text == "nan":
        return np.NaN

    if "FINDINGS:" in text:
        text = text.split("FINDINGS:")[1]
    elif "TECHNIQUE:" in text:
        text = text.split("TECHNIQUE:")[1]
    
    if "Dictated By:" in text:
        text = text.split("Dictated By:")[0]
    
    if "ATTESTATION:" in text:
        text = text.split("ATTESTATION:")[0]
    elif "Electronically Signed by:" in text:
        text = text.split("Electronically Signed by:")[0]
    
    return text

# method to pre-process radiology texts before applying LLM 
def preprocess_rad_text(df, OTV):
    ## restructure data and remove rows with missing text 
    df['ptid'] = df['ptid'].ffill()
    df['ptid'] = df['ptid'].apply(int)
    df = df[pd.notna(df['rad_text'])]

    df = df[['ptid', 'dt', 'rad_text']]

    ## drop patients not in OTV
    otv_ptids = OTV.ptid.unique()
    df = df[[ptid in otv_ptids for ptid in df['ptid']]]

    ## cut down on text sizes & standardize texts across hospitals
    df['rad_text'] = df['rad_text'].apply(reduce_rad_texts)

    ## remove rows with invalid rad_texts 
    df = df[pd.notna(df['rad_text'])]

    ## add lsw to rad text dfs from OTVs
    lsw_df = OTV[['ptid', 'lsw']].copy()
    df = pd.merge(df, lsw_df, on = 'ptid', how = "left")
    df['lsw'] = df['lsw'].apply(parse_dates)

    ## generate dt as hours since lsw in rad text dfs
    df['obsTime'] = df['dt'] - df['lsw']
    df['obsTime'] = df['obsTime'].dt.total_seconds() // 3600
    df.drop(columns = ['dt', 'lsw'], inplace = True)
    df.rename(columns = {'obsTime': 'dt'}, inplace = True)

    return df

# define columns related to outcomes of care (could leak information from future hours to the present epoch)
outcome_cols = [
    'parechymal_dt', 'petechial_dt', 'maxmlsdt', 'homeDischarge', 
    'hemorrhage', 'bleed', 'lvo_location', 'CMO_disch', 'deathByDischarge',
    'lastmls', 'lastmlsdt', 'longTermCareByDischarge', 'ph2_dt', 'maxmls', 
    'maxmlsdt', 'rehabByDischarge', 'deathOrHospice', 'hospiceByDischarge', 
    'DNR_dt', 'CMO_dt', 'DNR_disch', 'Discharge_Date', 'tt_mt'
]

# define method to remove outcome-related columns
def remove_outcomes(df, cols):
    cols = [col for col in cols if col in df.columns]
    df = df.drop(cols, axis = 1)
    df = df.reindex(sorted(df.columns), axis = 1)
    return df

# Method to recode all datetime variables as number of hours since lsw
def recode_time_vars(df, time_NaN_fill):
    times = ['pres'] + [col for col in df.columns if col.endswith('dt')]
    df['lsw'] = pd.to_datetime(df['lsw'])

    for time in times:
        df[time] = pd.to_datetime(df[time])
        df[time] = df[time] - df['lsw']
        df[time] = df[time].dt.total_seconds() // 3600
        df[time] = df[time].fillna(int(time_NaN_fill))
        df[time] = df[time].astype(int)

    df = df.drop('lsw', axis = 1)

    columns = df.columns
    df = df[columns]

    return df

# Method to create new threshold binary variables for various values of mls and pgs w/ datetime counterparts
def process_mls_pgs_values(df, val_type, values, size_col, time_NaN_fill):
    for val in values:
        col = val_type + str(val)
        coldt = col + "dt"
        df[col] = 0
        df[coldt] = 1e10
        df[col] = df.apply(lambda x: 1 if x[size_col] >= val else x[col], axis = 1)
        df[col] = df.groupby('ptid')[col].transform("max")
        df[coldt] = df.apply(lambda x: x['dt'] if x[size_col] >= val else x[coldt], axis = 1)
        df[coldt] = df.groupby('ptid')[coldt].transform("min")
        df[coldt] = df[coldt].replace(1e10, time_NaN_fill)
    return df

# Method to remove non-numeric columns from a dataframe
def process_non_numerics(df, disable_one_hot_encoding = True):
    if disable_one_hot_encoding:
        for col in df.columns:
            if df[col].dtype == "object":
                print(f"dropping: {col}")
                df = df.drop(col, axis = 1)
    else:
        for col in df.columns:
            if df[col].nunique() <= 10 & df[col].nunique() > 3:
                print(f"one hot encoding: {col}")
                df = pd.get_dummies(df, columns = [col], drop_first = True)
            elif df[col].dtype == "object":
                print(f"cannot one hot encode: {col}. dropping.")
                df = df.drop(col, axis = 1)
    
    for col in df.columns: 
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
    
    return df


# Method to censor variables corresponding to future times
def apply_time_censoring(df, time_NaN_fill):
    time_cols = [col for col in df.columns if "dt" in col]
    time_cols = [col for col in time_cols if col != "dt"]
    
    for time_col in time_cols:
        var_col = time_col.replace("_dt", "")
        var_col = var_col.replace("dt", "")

        if var_col in df.columns:
            df.loc[df['dt'] <= df[time_col], [var_col]] = int(time_NaN_fill)
            df[var_col] = df[var_col].replace(0, int(time_NaN_fill))
            df.rename(columns = {var_col: var_col + "_time_censored"}, inplace = True)
        
        df.loc[df['dt'] <= df[time_col], [time_col]] = int(time_NaN_fill)
        df.rename(columns = {time_col: time_col + "_time_censored"}, inplace = True)

    return df 

# Method to fill in any missing hours during a patient's stay in the hospital
def generate_missing_hours(df):
    full_index = pd.MultiIndex(
        levels = [[],[]],
        codes = [[],[]],
        names = ['ptid', 'dt']
    )

    for ptid in df.index.get_level_values('ptid').unique():
        min_hour = df.loc[ptid].index.min()
        max_hour = df.loc[ptid].index.max()
        idx = pd.MultiIndex.from_product([[ptid], range(min_hour, max_hour + 1)], names = ['ptid', 'dt'])
        full_index = full_index.append(idx)
    
    df = df.reindex(full_index).reset_index()
    return df

# Method to add previous mls score as feature 
def add_prev_mls(df):
    prev_values = np.zeros(shape = [df.shape[0], ])
    prev_pt = None
    for i, pt in enumerate(df['ptid']):
        if prev_pt is None or prev_pt != pt:
            prev_values[i] = 0
        else:
            prev_values[i] = df.iloc[i]['size_mls']
        prev_pt = pt
    df['prev_mls'] = prev_values

    return df

# Method to remove data after surgery occurs
def remove_data_after_surgery(df):
    df = df[~((df.dt >= df.surgdt_time_censored) & (df.surgdt_time_censored > 0))]
    return df

# Method to transform raw MLS value to one of four MLS classes
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


##################################################################
# Main data processing method to be called in execution section

@validate_call
def process_data(
    time_NaN_fill: int = -10,
    lookback_hours: int = 24,
    lookahead_hours: int = 24,
    col_na_drop_fraction: confloat(ge=0.0, le=1.0) = 0.4, 
    patient_na_drop_fraction: confloat(ge=0.0, le=1.0) = 0.2,
    pre_LLM: bool = False,
    disable_one_hot_encoding: bool = True,
    multiple_classes: bool = True,
):
    # Load Data
    ## load time-dependent variables (LTVs)
    LTV = pd.read_csv("data/raw/LTV_2.0.csv")
    LTV_date_cols = ['lsw', 'dt']
    bmc_LTV = pd.read_csv("data/raw/LTV_BMC_2.csv")
    bmc_LTV_date_cols = ['dt','LSW', 'first_arrival_dt', 'DISCH_DT', 'startdt']

    ## load static variables (OTVs)
    OTV = pd.read_csv("data/raw/OTV_2.1.csv")
    OTV_date_cols = [
        'lsw', 'pres', 'tpadt', 'mtdt', 
        'first_osmotic', 'aspects1dt', 
        'aspectsFUdt', 'rc_firstmlsdt', 
        'rc_lastmlsdt', 'rc_maxmlsdt', 
        'rc_mls5dt', 'rc_mls7dt', 'rc_pgs4dt', 
        'surgdt', 'rc_ltmedt', 'rc_ltme2dt', 
        'rc_petechial_dt', 'rc_ph2_dt', 'DNR_dt',
        'DNI_dt', 'CMO_dt', 'death30dt', 'Discharge_Date'
    ]
    bmc_OTV = pd.read_csv("data/raw/OTV_BMC_3.csv")
    bmc_OTV_date_cols = [
        'LSW', 'first_arrival_dt', 'DISCH_DT', 
        'startdt', 'DNR_DT', 'CMO_DT', 'nihss_dt', 
        'nddt', 'tpadt', 'mtdt', 'aspects1dt', 
        'rc_mls5dt', 'rc_mls7dt', 'rc_maxmlsdt', 
        'rc_firstmlsdt', 'rc_lastmlsdt', 'rc_pgs4dt', 
        'surgdt', 'sbp1dt', 'dbp1dt', 'map1dt', 
        'weight1dt', 'pulse1dt', 'temp1dt', 'gluc1dt', 
        'wbc1dt', 'hba1cdt', 'cr1dt', 'na1dt', 
        'osm1dt', 'bun1dt', 'first_osmotic', 'rc_petechial_dt', 
        'rc_parenchymal_dt', 'rc_ph2_dt', 'aca1dt', 'acadt'
    ]

    ## load radiology report texts
    rad_texts = pd.read_csv("data/raw/MGH_rad_reports_text.csv")
    rad_text_date_cols = ['reportdt']
    bmc_rad_texts = pd.read_csv("data/raw/BMC_rad_reports_text.csv")
    bmc_rad_text_date_cols = ['lsw', 'presdt', 'imagedt']

    ## bmc files use "NA" instead of blank, so need to correct misisng values
    bmc_LTV = bmc_LTV.replace("NA", np.nan)
    bmc_OTV = bmc_OTV.replace("NA", np.nan)

    # drop blank columns 
    LTV = LTV.dropna(how = "all", axis = 1)
    bmc_LTV = bmc_LTV.dropna(how = "all", axis = 1)

    ## apply method to parse dates on any date columns from datasets
    for col in LTV_date_cols:
        LTV[col] = LTV[col].apply(parse_dates)
    for col in bmc_LTV_date_cols:
        bmc_LTV[col] = bmc_LTV[col].apply(parse_dates)
    for col in OTV_date_cols:
        OTV[col] = OTV[col].apply(parse_dates)
    for col in bmc_OTV_date_cols:
        bmc_OTV[col] = bmc_OTV[col].apply(parse_dates)
    for col in rad_text_date_cols:
        rad_texts[col] = rad_texts[col].apply(parse_dates)
    for col in bmc_rad_text_date_cols:
        bmc_rad_texts[col] = bmc_rad_texts[col].apply(parse_dates)

    # preprocess LTVs and OTVs

    ## add surgeries to bmc LTV
    bmc_LTV['surg'] = np.nan
    events = ['surgdt', 'encounter_id', 'CDW_ID', 'CDW_MRN', 'LSW', 'first_arrival_dt', 'DISCH_DT', 'startdt', 'redcap_id']
    
    surg_events_df = bmc_OTV[bmc_OTV['surg'] == 1][events].rename(columns = {'surgdt': 'dt'})
    surg_events_df["surg"] = 1

    bmc_LTV = pd.concat([bmc_LTV, surg_events_df], ignore_index = True)
    bmc_LTV = bmc_LTV.sort_values(['encounter_id', 'dt']).reset_index(drop = True)

    ## correct for missing or incorrectly named variables in bmc datasets
    bmc_LTV['hts23'] = time_NaN_fill
    bmc_LTV.rename(columns = {
        'glucose': 'gluc', 
        'sodium': 'na',
        'creatinine': 'cr', 
        'osmolality': 'osm', 
        'CDW_MRN': 'ptid', 
        'LSW': 'lsw'
    }, inplace = True)

    bmc_OTV.rename(columns = {
        'CDW_MRN': 'ptid',
        'LSW': 'lsw',
        'CMO': 'CMO_disch',
        'CMO_DT': 'CMO_dt',
        'DNR': 'DNR_disch',
        'DNR_DT': 'DNR_dt',
        'DISCH_DT': 'Discharge_Date',
        'X.24.hours': '<24 hours',
        'first_arrival_dt': 'pres',
    }, inplace = True)

    ## construct firstmls, firstmlsdt in bmc OTV from the LTV data
    nonzero_mls_df = bmc_LTV[bmc_LTV['size_mls'] > 0]
    nonzero_mls_df = nonzero_mls_df.groupby('ptid').first().reset_index()
    nonzero_mls_df = nonzero_mls_df[['ptid', 'size_mls', 'dt']]
    nonzero_mls_df.rename(columns = {'size_mls': 'firstmls', 'dt': 'firstmlsdt'}, inplace = True)

    bmc_OTV = pd.merge(bmc_OTV, nonzero_mls_df, on = 'ptid', how = 'left')
    
    ## construct obsTime (hours between lsw and dt) for bmc LTV
    bmc_LTV['obsTime'] = bmc_LTV['dt'] - bmc_LTV['lsw']
    bmc_LTV['obsTime'] = bmc_LTV['obsTime'].dt.total_seconds() // 3600
    bmc_LTV['obsTime'] = bmc_LTV['obsTime'].apply(int)

    ## use only radiology values graded by clinicians ('rc_' columns)
    LTV = get_rc_columns(LTV)
    bmc_LTV = get_rc_columns(bmc_LTV)
    OTV = get_rc_columns(OTV)
    bmc_OTV = get_rc_columns(bmc_OTV)

    ## drop binary indicators for radiology values
    def mls_pgs_convert(df):
        if 'mls' in df.columns:
            df['size_mls'] = df[['mls', 'size_mls']].max(axis = 1).copy()
            df = df.drop('mls', axis = 1)
        if 'pgs' in df.columns:
            df['size_pgs'] = df[['pgs', 'size_pgs']].max(axis = 1).copy()
            df = df.drop('pgs', axis = 1)
        return df
    
    LTV = mls_pgs_convert(LTV)
    bmc_LTV = mls_pgs_convert(bmc_LTV)

    ## round obsTime to hours for mgh LTV
    LTV['obsTime'] = [np.round(val, 0) for val in LTV['obsTime']]

    ## fill missing values (NaN) with set value
    LTV.iloc[:, 4:-1] = LTV.iloc[:, 4:-1].fillna(time_NaN_fill)
    bmc_LTV.iloc[:, 2:-1] = bmc_LTV.iloc[:, 2:-1].fillna(time_NaN_fill)

    ## merge LTV and OTV files 
    OLTV = pd.merge(LTV, OTV, on = "ptid", how = "inner")
    bmc_OLTV = pd.merge(bmc_LTV, bmc_OTV, on = "ptid", how = "inner")

    ## delete LTV files
    del LTV, bmc_LTV

    ## rename incorrect column
    OLTV.rename(columns = {"first_osmotic": "first_osmotic_dt"}, inplace = True)
    
    ## correct for column issues after merging OTVs and LTVs
    def fix_cols_post_merge(df):
        df.rename(columns = {'lsw_x': 'lsw', 'surg_y': 'surg'}, inplace = True)
        df.drop(['lsw_y', 'surg_x'], axis = 1, inplace = True)
        return df
    
    OLTV = fix_cols_post_merge(OLTV)
    bmc_OLTV = fix_cols_post_merge(bmc_OLTV)

    ## keep only columns found in both datasets
    columns = list(set(OLTV.columns).intersection(set(bmc_OLTV.columns)))
    OLTV = OLTV[columns]
    bmc_OLTV = bmc_OLTV[columns]

    # Process OLTVs to improve ML performance

    ## remove columns related to outcome of care
    OLTV = remove_outcomes(OLTV, outcome_cols)
    bmc_OLTV = remove_outcomes(bmc_OLTV, outcome_cols)

    ## Remove patients with stays shorter than 24 hours
    print(f"Num MGB patients before dropping short stays: {len(OLTV.ptid.unique())}")
    print(f"Num BMC patients before dropping short stays: {len(bmc_OLTV.ptid.unique())}")

    def drop_short_stays(df):
        df = df[df['<24 hours'] == 1]
        df = df.drop('<24 hours', axis = 1)
        return df
    
    OLTV = drop_short_stays(OLTV)
    bmc_OLTV = drop_short_stays(bmc_OLTV)

    print(f"Num MGB patients after dropping short stays: {len(OLTV.ptid.unique())}")
    print(f"Num BMC patients after dropping short stays: {len(bmc_OLTV.ptid.unique())}")

    
    ## Recode hispanic as race and drop ethnicity column
    def recode_ethnicity(df):
        df['race'] = [ethnic if ethnic == "Hispanic" else race for ethnic, race in zip(df['ethnicity'], df['race'])]
        df.drop(columns = ['ethnicity'], inplace = True)
        return df 
    
    OLTV = recode_ethnicity(OLTV)
    bmc_OLTV = recode_ethnicity(bmc_OLTV)

    ## drop any missing columns
    OLTV = OLTV.dropna(how = "all", axis = 1)
    bmc_OLTV = bmc_OLTV.dropna(how = "all", axis = 1)

    # Drop ph1, ph2, and tt columns if present
    def drop_ph_and_tt_cols(df):
        df = df.drop([col for col in df.columns if "ph1" in col or "ph2" in col or "tt" in col], axis = 1)
        return df
    
    OLTV = drop_ph_and_tt_cols(OLTV)
    bmc_OLTV = drop_ph_and_tt_cols(bmc_OLTV)

    ## base all times off of hours since lsw 
    OLTV = recode_time_vars(OLTV, time_NaN_fill = time_NaN_fill)
    bmc_OLTV = recode_time_vars(bmc_OLTV, time_NaN_fill = time_NaN_fill)

    ## Create new variables which capture the presence and dt of various mls and pgs values
    ### apply method for mls 
    OLTV = process_mls_pgs_values(OLTV, "mls", [3, 6, 9, 12, 15], 'size_mls', time_NaN_fill=time_NaN_fill)
    bmc_OLTV = process_mls_pgs_values(bmc_OLTV, "mls", [3, 6, 9, 12, 15], 'size_mls', time_NaN_fill=time_NaN_fill)

    ### apply method for pgs
    OLTV = process_mls_pgs_values(OLTV, "pgs", [2, 4, 6, 8, 10], 'size_pgs', time_NaN_fill=time_NaN_fill)
    bmc_OLTV = process_mls_pgs_values(bmc_OLTV, "pgs", [2, 4, 6, 8, 10], 'size_pgs', time_NaN_fill=time_NaN_fill)

    ## Drop or one hot encode any non-numeric cols
    OLTV = process_non_numerics(OLTV, disable_one_hot_encoding= disable_one_hot_encoding)
    bmc_OLTV = process_non_numerics(bmc_OLTV, disable_one_hot_encoding= disable_one_hot_encoding)

    ## Censor variables which correspond to times in the future
    OLTV = apply_time_censoring(OLTV, time_NaN_fill = time_NaN_fill)
    bmc_OLTV = apply_time_censoring(bmc_OLTV, time_NaN_fill = time_NaN_fill)

    ## group by ptid and hour
    OLTV = OLTV.groupby(['ptid', 'dt']).max().reset_index()
    OLTV = OLTV.set_index(['ptid', 'dt'])

    bmc_OLTV = bmc_OLTV.groupby(['ptid', 'dt']).max().reset_index()
    bmc_OLTV = bmc_OLTV.set_index(['ptid', 'dt'])

    ## reindex to include all hours of patient stay
    OLTV = generate_missing_hours(OLTV)
    bmc_OLTV = generate_missing_hours(bmc_OLTV)

    # Process and apply LLM to radiology report texts (except when in pre_LLM mode)
    
    ## rename columns to standardize
    rad_texts.rename(columns = {
        'reportdt': 'dt',
        'rad_report_text': 'rad_text'
    }, inplace = True)

    bmc_rad_texts.rename(columns = {
        'mrn': 'ptid',
        'imagedt': 'dt',
        'rad_rep': 'rad_text'
    }, inplace = True)
    
    ## apply text preprocessing
    rad_texts = preprocess_rad_text(rad_texts, OTV)
    bmc_rad_texts = preprocess_rad_text(bmc_rad_texts, bmc_OTV)

    if not pre_LLM:  # if LLMs have already been fine tuned
        ## Convert rad_text to mls class probabilities
        ### Load tokenizer and fine-tuned LLMs (8hr, 24hr, and 36hr) --- Note: LLM models will not load unless given access by authors through HuggingFace
        tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
        
        LLM_8hr = AutoModelForSequenceClassification.from_pretrained("ethanp5/edema_prediction_LLM_8hr")
        LLM_24hr = AutoModelForSequenceClassification.from_pretrained("ethanp5/edema_prediction_LLM_24hr")
        LLM_36hr = AutoModelForSequenceClassification.from_pretrained("ethanp5/edema_prediction_LLM_36hr")
        
        pipe8 = pipeline("text-classification", model = LLM_8hr, tokenizer = tokenizer, top_k = None)
        pipe24 = pipeline("text-classification", model = LLM_24hr, tokenizer = tokenizer, top_k = None)
        pipe36 = pipeline("text-classification", model = LLM_36hr, tokenizer = tokenizer, top_k = None)

        ### generate LLM predictions per class (x4) per window length (x3) = 12 new variables
        def generate_LLM_probas(df):
            texts = [str(element) for element in df['rad_text'].copy()]

            def get_probas_from_LLM(texts, pipeline):
                output_dicts = pipeline(texts)
                probas = [[d['score'] for d in dict] for dict in output_dicts]
                probas_df = pd.DataFrame(probas, columns = ['class0', 'class1', 'class2', 'class3'])
                return probas_df
            
            LLM8_probas = get_probas_from_LLM(texts, pipe8)
            LLM24_probas = get_probas_from_LLM(texts, pipe24)
            LLM36_probas = get_probas_from_LLM(texts, pipe36)

            for i in range(4):
                df[f"LLM8_{i}"] = LLM8_probas[LLM8_probas.columns[i]]
                df[f"LLM24_{i}"] = LLM24_probas[LLM24_probas.columns[i]]
                df[f"LLM36_{i}"] = LLM36_probas[LLM36_probas.columns[i]]
            
            return df
            
        ## apply LLM to get class probas
        rad_texts = generate_LLM_probas(rad_texts)
        bmc_rad_texts = generate_LLM_probas(bmc_rad_texts)

        ## drop raw texts
        if 'rad_text' in rad_texts.columns:
            rad_texts.drop(columns = ['rad_text'], inplace = True)
        
        if 'rad_text' in bmc_rad_texts.columns:
            bmc_rad_texts.drop(columns = ['rad_text'], inplace = True)

    ## join LLM predictions (or raw texts if in pre_LLM mode) to OLTVs
    OLTV_wLLM = pd.merge(OLTV, rad_texts, on = ['ptid', 'dt'], how = 'left')
    bmc_OLTV_wLLM = pd.merge(bmc_OLTV, bmc_rad_texts, on = ['ptid', 'dt'], how = 'left')

    # create prediction epoch target MLS values
    def create_targets(df):
        targets = (df.groupby(['ptid'])['size_mls'].shift(-lookahead_hours).rolling(lookahead_hours, min_periods = 1).max())
        return targets
    
    OLTV_targets = create_targets(OLTV_wLLM)
    bmc_OLTV_targets = create_targets(bmc_OLTV_wLLM)

    # generate rolling maximum columns
    def create_rolling_cols(OLTV, OTV):
        #dont apply rolling to OTV (static) features
        rolling_cols = [col for col in OLTV.columns if col not in OTV.columns]
        
        #dont apply rolling to LLM probas or rad text
        rolling_cols = [col for col in rolling_cols if not col.startswith('LLM')]
        rolling_cols = [col for col in rolling_cols if not OLTV[col].dtype == "object"]

        
        #add ptid and dt
        if 'ptid' not in rolling_cols:
            rolling_cols.append('ptid')
        if 'dt' not in rolling_cols:
            rolling_cols.append('dt')    


        rolling_df = OLTV[rolling_cols].copy()

        rolling_df = rolling_df.groupby('ptid').apply(lambda x: x.rolling(lookback_hours, min_periods = 1).max())

        rolling_df.columns = ["rolling_" + col for col in rolling_df.columns]
        rolling_df.rename(columns = {'rolling_dt': 'dt', 'rolling_ptid': 'ptid'}, inplace = True)
        rolling_df = rolling_df.drop('ptid', axis = 1).reset_index()
        rolling_df = rolling_df.drop('level_1', axis = 1)

        return rolling_df
    
    OLTV_rolling = create_rolling_cols(OLTV = OLTV_wLLM, OTV = OTV)
    bmc_OLTV_rolling = create_rolling_cols(OLTV = bmc_OLTV_wLLM, OTV = bmc_OTV)
    
    # Forward fill any missing values in the OLTV (non rolling columns)
    OLTV_ffill = OLTV_wLLM.copy()
    OLTV_ffill = OLTV_ffill.replace(int(time_NaN_fill), np.nan).groupby('ptid').ffill()
    OLTV_ffill = OLTV_ffill.fillna(int(time_NaN_fill))

    bmc_OLTV_ffill = bmc_OLTV_wLLM.copy()
    bmc_OLTV_ffill = bmc_OLTV_ffill.replace(int(time_NaN_fill), np.nan).groupby('ptid').ffill()
    bmc_OLTV_ffill = bmc_OLTV_ffill.fillna(int(time_NaN_fill))

    OLTV_ffill = OLTV_ffill.drop(columns = ['dt'])
    bmc_OLTV_ffill = bmc_OLTV_ffill.drop(columns = ['dt'])

    # concatenate rolling cols and ffill cols
    OLTV_final = pd.concat([OLTV_rolling, OLTV_ffill], axis = 1)
    bmc_OLTV_final = pd.concat([bmc_OLTV_rolling, bmc_OLTV_ffill], axis = 1)

    # add targets to OLTVs
    OLTV_final['target'] = OLTV_targets
    bmc_OLTV_final['target'] = bmc_OLTV_targets

    # Post processing of OLTVs
    ## drop rows with missing targets
    OLTV_final = OLTV_final.dropna(subset = ['target'])
    bmc_OLTV_final = bmc_OLTV_final.dropna(subset = ['target'])

    OLTV_final = OLTV_final[OLTV_final.target != time_NaN_fill]
    bmc_OLTV_final = bmc_OLTV_final[bmc_OLTV_final.target != time_NaN_fill]

    ## Drop data after surgery occurs
    OLTV_final = remove_data_after_surgery(OLTV_final)
    bmc_OLTV_final = remove_data_after_surgery(bmc_OLTV_final)

    ## drop any rows that are the missing in all values 
    OLTV_final = OLTV_final.dropna(how = "all", subset = [col for col in OLTV_final.columns if col != 'target'])
    bmc_OLTV_final = bmc_OLTV_final.dropna(how = "all", subset = [col for col in bmc_OLTV_final.columns if col != 'target'])

    ## drop any cols that are duplicates in all values
    OLTV_final = OLTV_final.drop_duplicates(subset = [col for col in OLTV_final.columns if col not in ['dt', 'obsTime']], keep = "last")                             
    bmc_OLTV_final = bmc_OLTV_final.drop_duplicates(subset = [col for col in bmc_OLTV_final.columns if col not in ['dt', 'obsTime']], keep = "last")

    ## reset indices
    OLTV_final = OLTV_final.reset_index(drop = True)
    bmc_OLTV_final = bmc_OLTV_final.reset_index(drop = True)

    ## keep only columns found in both datasets
    columns = list(set(OLTV.columns).intersection(set(bmc_OLTV.columns)))
    OLTV = OLTV[columns]
    bmc_OLTV = bmc_OLTV[columns]

    # recode times in OTVs
    times = ['pres'] + [col for col in columns if col.endswith('dt')]
    bmc_times = ['pres'] + [col for col in columns if col.endswith('dt')]

    for col in OTV.columns:
        if col.endswith('dt'):
            times.append(col)
    
    for col in bmc_OTV.columns:
        if col.endswith('dt'):
            bmc_times.append(col)
    
    def recode_times_OTV(OTV, times, time_NaN_fill):
        OTV['lsw'] = pd.to_datetime(OTV['lsw'])
        for time in times:
            if time in OTV.columns:
                OTV[time] = pd.to_datetime(OTV[time])
                OTV[time] = (OTV[time] - OTV['lsw']).dt.total_seconds() // 3600
                OTV[time] = OTV[time].fillna(int(time_NaN_fill))
        
        return OTV
    
    OTV = recode_times_OTV(OTV, times, time_NaN_fill)
    bmc_OTV = recode_times_OTV(bmc_OTV, bmc_times, time_NaN_fill)

    # add previous mls value as column
    OLTV_final = add_prev_mls(OLTV_final)
    bmc_OLTV_final = add_prev_mls(bmc_OLTV_final)

    # drop rows exceeding a percentage of missing variables 
    num_rows = OLTV_final.shape[0]
    OLTV_final = OLTV_final.dropna(axis = 0, thresh = OLTV_final.shape[1] * patient_na_drop_fraction)
    print(f"Number of rows dropped from MGB: {num_rows - OLTV_final.shape[0]}")

    bmc_num_rows = bmc_OLTV_final.shape[0]
    bmc_OLTV_final = bmc_OLTV_final.dropna(axis = 0, thresh = bmc_OLTV_final.shape[1] * patient_na_drop_fraction)
    print(f"Number of rows dropped from BMC: {bmc_num_rows - bmc_OLTV_final.shape[0]}")

    # drop cols exceeding a percentage of missing patients
    num_cols = OLTV_final.shape[1]
    bmc_num_cols = bmc_OLTV_final.shape[1]

    cols = OLTV_final.columns
    bmc_cols = bmc_OLTV_final.columns

    ## drop cols exceeding na threshold
    OLTV_final = OLTV_final.dropna(axis = 1, thresh = OLTV_final.shape[0] * col_na_drop_fraction)
    bmc_OLTV_final = bmc_OLTV_final.dropna(axis = 1, thresh = bmc_OLTV_final.shape[0] * col_na_drop_fraction)

    ## report dropped cols
    print(f"Number of columns dropped from MGB: {num_cols - OLTV_final.shape[1]}")
    print(f"Columns dropped from MGB: {[col for col in cols if col not in OLTV_final.columns]}")

    print(f"Number of columns dropped from BMC: {bmc_num_cols - bmc_OLTV_final.shape[1]}")
    print(f"Columns dropped from BMC: {[col for col in bmc_cols if col not in bmc_OLTV_final.columns]}")

    ## make columns match between sets
    columns = list(set(OLTV_final.columns).intersection(set(bmc_OLTV_final.columns)))
    OLTV_final = OLTV_final[columns]
    bmc_OLTV_final = bmc_OLTV_final[columns]

    # reset index
    OLTV_final = OLTV_final.reset_index(drop = True)
    bmc_OLTV_final = bmc_OLTV_final.reset_index(drop = True)

    # convert target mls values to classes
    def target_transform_norm(val):
        return target_transform_partial(val, time_NaN_fill=time_NaN_fill)
    
    def target_transform_basic(val):
        if x < 5:
            return 0
        elif x >= 5:
            return 1
        else:
            return time_NaN_fill
    
    if multiple_classes:
        OLTV_final['target'] = OLTV_final['target'].apply(target_transform_norm)
        bmc_OLTV_final['target'] = bmc_OLTV_final['target'].apply(target_transform_norm)
    else:
        OLTV_final['target'] = OLTV_final['target'].apply(target_transform_basic)
        bmc_OLTV_final['target'] = bmc_OLTV_final['target'].apply(target_transform_basic)

    # separate targets
    OLTV_targets = OLTV_final['target'].copy()
    OLTV_final.drop(columns = ['target'], inplace = True)

    bmc_OLTV_targets = bmc_OLTV_final['target'].copy()
    bmc_OLTV_final.drop(columns = ['target'], inplace = True)

    return OLTV_final, OTV, OLTV_targets, bmc_OLTV_final, bmc_OTV, bmc_OLTV_targets

##############################################################
# EXECUTE DATA PROCESSING
if __name__ == "__main__":
    pre_LLM = False # set to false if LLMs are trained and accessible
    multi_class = False

    OLTV_24, OTV_24, targets_24, bmc_OLTV_24, bmc_OTV_24, bmc_targets_24= process_data(pre_LLM = pre_LLM, lookahead_hours=24, multiple_classes=multi_class)
    OLTV_8, OTV_8, targets_8, bmc_OLTV_8, bmc_OTV_8, bmc_targets_8= process_data(pre_LLM = pre_LLM, lookahead_hours=8, multiple_classes=multi_class) # only targets are different

    if multi_class:
        if pre_LLM:
            OLTV_24.to_json("data/processed/processed_OLTV_preLLM.json")
            bmc_OLTV_24.to_json("data/processed/processed_bmc_OLTV_preLLM.json")
        
        else:
            OLTV_24.to_json("data/processed/processed_OLTV_final_.json")
            bmc_OLTV_24.to_json("data/processed/processed_bmc_OLTV_final.json")

        OTV_24.to_json("data/processed/processed_OTV.json")
        pd.DataFrame(targets_24, columns = ['target']).to_json("data/processed/processed_targets_24hr.json")
        pd.DataFrame(targets_8, columns = ['target']).to_json("data/processed/processed_targets_8hr.json")

        bmc_OTV_24.to_json("data/processed/processed_bmc_OTV.json")
        pd.DataFrame(bmc_targets_24, columns = ['target']).to_json("data/processed/processed_bmc_24hr_targets.json")
        pd.DataFrame(bmc_targets_8, columns = ['target']).to_json("data/processed/processed_bmc_8hr_targets.json")
    else:
        if pre_LLM:
            OLTV_24.to_json("data/processed/processed_OLTV_preLLM_basic.json")
            bmc_OLTV_24.to_json("data/processed/processed_bmc_OLTV_preLLM_basic.json")
        else:
            OLTV_24.to_json("data/processed/processed_OLTV_final_basic.json")
            bmc_OLTV_24.to_json("data/processed/processed_bmc_OLTV_final_basic.json")

        OTV_24.to_json("data/processed/processed_OTV.json")
        pd.DataFrame(targets_24, columns = ['target']).to_json("data/processed/processed_targets_24hr_basic.json")
        pd.DataFrame(targets_8, columns = ['target']).to_json("data/processed/processed_targets_8hr_basic.json")

        bmc_OTV_24.to_json("data/processed/processed_bmc_OTV.json")
        pd.DataFrame(bmc_targets_24, columns = ['target']).to_json("data/processed/processed_bmc_24hr_targets_basic.json")
        pd.DataFrame(bmc_targets_8, columns = ['target']).to_json("data/processed/processed_bmc_8hr_targets_basic.json")


    print('processed data saved')

    if pre_LLM:
        print('Data ready for LLM training. \n After training is complete, run again with pre_LLM = False to add LLM features to data.')

