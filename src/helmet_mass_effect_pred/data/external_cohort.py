# %%
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from pandas import RangeIndex
from pandas.api.indexers import FixedForwardWindowIndexer as FFWI
from pydantic import validate_arguments
from typing import Literal
from joblib import Memory

# Root for the private clinical data. Override with the MASS_EFFECT_DATA env var;
# see docs/data.md for the expected layout.
DATA_ROOT = os.environ.get("MASS_EFFECT_DATA", "data")


memory = Memory(location="./cache", verbose=0)


# Transform target columns
@validate_arguments
#@memory.cache
def create_MGH_and_BMC(
    filter_out_less_than_24_hours=True,
    disable_one_hot_encoding=True,
    time_NaN_fill: int = -10,
    use_only_rc_values=True,
    lookback_hours=24,
    lookahead_hours=24,
    use_lsw_as_base_time_instead_of_pres=True,
    create_new_mls_and_pgs_vals=True,
    use_ffil_and_rolling=True,
    dont_transform_otv=False,
    panos_mode=False,
    add_min_values=False,
    use_mode_as_feature: bool = False,
    use_prev_as_feature: bool = False,
    preserve_true_scans: bool = False,

):
    dt_parser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")  # noqa: E731

    def read_csv_file(filename):
        if panos_mode:
            path = os.path.join(DATA_ROOT, "censored", filename)
        else:
            path = os.path.join(DATA_ROOT, "raw", filename)
        return pd.read_csv(path, parse_dates=True, date_parser=dt_parser)

    LTV = read_csv_file("LTV_2.0.csv")
    bmc_LTV = read_csv_file("LTV_BMC_2.csv")
    OTV = read_csv_file("OTV_2.1.csv")
    bmc_OTV = read_csv_file("OTV_BMC_3.csv")

    LTV = LTV.dropna(how="all", axis=1)
    bmc_LTV = bmc_LTV.dropna(how="all", axis=1)

    bmc_LTV = bmc_LTV.replace("NA", np.nan)
    bmc_OTV = bmc_OTV.replace("NA", np.nan)

    bmc_LTV["surg"] = np.nan

    events = ["surgdt", "encounter_id", "CDW_ID", "CDW_MRN"]
    events += ["LSW", "first_arrival_dt", "DISCH_DT", "startdt", "redcap_id"]

    surg_events = bmc_OTV[bmc_OTV["surg"] == 1][events].rename(columns={"surgdt": "dt"})
    surg_events["surg"] = 1

    bmc_LTV = pd.concat([bmc_LTV, surg_events], ignore_index=True)
    # bmc_LTV = bmc_LTV.append(surg_events, ignore_index=True)
    bmc_LTV = bmc_LTV.sort_values(["encounter_id", "dt"]).reset_index(drop=True)

    bmc_LTV["hts23"] = time_NaN_fill
    bmc_LTV.rename(columns={"glucose": "gluc", "sodium": "na"}, inplace=True)
    bmc_LTV.rename(columns={"creatinine": "cr", "osmolality": "osm"}, inplace=True)
    bmc_LTV.rename(columns={"CDW_MRN": "ptid", "LSW": "lsw"}, inplace=True)
    
    if not use_only_rc_values:
        bmc_LTV.rename(columns={"size_max_pgs": "size_pgs"}, inplace=True)

    bmc_OTV.rename(columns={"CDW_MRN": "ptid", "LSW": "lsw"}, inplace=True)
    bmc_OTV.rename(columns={"CMO": "CMO_disch", "CMO_DT": "CMO_dt"}, inplace=True)
    bmc_OTV.rename(columns={"DNR": "DNR_disch", "DNR_DT": "DNR_dt"}, inplace=True)
    bmc_OTV.rename(columns={"DISCH_DT": "Discharge_Date"}, inplace=True)
    bmc_OTV.rename(columns={"X.24.hours": "<24 hours"}, inplace=True)
    bmc_OTV.rename(columns={"first_arrival_dt": "pres"}, inplace=True)

    # need to construct firstmls, firstmlsdt in the bmc_OTV from the bmc_LTV data

    nonzero_mls = bmc_LTV[bmc_LTV["size_mls"] > 0]
    nonzero_mls = nonzero_mls.groupby("ptid").first().reset_index()
    nonzero_mls = nonzero_mls[["ptid", "size_mls", "dt"]]
    nonzero_mls.rename(columns={"size_mls": "firstmls"}, inplace=True)
    nonzero_mls.rename(columns={"dt": "firstmlsdt"}, inplace=True)

    bmc_OTV = pd.merge(bmc_OTV, nonzero_mls, on="ptid", how="left")

    bmc_LTV["lsw"] = pd.to_datetime(bmc_LTV["lsw"])
    bmc_LTV["dt"] = pd.to_datetime(bmc_LTV["dt"])
    bmc_LTV["obsTime"] = bmc_LTV["dt"] - bmc_LTV["lsw"]
    bmc_LTV["obsTime"] = bmc_LTV["obsTime"].dt.total_seconds() // 3600

    def get_rc_columns(df):
        rc_columns = [column for column in df.columns if "rc_" in column]
        non_rc_versions = [column.replace("rc_", "") for column in rc_columns]
        non_rc_versions = [col for col in non_rc_versions if col in df.columns]
        df = df.drop(non_rc_versions, axis=1)
        df.columns = [column.replace("rc_", "") for column in df.columns]
        return df

    def mls_pgs_convert(df):
        if "mls" in df.columns:
            df["size_mls"] = df[["mls", "size_mls"]].max(axis=1).copy()
            df = df.drop("mls", axis=1)
        if "pgs" in df.columns:
            df["size_pgs"] = df[["pgs", "size_pgs"]].max(axis=1).copy()
            df = df.drop("pgs", axis=1)
        return df

    if use_only_rc_values:
        LTV = get_rc_columns(LTV)
        bmc_LTV = get_rc_columns(bmc_LTV)
        OTV = get_rc_columns(OTV)
        bmc_OTV = get_rc_columns(bmc_OTV)

    LTV = mls_pgs_convert(LTV)
    bmc_LTV = mls_pgs_convert(bmc_LTV)

    # blindly filling all NaNs with time_NaN_fill is a bad idea
    LTV.iloc[:, 4:-1] = LTV.iloc[:, 4:-1].fillna(time_NaN_fill)
    bmc_LTV = bmc_LTV.fillna(time_NaN_fill)

    OLTV = pd.merge(LTV, OTV, on="ptid", how="inner")
    bmc_OLTV = pd.merge(bmc_LTV, bmc_OTV, on="ptid", how="inner")
    OLTV.rename(columns={"first_osmotic": "first_osmotic_dt"}, inplace=True)

    del LTV, bmc_LTV

    # keep only the interstion of bmc_OLTV and OLTV columns

    columns = list(set(OLTV.columns).intersection(set(bmc_OLTV.columns)))
    OLTV = OLTV[columns]
    bmc_OLTV = bmc_OLTV[columns]

    base_time, secondary_time = (
        ("lsw", "pres") if use_lsw_as_base_time_instead_of_pres else ("pres", "lsw")
    )

    columns = OLTV.columns
    times = [secondary_time] + [column for column in columns if column.endswith("dt")]

    bmc_columns = bmc_OLTV.columns
    bmc_times = [secondary_time] + [
        column for column in bmc_columns if column.endswith("dt")
    ]

    def generate_OLTV(DF, preserved_otv_columns):
        DF.rename(columns={"lsw_x": "lsw"}, inplace=True)
        DF.drop("lsw_y", axis=1, inplace=True)
        DF.drop("surg_x", axis=1, inplace=True)
        preserved_otv_columns.append("surg_y")
        preserved_otv_columns.append("lsw")

        outcome_cols = ["parenchymal_dt", "petechial_dt", "maxmlsdt"]
        outcome_cols += ["homeDischarge", "hemorrhage", "bleed"]
        outcome_cols += ["lvo_location", "CMO_disch", "deathByDischarge", "lastmls"]
        outcome_cols += ["longTermCareByDischarge", "lastmlsdt", "ph2_dt", "maxmls"]
        outcome_cols += ["rehabByDischarge", "deathOrHospice", "hospiceByDischarge"]
        outcome_cols += ["DNR_dt", "CMO_dt", "DNR_disch", "Discharge_Date", "tt_mt"]

        DF = DF.drop(outcome_cols, axis=1)
        DF = DF.reindex(sorted(DF.columns), axis=1)
        columns = DF.columns
        times = [secondary_time] + [
            column for column in columns if column.endswith("dt")
        ]
        # operation below removes about 200 patients of data
        DF = DF[DF["<24 hours"] == 1] if filter_out_less_than_24_hours else DF

        DF[DF.ethnicity == "Hispanic"]["Race"] = "Hispanic"
        DF = DF.drop(["ethnicity", "<24 hours"], axis=1)
        DF = DF.dropna(how="all", axis=1)
        DF = DF.drop([column for column in DF.columns if "ph2" in column], axis=1)
        DF = DF.drop([column for column in DF.columns if "ph1" in column], axis=1)
        columns = [column for column in DF.columns if "tt_" not in column]

        # time conversion
        DF[base_time] = pd.to_datetime(DF[base_time])
        for time in times:
            DF[time] = pd.to_datetime(DF[time])
            DF[time] = DF[time] - DF[base_time]
            DF[time] = DF[time].dt.total_seconds() // 3600
            DF[time] = DF[time].fillna(int(time_NaN_fill))
        DF = DF.drop(base_time, axis=1)
        DF["dt"] = DF["dt"].astype(int)
        columns = DF.columns
        DF = DF[columns]

        def process_values(DF, prefix, values, size_column, time_NaN_fill):
            for val in values:
                col = prefix + str(val)
                coldt = col + "dt"
                DF[col] = 0
                DF[coldt] = 1e10
                DF[col] = DF.apply(
                    lambda x: 1 if x[size_column] >= val else x[col], axis=1
                )
                DF[col] = DF.groupby("ptid")[col].transform("max")
                DF[coldt] = DF.apply(
                    lambda x: x["dt"] if x[size_column] >= val else x[coldt], axis=1
                )
                DF[coldt] = DF.groupby("ptid")[coldt].transform("min")
                DF[coldt] = DF[coldt].replace(1e10, time_NaN_fill)
            return DF

        if create_new_mls_and_pgs_vals:
            DF = process_values(DF, "mls", [3, 9, 12, 15], "size_mls", time_NaN_fill)
            DF = process_values(DF, "pgs", [2, 4, 6, 8, 10], "size_pgs", time_NaN_fill)

            mls_cols = [f"mls{i}" for i in [3, 9, 12, 15]]
            mls_dt_cols = [f"mls{i}dt" for i in [3, 9, 12, 15]]
            pgs_cols = [f"pgs{i}" for i in [2, 4, 6, 8, 10]]
            pgs_dt_cols = [f"pgs{i}dt" for i in [2, 4, 6, 8, 10]]

            preserved_otv_columns += mls_cols + pgs_cols + mls_dt_cols + pgs_dt_cols

        if disable_one_hot_encoding:
            for column in DF.columns:
                if DF[column].dtype == "object":
                    print("dropping ", column)
                    DF = DF.drop(column, axis=1)
                    preserved_otv_columns.remove(column)
        else:
            for column in columns:
                if DF[column].nunique() <= 10:
                    if DF[column].nunique() > 3:
                        print("one hot encoding ", column)
                        DF = pd.get_dummies(DF, columns=[column], drop_first=True)
        for column in DF.columns:
            if DF[column].dtype == "bool":
                DF[column] = DF[column].astype(int)

        time_columns = [column for column in DF.columns if column.endswith("dt")]
        time_columns = [column for column in time_columns if column != "dt"]
        time_column_indices = [DF.columns.get_loc(column) for column in time_columns]
        maybe_val_indices = [i - 1 for i in time_column_indices]

        for index, name in enumerate(time_columns):
            # remove dt or potential _ from column name
            print("hiding future times for ", name)
            time_string = name.replace("_dt", "")
            time_string = time_string.replace("dt", "")

            maybe_val_index = maybe_val_indices[index]
            if time_string in DF.columns[maybe_val_index]:
                val_col_name = DF.columns[maybe_val_index]
                print("found ", val_col_name)
                DF[val_col_name].loc[DF["dt"] <= DF[name]] = int(time_NaN_fill)
                DF[val_col_name].loc[DF[val_col_name] == 0] = int(time_NaN_fill)
                DF.rename(
                    columns={val_col_name: val_col_name + "_time_censored"},
                    inplace=True,
                )
                preserved_otv_columns.append(val_col_name + "_time_censored")
                preserved_otv_columns.remove(val_col_name)

            DF[name].loc[DF["dt"] <= DF[name]] = int(time_NaN_fill)
            DF.rename(columns={name: name + "_time_censored"}, inplace=True)
            preserved_otv_columns.append(name + "_time_censored")
            preserved_otv_columns.remove(name)

        # group by ptid and hour
        DF = DF.groupby(["ptid", "dt"]).max().reset_index()
        DF = DF.set_index(["ptid", "dt"])

        # reindex missing hours
        full_index = pd.MultiIndex(
            levels=[[], []], codes=[[], []], names=["ptid", "dt"]
        )
        for ptid in DF.index.get_level_values("ptid").unique():
            min_dt = DF.loc[ptid].index.min()
            max_dt = DF.loc[ptid].index.max()
            idx = pd.MultiIndex.from_product(
                [[ptid], range(min_dt, max_dt + 1)], names=["ptid", "dt"]
            )
            full_index = full_index.append(idx)

        DF = DF.reindex(full_index).reset_index()

        target = (
            DF.groupby(["ptid"])["size_mls"]
            .shift(-lookahead_hours)
            .rolling(lookahead_hours, min_periods=1)
            .max()
        )

        #preserve true scans
        def get_true_scans(df):
            true_scans = np.zeros(df.shape[0])

            for i in range(df.shape[0]):
                if df['size_mls'][i] != time_NaN_fill and df['size_mls'][i] != np.nan:
                    true_scans[i] == 1
                
            df['true_scan'] = true_scans
            return df

        if preserve_true_scans:
            DF = get_true_scans(DF)

                

        if use_ffil_and_rolling:
            otv_columns = [
                column for column in DF.columns if column in preserved_otv_columns
            ]
            otv_columns.remove("ptid")

            if dont_transform_otv:
                df_otv = DF[otv_columns].copy()
                source_DF = DF.drop(otv_columns, axis=1).copy()
            else:
                source_DF = DF.copy()

            DF_rolling = source_DF.groupby("ptid").apply(
                lambda x: x.rolling(lookback_hours, min_periods=1).max()
            )
            if add_min_values:
                min_df = source_DF.copy()
                # replace all time_NaN_fill with NaN
                min_df = min_df.replace(int(time_NaN_fill), np.nan)
                DF_min = min_df.groupby("ptid").apply(
                    lambda x: x.rolling(lookback_hours, min_periods=1).min()
                )

            DF_ffill = (
                source_DF.replace(int(time_NaN_fill), np.nan).groupby("ptid").ffill()
            )
            DF_ffill = DF_ffill.fillna(int(time_NaN_fill))

            DF_ffill = DF_ffill.drop(["dt"], axis=1)
            DF_ffill.columns = ["ffill_" + column for column in DF_ffill.columns]

            # add rolling to name of every col in DF_rolling
            DF_rolling = DF_rolling.drop(["obsTime"], axis=1)
            DF_rolling.columns = ["rolling_" + column for column in DF_rolling.columns]
            DF_rolling.rename(columns={"rolling_ptid": "ptid"}, inplace=True)
            DF_rolling.rename(columns={"rolling_dt": "dt"}, inplace=True)

            if add_min_values:
                DF_min = DF_min.drop(["obsTime"], axis=1)
                DF_min.columns = ["min_" + column for column in DF_min.columns]
                DF_min = DF_min.drop(["min_ptid", "min_dt"], axis=1)
                DF_rolling = pd.concat([DF_rolling, DF_min], axis=1)

            del DF
            del source_DF

            DF_rolling = DF_rolling.drop('ptid', axis=1).reset_index()
            DF_rolling = DF_rolling.drop('level_1', axis=1)         

            if dont_transform_otv:

                DF = pd.concat(
                    [DF_rolling, DF_ffill, df_otv],
                    axis=1,
                )
            else:
                DF = pd.concat([DF_rolling, DF_ffill], axis=1)

        if preserve_true_scans:
            DF.drop('rolling_true_scan', axis =1)
            DF.rename(columns = {'ffill_true_scan':'true_scan'})

        DF["target"] = target
        DF = DF.dropna(subset=["target"])
        DF = DF[DF.target != time_NaN_fill]

        DF = DF.dropna(
            how="all",
            subset=[column for column in DF.columns if column != "target"],
        )

        # drop rows that are duplicates in all values except dt and obsTime
        DF = DF.drop_duplicates(
            subset=[column for column in DF.columns if column not in ["dt", "obsTime"]],
            keep="last",
        )

        DF = DF.reset_index(drop=True)
        target = DF["target"].copy()
        DF = DF.drop("target", axis=1)
        return DF, target

    preserved_bmc_otv_columns = list(
        set(OTV.columns).intersection(set(bmc_OLTV.columns))
    )

    presevered_otv_columns = list(set(OTV.columns).intersection(set(OLTV.columns)))

    OLTV, target = generate_OLTV(OLTV, presevered_otv_columns)
    bmc_OLTV, bmc_target = generate_OLTV(bmc_OLTV, preserved_bmc_otv_columns)

    def process_dataframe(df, times, base_time, time_NaN_fill):
        df[base_time] = pd.to_datetime(df[base_time])
        for time in times:
            if time in list(df.columns):
                df[time] = pd.to_datetime(df[time])
                df[time] = (df[time] - df[base_time]).dt.total_seconds() // 3600
                df[time] = df[time].fillna(int(time_NaN_fill))
        return df

    OTV = process_dataframe(OTV, times, base_time, time_NaN_fill)
    bmc_OTV = process_dataframe(bmc_OTV, bmc_times, base_time, time_NaN_fill)

    if use_mode_as_feature:
        OLTV['target_mode'] = target.mode().values.item()
        bmc_OLTV['target_mode'] = bmc_target.mode().values.item()
    
    if use_prev_as_feature:
        prev_values = np.zeros(shape = [OLTV.shape[0], ])
        prev_pt = None
        for i, pt in enumerate(OLTV['ptid']):
            if prev_pt is None or prev_pt != pt:
                prev_values[i] = 0
            else:
                prev_values[i] = OLTV.iloc[i]['ffill_size_mls']
            prev_pt = pt
        
        prev_values_bmc = np.zeros(shape = [bmc_OLTV.shape[0], ])
        prev_pt = None
        for i, pt in enumerate(bmc_OLTV['ptid']):
            if prev_pt is None or prev_pt != pt:
                prev_values_bmc[i] = 0
            else:
                prev_values_bmc[i] = bmc_OLTV.iloc[i]['ffill_size_mls']
            prev_pt = pt

        OLTV['prev_mls'] = prev_values
        bmc_OLTV['prev_mls'] = prev_values_bmc


    return OLTV, target, bmc_OLTV, bmc_target, OTV, bmc_OTV


# LTV.to_csv(os.path.join(processed_data_path, f"simplified_processed_LTV_{data_version}.csv"), index=False)


# %%

# %%

# %%
if __name__ == "__main__":
    disable_one_hot_encoding = True
    time_NaN_fill = -10
    use_only_rc_values = True
    lookback_hours = 24
    lookahead_hours = 24
    use_lsw_as_base_time_instead_of_pres = True
    create_new_mls_and_pgs_vals = True
    use_ffil_and_rolling = True
    OLTV, target, bmc_OLTV, bmc_target, OTV, bmc_OTV = create_MGH_and_BMC(
        filter_out_less_than_24_hours=True,
        disable_one_hot_encoding=disable_one_hot_encoding,
        time_NaN_fill=time_NaN_fill,
        use_only_rc_values=use_only_rc_values,
        lookback_hours=lookback_hours,
        lookahead_hours=lookahead_hours,
        use_lsw_as_base_time_instead_of_pres=use_lsw_as_base_time_instead_of_pres,
        create_new_mls_and_pgs_vals=create_new_mls_and_pgs_vals,
        use_ffil_and_rolling=use_ffil_and_rolling,
        dont_transform_otv=True,
        panos_mode=False,
        add_min_values=True,
        use_mode_as_feature=True,
        use_prev_as_feature=True,
        preserve_true_scans=True,
    )
    # save target histograms for each dataset to png

    target.hist().get_figure().savefig("histogram.png")
    bmc_target.hist().get_figure().savefig("bmc_histogram.png")

    print("done")

# %%
