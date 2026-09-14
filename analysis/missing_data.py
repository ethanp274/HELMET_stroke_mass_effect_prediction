# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns

file_paths = [
    "data/raw/LTV_BMC_2.csv",
    "data/raw/OTV_BMC_3.csv",
    "data/raw/LTV_2.0.csv",
    "data/raw/OTV_2.1.csv",
]
LTV_BMC, OTV_BMC, LTV_MGH, OTV_MGH = [pd.read_csv(file) for file in file_paths]
OTV_MGH.drop(
    ["ethnicity", "aspects1", "aspects1dt", "death30", "gluc1", "na1", "wbc1", "cr1"],
    axis=1,
    inplace=True,
)


LTV_BMC = LTV_BMC.sort_values(["encounter_id", "dt"]).reset_index(drop=True)
LTV_BMC.rename(columns={"glucose": "gluc", "sodium": "na"}, inplace=True)
LTV_BMC.rename(columns={"creatinine": "cr", "osmolality": "osm"}, inplace=True)
LTV_BMC.rename(columns={"CDW_MRN": "ptid", "LSW": "lsw"}, inplace=True)

OTV_BMC.rename(columns={"CDW_MRN": "ptid", "LSW": "lsw"}, inplace=True)
OTV_BMC.rename(columns={"CMO": "CMO_disch", "CMO_DT": "CMO_dt"}, inplace=True)
OTV_BMC.rename(columns={"DNR": "DNR_disch", "DNR_DT": "DNR_dt"}, inplace=True)
OTV_BMC.rename(columns={"DISCH_DT": "Discharge_Date"}, inplace=True)
OTV_BMC.rename(columns={"X.24.hours": "<24 hours"}, inplace=True)
OTV_BMC.rename(columns={"first_arrival_dt": "pres"}, inplace=True)


BMC_los = OTV_BMC[["ptid", "lsw", "LOS"]]
OTV_MGH["Discharge_Date"] = pd.to_datetime(OTV_MGH["Discharge_Date"])
OTV_MGH["pres"] = pd.to_datetime(OTV_MGH["pres"])
# calculate the length of stay for MGH data using 'pres' and 'Discharge_Date'
OTV_MGH["LOS"] = (OTV_MGH["Discharge_Date"] - OTV_MGH["pres"]).dt.days


# Function to adjust LOS values and calculate mean and standard deviation
def adjust_los_and_calculate_stats(df, df_name):
    # Adjust LOS values
    df["LOS"] = df["LOS"].apply(lambda x: max(x, 1))
    # Calculate mean and standard deviation
    los_mean = df["LOS"].mean()
    los_sd = df["LOS"].std()
    # Print results
    print(f"Mean length of stay for {df_name} data: {los_mean:.2f} days")
    print(f"Standard deviation of length of stay for {df_name} data: {los_sd:.2f} days")
    # Drop LOS column
    df.drop("LOS", axis=1, inplace=True)


# Adjust LOS values and calculate statistics for both datasets
adjust_los_and_calculate_stats(OTV_MGH, "MGH")
adjust_los_and_calculate_stats(OTV_BMC, "BMC")


# Calculate the number of scans per patient
def calculate_scans_per_patient(df):
    # Group by 'ptid' and count non-NaN 'imagetype' values
    scans_per_patient = (
        df.groupby("ptid")["imagetype"].count().reset_index(name="scans_per_patient")
    )
    # take mean of the scans per patient
    mean_scans_per_patient = scans_per_patient["scans_per_patient"].mean()
    sd_scans_per_patient = scans_per_patient["scans_per_patient"].std()
    print(f"Mean number of scans per patient: {mean_scans_per_patient:.2f}")
    print(f"Standard deviation of scans per patient: {sd_scans_per_patient:.2f}")


calculate_scans_per_patient(LTV_BMC)
calculate_scans_per_patient(LTV_MGH)


OTV_columns_intersection = OTV_BMC.columns.intersection(OTV_MGH.columns)
OTV_BMC = OTV_BMC[OTV_columns_intersection]
OTV_MGH = OTV_MGH[OTV_columns_intersection]


columns_to_drop = [
    "DNR_dt",
    "CMO_dt",
    "surgdt",
    "rc_ph2_dt",
    "rc_parenchymal_dt",
    "mtdt",
    "tt_mt",
    "tici",
    "rc_mls7dt",
    "rc_pgs4dt",
    "tpadt",
    "rc_mls5dt",
    "rc_petechial_dt",
    "lvo_location",
    "first_osmotic",
    "cs",
    "gca",
]

for df in [OTV_BMC, OTV_MGH]:
    df.drop(columns_to_drop, axis=1, inplace=True)

# drop columns with mls in the name
columns_to_drop = [col for col in OTV_BMC.columns if "mls" in col]
for df in [OTV_BMC, OTV_MGH]:
    df.drop(columns_to_drop, axis=1, inplace=True)


def plot_events_per_patient(df, id_col, title):
    # Group by the ID column and count the number of events per patient
    events_per_patient = df.groupby(id_col).size()

    # Plot the distribution of events per patient
    plt.figure(figsize=(10, 6))
    events_per_patient.plot(kind="hist", bins=50, alpha=0.7)
    plt.title(f"Number of Events per Patient in {title}")
    plt.xlabel("Number of Events")
    plt.ylabel("Number of Patients")
    plt.grid(True)
    plt.show()


# Apply the function to the LTV dataframes
plot_events_per_patient(LTV_BMC, "ptid", "LTV_BMC")
plot_events_per_patient(LTV_MGH, "ptid", "LTV_MGH")


def calculate_percentage_less_than_events(df, df_name, id_col, event_thresholds):
    events_per_patient = df.groupby(id_col).size()
    for threshold in event_thresholds:
        less_than_threshold = events_per_patient[events_per_patient < threshold].count()
        percentage_less_than_threshold = (
            less_than_threshold / len(events_per_patient)
        ) * 100
        print(
            f"Percentage of {df_name} patients with fewer than {threshold} events: {percentage_less_than_threshold:.2f}%"
        )


# Run the function for different event thresholds for both MGH and BMC data
event_thresholds = [50, 100, 150]
calculate_percentage_less_than_events(LTV_MGH, "MGH", "ptid", event_thresholds)
calculate_percentage_less_than_events(LTV_BMC, "bmc", "ptid", event_thresholds)


# Extract year from 'dt' column of LTV_MGH using regex
MGH_LTV_dt_series = LTV_MGH["dt"].astype(str)
MGH_LTV_lsw_year = MGH_LTV_dt_series.apply(
    lambda x: re.search(r"\d{4}", x).group() if re.search(r"\d{4}", x) else "Unknown"
)
LTV_MGH["year"] = MGH_LTV_lsw_year


def plot_average_events_per_year_bar_chart(df, year_col, id_col):
    # Group by year and ID column, then calculate the average number of events
    average_events_per_year = (
        df.groupby([year_col, id_col])
        .size()
        .groupby(level=0)
        .mean()
        .reset_index(name="average_events")
    )

    # plot the bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(data=average_events_per_year, x=year_col, y="average_events")
    plt.title("Average Number of Events per Year")
    plt.xlabel("Year")
    plt.ylabel("Average Number of Events")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


# Apply the function to the LTV_MGH dataframe
plot_average_events_per_year_bar_chart(LTV_MGH, "year", "ptid")

# %%

def plot_percentage_missing_values(
    df, missing_value_threshold=0, title="Percentage of Missing Values per Column", return_table=False
):
    """
    This function takes a DataFrame and plots the percentage of missing values per column as a bar chart using seaborn,
    removing columns with less than a specified percentage of missing values before plotting.

    Parameters:
    - df: pandas DataFrame
    - missing_value_threshold: int, the minimum percentage of missing values a column must have to be included in the plot
    - title: str, the title of the plot
    """
    # Calculate the percentage of missing values per column
    total_values = len(df)
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / total_values) * 100

    # print the percentage of columns that have greater than the specified threshold of missing values
    columns_with_missing = missing_percentage[
        missing_percentage > missing_value_threshold
    ]
    print(
        f"Percentage of columns with greater than {missing_value_threshold}% missing values: {len(columns_with_missing) / len(df.columns) * 100:.2f}%"
    )

    # Remove columns with less than the specified threshold percent missing values
    missing_percentage_filtered = missing_percentage[
        missing_percentage >= missing_value_threshold
    ]

    # Sort the filtered missing percentages
    missing_percentage_sorted = missing_percentage_filtered.sort_values(ascending=False)

    if return_table:
        return missing_percentage_sorted

    # Plot the filtered missing percentages
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_percentage_sorted.index, y=missing_percentage_sorted.values)
    plt.title(title)
    plt.xlabel("Columns")
    plt.ylabel("Percentage of Missing Values")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

# %%
OTV_BMC_missing = plot_percentage_missing_values(
    OTV_BMC, 0.5, "Percentage of Missing Values in BMC OTV Data", return_table=True
)

OTV_MGH_missing = plot_percentage_missing_values(
    OTV_MGH, 0.5, "Percentage of Missing Values in MGH OTV Data", return_table=True
)

# %%
# Apply the function to the OTV data
plot_percentage_missing_values(
    OTV_BMC, 1, "Percentage of Missing Values in BMC OTV Data"
)
plot_percentage_missing_values(
    OTV_MGH, 1, "Percentage of Missing Values in MGH OTV Data"
)

# # Apply the function to the OTV data
# plot_percentage_missing_values(
#     OTV_BMC, 1, "Percentage of Missing Values in BMC OTV Data"
# )
# plot_percentage_missing_values(
#     OTV_MGH, 1, "Percentage of Missing Values in MGH OTV Data"
# )

# plot_percentage_missing_values(
#     OTV_BMC, 51, "Percentage of Missing Values in BMC OTV Data"
# )
# plot_percentage_missing_values(
#     OTV_MGH, 51, "Percentage of Missing Values in MGH OTV Data"
# )


# %%
