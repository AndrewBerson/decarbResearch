import pandas as pd
import itertools
from pathlib import Path
import numpy as np
import os
from tqdm import tqdm

# Paths
INPUT_PATH = Path("resultsFiles/112_35results")
CLEAN_RESULTS_PATH = INPUT_PATH / "clean_results"


def main():
    reload_results = True    # set to True if using a new raw results excel document

    # load data and make copies of scenarios specified in the controller
    df = load_data(reload=reload_results)
    df_loads = load_load_shapes(reload=reload_results)


def load_data(reload):
    """ Function to load either raw or already cleaned LEAP data"""
    if reload:
        df = load_all_files(INPUT_PATH)    # load all raw results file from LEAP output
        df = reformat(df)                  # cleanup up all data to preferred format
        df.to_csv(CLEAN_RESULTS_PATH / 'combined_results.csv')
    else:
        df = pd.read_csv(CLEAN_RESULTS_PATH / 'combined_results.csv', header=0, index_col=0)

    return df


def load_all_files(input_path, sheet="Results"):
    """ function to intake all raw results files within the specified path """
    df = pd.DataFrame
    added_scenarios = set()

    # iterate through all files and combine the datasets from the excel documents
    file_list = os.listdir(input_path)
    file_list.sort(reverse=True)
    i = 0
    for fname in file_list:
        # print(fname)
        f = os.path.join(input_path, fname)
        if os.path.isfile(f) and (fname[0] not in [".", "~"]):
            df_excel = pd.read_excel(f, sheet_name=sheet)

            # exclude scenarios that have already been added
            # eg: if baseline is included in multiple results files, only add it once
            row_ids = [j for j, sce in enumerate(df_excel["Scenario"]) if sce not in added_scenarios]
            if i == 0:
                df = df_excel.iloc[row_ids, :].copy()
            else:
                df = pd.concat([df, df_excel.iloc[row_ids, :].copy()], sort=True)
            added_scenarios = set(df["Scenario"].unique())
            i += 1

    return df.reset_index(drop=True)


def reformat(df_excel):
    """ Function to take LEAP script output and convert it to cleaned up long dataframe """
    df_excel = df_excel.drop(columns=["Index"])
    df_excel = df_excel.transpose()

    scenarios = df_excel.loc['Scenario', :].unique()
    result_vars = df_excel.loc['Result Variable', :].unique()
    fuels = df_excel.loc['Fuel', :].unique()
    branches = df_excel.loc['Branch', :].unique()

    id_cols = ['Scenario', 'Result Variable', 'Fuel']
    drop_rows = id_cols

    df = pd.DataFrame(columns=['Year'] + id_cols + branches.tolist())

    # iterate through all combinations of scenario, result and fuel
    progress_bar = tqdm(total=len(scenarios)*len(result_vars)*len(fuels), desc='Reformatting results')
    for scenario, result_var, fuel in itertools.product(scenarios, result_vars, fuels):
        progress_bar.update()

        # find columns in df_excel that contain relevant scenario, result, and fuel
        col_mask = np.array(
            (df_excel.loc['Scenario', :] == scenario) &
            (df_excel.loc['Result Variable', :] == result_var) &
            (df_excel.loc['Fuel', :] == fuel)
        )
        col_ids = list(np.where(col_mask)[0])

        # some scenario, result, fuel combos are null, skip these ones
        if len(col_ids) == 0:
            continue

        # form new df of results just for this combo of scenario, result, fuel
        df_new = df_excel.iloc[:, col_ids].copy()

        # drop scenario, result, and fuel rows (they should be columns)
        df_new.drop(labels=drop_rows, axis=0, inplace=True)

        # set Branch as the column names instead of as the top row
        df_new.columns = df_new.loc['Branch', :]
        df_new.drop(labels='Branch', axis=0, inplace=True)

        # add back in scenario, result variable, and fuel as columns
        df_new[['Scenario', 'Result Variable', 'Fuel']] = [scenario, result_var, fuel]

        # make Year its own column
        df_new.reset_index(inplace=True)
        df_new.rename({'index': 'Year'}, axis=1, inplace=True)

        # append new dataframe to dataframe that will ultimately be returned
        df = pd.concat([df, df_new], ignore_index=True, sort=True)

    progress_bar.close()

    # organize columns
    cols = ['Year'] + id_cols + list(set(df.columns) - set(id_cols + ['Year']))
    df = df[cols]

    return df.fillna(0)


def load_load_shapes(reload):
    """ Function to generate df of loadshapes """

    if reload:
        df_excel = load_all_files(INPUT_PATH, "Shapes")

        id_cols = ['Index', 'Year', 'Scenario', 'Result Variable', 'Branch']
        hour_cols = list(set(df_excel.columns) - set(id_cols))

        df = pd.DataFrame(columns=['Year', 'Hour', 'Scenario', 'Branch', 'Result Variable', 'Value'])

        for row in tqdm(df_excel.index, 'Loading load shapes'):
            df_to_add = pd.DataFrame(columns=['Year', 'Hour', 'Scenario', 'Branch', 'Result Variable', 'Value'])
            df_to_add['Hour'] = pd.Series(hour_cols)
            df_to_add['Value'] = pd.Series(df_excel.loc[row, hour_cols]).reset_index(drop=True)
            df_to_add[['Year', 'Scenario', 'Branch', 'Result Variable']] = df_excel.loc[
                row, ['Year', 'Scenario', 'Branch', 'Result Variable']]
            df = pd.concat([df, df_to_add], ignore_index=True)

        df.to_csv(CLEAN_RESULTS_PATH / "shapes.csv")
        return df

    else:
        df = pd.read_csv(CLEAN_RESULTS_PATH / "shapes.csv", header=0, index_col=0)
        return df


if __name__ == "__main__":
    CLEAN_RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    main()