import pandas as pd
import itertools
from pathlib import Path
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from PyPDF2 import PdfMerger
from matplotlib.backends.backend_pdf import PdfPages

import seaborn as sns

DISCOUNT_RATE = 0.05
INPUT_PATH = Path("resultsFiles/mar6_2023")
CONTROLLER_PATH = INPUT_PATH / "results_controller"
CLEAN_RESULTS_PATH = INPUT_PATH / "clean_results"
CARB_PATH = INPUT_PATH / "carb_results"
FIGURES_PATH = INPUT_PATH / "figures"
EMISSIONS_RESULT_STRING = "One_Hundred Year GWP Direct At Point of Emissions"
COST_RESULT_STRING = "Social Costs"



def main():

    df = load_data(reload=False)
    scenario_comparisons = load_sce_comps()
    egen_resource_color_map, sector_color_map = load_color_maps()
    graph_emissions_over_time_scenario_comparisons(df, scenario_comparisons)
    graph_marginal_cost_over_time_scenario_comparisons(df, scenario_comparisons)


def load_data(reload):
    """ Function to load either raw or already cleaned LEAP data"""
    if reload:
        df = load_all_files(INPUT_PATH)
        df = reformat(df)
        df.to_csv(CLEAN_RESULTS_PATH / 'combined_results.csv')
    else:
        df = pd.read_csv(CLEAN_RESULTS_PATH / 'combined_results.csv', header=0, index_col=0)

    return df


def load_all_files(input_path):
    """ function to intake all raw results files within the specified path """
    df = pd.DataFrame
    added_scenarios = set()
    i = 0
    for fname in os.listdir(input_path):
        f = os.path.join(input_path, fname)
        if os.path.isfile(f) and (fname[0] not in [".", "~"]):
            df_excel = pd.read_excel(f, sheet_name="Results")

            # exclude scenarios that have already been added
            # eg: if baseline is included in multiple results files, only add it once
            row_ids = [i for i, sce in enumerate(df_excel["Scenario"]) if sce not in added_scenarios]
            if i == 0:
                df = df_excel.iloc[row_ids, :].copy()
            else:
                df = pd.concat([df, df_excel.iloc[row_ids, :].copy()], sort=True)
            added_scenarios = set(df["Scenario"].unique())
            i += 1

    return df


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

    df = pd.DataFrame(columns=id_cols + branches.tolist())

    # iterate through all combinations of scenario, result and fuel
    for s, r, f in itertools.product(scenarios, result_vars, fuels):

        # find columns in df_excel that contain relevant scenario, result, and fuel
        col_mask = np.array(
            (df_excel.loc['Scenario', :] == s) &
            (df_excel.loc['Result Variable', :] == r) &
            (df_excel.loc['Fuel', :] == f)
        )
        col_ids = [i for i, mask in enumerate(col_mask) if mask]

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
        df_new.loc[:, 'Scenario'] = s
        df_new.loc[:, 'Result Variable'] = r
        df_new.loc[:, 'Fuel'] = f

        # append new dataframe to dataframe that will ultimately be returned
        df = pd.concat([df, df_new], sort=True)

    # organize columns
    df.reset_index(inplace=True)
    df.rename({'index' : 'Year'}, axis=1, inplace=True)
    id_cols = ['Year'] + id_cols
    cols = id_cols + list(set(df.columns) - set(id_cols))
    df = df[cols]

    return df.fillna(0)


def load_sce_comps():
    """ Function to load scenario comparisons as dictated in controller """
    df = pd.read_excel(CONTROLLER_PATH / 'controller.xlsx', sheet_name="base_scenario_comparisons")

    scenario_comps = []

    for _, dfg in df.groupby('Group'):
        sc = dict()
        sc['scenarios'] = dfg['Scenario'].tolist()
        sc['name_map'] = dict(zip(dfg['Scenario'], dfg['Naming']))
        sc['line_map'] = dict(zip(dfg['Scenario'], dfg['Line']))
        sc['color_map'] = dict(zip(dfg['Scenario'], dfg['Color']))
        sc['legend_map'] = dict(zip(dfg['Scenario'], dfg['Include in legend']))
        scenario_comps.append(sc)

    return scenario_comps


def load_color_maps():
    """ Function to load color maps from controller """

    df = pd.read_excel(CONTROLLER_PATH / 'controller.xlsx', sheet_name="egen_resource_colors", index_col=0)
    egen_resource_color_map = {
        'long': df[df['Length'] == 'Long'].to_dict()['Color'],
        'short': df[df['Length'] == 'Short'].to_dict()['Color'],
    }

    df = pd.read_excel(CONTROLLER_PATH / 'controller.xlsx', sheet_name="sector_colors", index_col=0)
    sector_color_map = df.to_dict()['Color']

    return egen_resource_color_map, sector_color_map


def calculate_annual_result_by_subgroup(df_in, result_str, subgroup_dict):
    """ Function to calculate annual result summed branch subgroups for all scenarios """

    df_out = pd.DataFrame(columns=['Year', 'Scenario', 'Subgroup', 'Value'])
    for key, dfg in df_in[df_in['Result Variable'] == result_str].groupby(by=['Year', 'Scenario']):
        yr, sce = key
        mask = np.array(
            (dfg['Year'] == yr) &
            (dfg['Scenario'] == sce)
        )
        row_ids = list(np.where(mask)[0])
        for subgroup, branches in subgroup_dict.items():
            value = dfg[branches].iloc[row_ids].sum(axis=1).sum()
            df_out.loc[len(df_out.index)] = [yr, sce, subgroup, value]

    return df_out


def marginalize_it(df, relative_to):
    for subg, yr in itertools.product(df['Subgroup'].unique(), df['Year'].unique()):
        # create mask for values in the relative_to scenario
        relative_to_mask = np.array(
            (df['Scenario'] == relative_to) &
            (df['Subgroup'] == subg) &
            (df['Year'] == yr)
        )
        relative_to_ids = list(np.where(relative_to_mask)[0])

        # create mask for the values that are being marginalized
        marginalize_mask = np.array(
            (df['Subgroup'] == subg) &
            (df['Year'] == yr)
        )
        marginalize_ids = list(np.where(marginalize_mask)[0])

        df.iloc[marginalize_ids, df.columns.get_loc('Value')] -= float(
            df.iloc[relative_to_ids, df.columns.get_loc('Value')])

    return df



def graph_emissions_over_time_scenario_comparisons(df_in, scenario_comparisons):
    id_cols = ["Year", "Scenario", "Result Variable", "Fuel"]
    result_cols = list(set(df_in.columns) - set(id_cols))
    subgroup_dict = {
        'all_branches': result_cols,
    }

    df = calculate_annual_result_by_subgroup(df_in, EMISSIONS_RESULT_STRING, subgroup_dict)
    df['Value'] = df['Value'] / 1e6

    for i, sc in enumerate(scenario_comparisons):
        fig = plot_line_scenario_comparison_over_time(
            df, 'Scenario Emissions', 'Annual Emissions (Mt CO2e)', '', sc,
        )

        fig.write_image(FIGURES_PATH / f"emissions_comparison_{i}.pdf")


def graph_marginal_cost_over_time_scenario_comparisons(df_in, scenario_comparisons, relative_to='LEAP Version CARB Reference_0_nan'):
    id_cols = ["Year", "Scenario", "Result Variable", "Fuel"]
    result_cols = list(set(df_in.columns) - set(id_cols))
    subgroup_dict = {
        'all_branches': result_cols,
    }

    df = calculate_annual_result_by_subgroup(df_in, COST_RESULT_STRING, subgroup_dict)
    df = marginalize_it(df, relative_to)
    df['Value'] = df['Value'] / 1e9

    for i, sc in enumerate(scenario_comparisons):
        fig = plot_line_scenario_comparison_over_time(
            df, 'Scenario Marginal Costs', '$/yr (Billion)', '', sc,
        )

        fig.write_image(FIGURES_PATH / f"cost_comparison_{i}.pdf")


def plot_line_scenario_comparison_over_time(df, title, yaxis_title, xaxis_title, sce_comp):
    fig = go.Figure()

    for sce in sce_comp['scenarios']:
        df_sce = df[df['Scenario'] == sce].copy()
        fig.add_trace(go.Scatter(
            mode='lines',
            x=df_sce['Year'],
            y=df_sce['Value'],
            name=sce_comp['name_map'][sce],
            showlegend=sce_comp['legend_map'][sce],
            line=dict(
                color=sce_comp['color_map'][sce],
                dash=sce_comp['line_map'][sce],
            ),

        ))

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        autosize=False,
        width=800,
        height=500,
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.08,
            xanchor='left',
            x=0,
        )
    )

    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)

    return fig


if __name__ == "__main__":
    CLEAN_RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    main()