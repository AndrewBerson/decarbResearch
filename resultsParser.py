import pandas as pd
import itertools
from pathlib import Path
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

import seaborn as sns

DISCOUNT_RATE = 0.05
# RELATIVE_TO = 'CARB Ref + Clean Fuels_0'
RELATIVE_TO = 'LEAP Version CARB Reference_0_nan'
INPUT_PATH = Path("resultsFiles/apr13_2023")
CONTROLLER_PATH = INPUT_PATH / "results_controller"
CLEAN_RESULTS_PATH = INPUT_PATH / "clean_results"
FIGURES_PATH = INPUT_PATH / "figures"
EMISSIONS_RESULT_STRING = "One_Hundred Year GWP Direct At Point of Emissions"
COST_RESULT_STRING = "Social Costs"
GENERATION_STRING = "Outputs by Output Fuel"
CAPACITY_ADDED_STRING = "Capacity Added"


def main():
    reload_results = False    # set to True if using a new raw results excel document

    df = load_data(reload=reload_results)
    egen_resource_color_map_long, egen_resource_color_map_short, sector_color_map = load_color_maps()
    egen_branch_map_long, egen_branch_map_short = form_egen_branch_maps(df)
    sector_branch_map = form_sector_branch_map(df)

    # comparisons between scenarios
    base_scenario_comparison_graphs(df, egen_resource_color_map_long, egen_resource_color_map_short, sector_color_map,
                                    egen_branch_map_long, egen_branch_map_short, sector_branch_map)

    # results limited to one scenario
    individual_scenario_graphs(df, egen_resource_color_map_long, egen_resource_color_map_short, sector_color_map,
                               egen_branch_map_long, egen_branch_map_short, sector_branch_map)

    # load shape graphs
    # df_loads = load_load_shapes(reload=reload_results)
    # load_shape_graphs(df_loads, sector_color_map)

    # Sensitivity graphs
    sensitivity_graphs(df, sector_color_map)

    # RPS graphs


def sensitivity_graphs(df, sector_color_map):
    tech_choice_scenarios, tech_choice_graph_params = load_tech_choice_graph_params()
    tech_choice_scenarios = tech_choice_scenarios + [RELATIVE_TO]
    graph_tech_choice_emissions(
        df_in=df[df['Scenario'].isin(tech_choice_scenarios)],
        tech_choice_graph_params=tech_choice_graph_params,
        color_map=sector_color_map,
        year=2045,
    )
    graph_tech_choice_marginal_cost(
        df_in=df[df['Scenario'].isin(tech_choice_scenarios)],
        tech_choice_graph_params=tech_choice_graph_params,
        color_map=sector_color_map,
    )
    graph_tech_choice_cost_of_abatement(
        df_in=df[df['Scenario'].isin(tech_choice_scenarios)],
        tech_choice_graph_params=tech_choice_graph_params,
        color_map=sector_color_map,
    )


def base_scenario_comparison_graphs(df, egen_resource_color_map_long, egen_resource_color_map_short, sector_color_map,
                                    egen_branch_map_long, egen_branch_map_short, sector_branch_map):
    scenarios_to_compare, scenario_comparison_params = load_sce_comps()
    relevant_scenarios = scenarios_to_compare + [RELATIVE_TO]
    df_scenario_comp = df[df['Scenario'].isin(relevant_scenarios)]
    graph_emissions_over_time_scenario_comparisons(
        df_in=df_scenario_comp,
        scenario_comparisons=scenario_comparison_params,
    )
    graph_marginal_cost_over_time_scenario_comparisons(
        df_in=df_scenario_comp,
        scenario_comparisons=scenario_comparison_params,
    )
    graph_marginal_emissions_vs_marginal_cost_scatter_scenario_comparison(
        df_in=df_scenario_comp,
        scenario_comparisons=scenario_comparison_params,
    )
    graph_cost_of_co2_abatement_bar(
        df_in=df_scenario_comp,
        scenario_comparisons=scenario_comparison_params,
    )
    graph_egen_by_resource_scenario_comparison(
        df_in=df_scenario_comp,
        scenario_comparisons=scenario_comparison_params,
        color_map=egen_resource_color_map_long,
        branch_map=egen_branch_map_long,
        file_suffix='long',
    )
    graph_egen_by_resource_scenario_comparison(
        df_in=df_scenario_comp,
        scenario_comparisons=scenario_comparison_params,
        color_map=egen_resource_color_map_short,
        branch_map=egen_branch_map_short,
        file_suffix='short',
    )
    graph_cumulative_marginal_costs_by_sector_scenario_comparison(
        df_in=df_scenario_comp,
        scenario_comparisons=scenario_comparison_params,
        color_map=sector_color_map,
        branch_map=sector_branch_map,
        year=2045,
        relative_to=RELATIVE_TO,
    )
    graph_cumulative_marginal_abated_emissions_by_sector_scenario_comparison(
        df_in=df_scenario_comp,
        scenario_comparisons=scenario_comparison_params,
        color_map=sector_color_map,
        branch_map=sector_branch_map,
        year=2045,
        relative_to=RELATIVE_TO,
    )
    graph_annual_marginal_abated_emissions_by_sector_scenario_comparison(
        df_in=df_scenario_comp,
        scenario_comparisons=scenario_comparison_params,
        color_map=sector_color_map,
        branch_map=sector_branch_map,
        year=2045,
        relative_to=RELATIVE_TO,
    )


def individual_scenario_graphs(df, egen_resource_color_map_long, egen_resource_color_map_short, sector_color_map,
                               egen_branch_map_long, egen_branch_map_short, sector_branch_map):
    individual_sce_graph_params = load_individual_scenarios()
    relevant_scenarios = individual_sce_graph_params['scenarios'] + [RELATIVE_TO]
    df_individual_scenarios = df[df['Scenario'].isin(relevant_scenarios)]
    graph_marginal_costs_by_sector_over_time(
        df_in=df_individual_scenarios,
        individual_sce_graph_params=individual_sce_graph_params,
        color_map=sector_color_map,
        branch_map=sector_branch_map,
        relative_to=RELATIVE_TO,
    )
    graph_marginal_emissions_by_sector_over_time(
        df_in=df_individual_scenarios,
        individual_sce_graph_params=individual_sce_graph_params,
        color_map=sector_color_map,
        branch_map=sector_branch_map,
        relative_to=RELATIVE_TO,
    )
    graph_emissions_by_sector_over_time(
        df_in=df_individual_scenarios,
        individual_sce_graph_params=individual_sce_graph_params,
        color_map=sector_color_map,
        branch_map=sector_branch_map,
    )
    graph_egen_by_resource_over_time(
        df_in=df_individual_scenarios,
        individual_sce_graph_params=individual_sce_graph_params,
        color_map=egen_resource_color_map_short,
        branch_map=egen_branch_map_short,
        suffix='short'
    )
    graph_cumulative_egen_capacity_added_over_time(
        df_in=df_individual_scenarios,
        individual_sce_graph_params=individual_sce_graph_params,
        color_map=egen_resource_color_map_short,
        branch_map=egen_branch_map_short,
        suffix='short'
    )


def load_shape_graphs(df_loads, sector_color_map):
    load_scenarios_to_compare, load_scenario_comparison_params, individual_load_params = load_load_comps()
    df_load_comparison = df_loads[df_loads['Scenario'].isin(load_scenarios_to_compare)]
    graph_load_comparison(
        df_in=df_load_comparison,
        comp_params=load_scenario_comparison_params,
    )
    df_invidividual_loads = df_loads[df_loads['Scenario'].isin(list(individual_load_params.keys()))]
    graph_load_by_sector(
        df_in=df_invidividual_loads,
        params=individual_load_params,
        color_map=sector_color_map,
    )


def load_data(reload):
    """ Function to load either raw or already cleaned LEAP data"""
    if reload:
        df = load_all_files(INPUT_PATH)
        df = reformat(df)
        df.to_csv(CLEAN_RESULTS_PATH / 'combined_results.csv')
    else:
        df = pd.read_csv(CLEAN_RESULTS_PATH / 'combined_results.csv', header=0, index_col=0)

    return df


def load_all_files(input_path, sheet="Results"):
    """ function to intake all raw results files within the specified path """
    df = pd.DataFrame
    added_scenarios = set()
    i = 0
    for fname in os.listdir(input_path):
        f = os.path.join(input_path, fname)
        if os.path.isfile(f) and (fname[0] not in [".", "~"]):
            df_excel = pd.read_excel(f, sheet_name=sheet)

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

    df = pd.DataFrame(columns=['Year'] + id_cols + branches.tolist())

    # iterate through all combinations of scenario, result and fuel
    for s, r, f in itertools.product(scenarios, result_vars, fuels):
        print(f"{s}, {r}, {f}")
        # find columns in df_excel that contain relevant scenario, result, and fuel
        col_mask = np.array(
            (df_excel.loc['Scenario', :] == s) &
            (df_excel.loc['Result Variable', :] == r) &
            (df_excel.loc['Fuel', :] == f)
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
        df_new['Scenario'] = s
        df_new['Result Variable'] = r
        df_new['Fuel'] = f

        # make Year its own column
        df_new.reset_index(inplace=True)
        df_new.rename({'index': 'Year'}, axis=1, inplace=True)

        # append new dataframe to dataframe that will ultimately be returned
        df = pd.concat([df, df_new], ignore_index=True, sort=True)

    # organize columns
    id_cols = ['Year'] + id_cols
    cols = id_cols + list(set(df.columns) - set(id_cols))
    df = df[cols]

    return df.fillna(0)


def load_load_shapes(reload):
    """ Function to generate df of loadshapes """

    if reload:
        df_excel = load_all_files(INPUT_PATH, "Shapes")

        id_cols = ['Index', 'Year', 'Scenario', 'Result Variable', 'Branch']
        hour_cols = list(set(df_excel.columns) - set(id_cols))

        df = pd.DataFrame(columns=['Year', 'Hour', 'Scenario', 'Branch', 'Result Variable', 'Value'])

        for row in df_excel.index:
            # todo: fix error
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


def load_sce_comps():
    """ Function to load scenario comparisons as dictated in controller """
    df = pd.read_excel(CONTROLLER_PATH / 'controller.xlsx', sheet_name="base_scenario_comparisons")

    relevant_scenarios = set()
    scenario_comp_params = []

    for _, dfg in df.groupby('Group'):
        relevant_scenarios.update(set(dfg['Scenario'].unique()))
        sc = dict()
        sc['scenarios'] = dfg['Scenario'].tolist()
        sc['name_map'] = dict(zip(dfg['Scenario'], dfg['Naming']))
        sc['line_map'] = dict(zip(dfg['Scenario'], dfg['Line']))
        sc['color_map'] = dict(zip(dfg['Scenario'], dfg['Color']))
        sc['legend_map'] = dict(zip(dfg['Scenario'], dfg['Include in legend']))
        sc['marker_map'] = dict(zip(dfg['Scenario'], dfg['Marker']))
        sc['line_map_name'] = dict(zip(dfg['Naming'], dfg['Line']))
        sc['color_map_name'] = dict(zip(dfg['Naming'], dfg['Color']))
        sc['legend_map_name'] = dict(zip(dfg['Naming'], dfg['Include in legend']))
        sc['marker_map_name'] = dict(zip(dfg['Naming'], dfg['Marker']))
        sc['emissions_over_time'] = dfg['emissions over time'].unique()[0]
        sc['marginal_cost_over_time'] = dfg['marginal cost over time'].unique()[0]
        sc['marginal_emissions_vs_marginal_cost'] = dfg['marginal emissions vs marginal cost'].unique()[0]
        sc['cost_of_co2_abatement'] = dfg['cost of co2 abatement'].unique()[0]
        sc['egen_by_resource'] = dfg['egen by resource'].unique()[0]
        sc['cumulative_marginal_cost_by_sector'] = dfg['cumulative marginal cost by sector'].unique()[0]
        sc['cumulative_marginal_abated_emissions_by_sector'] = dfg['cumulative marginal abated emissions by sector'].unique()[0]
        sc['annual_marginal_abated_emissions_by_sector'] = dfg['annual marginal abated emissions by sector'].unique()[0]
        scenario_comp_params.append(sc)

    # relevant_scenarios.update([RELATIVE_TO])
    return list(relevant_scenarios), scenario_comp_params


def load_load_comps():
    """ Function to load scenario comparisons as dictated in controller """
    df = pd.read_excel(CONTROLLER_PATH / 'controller.xlsx', sheet_name="load_shape_comparisons")

    relevant_scenarios = set(df['Scenario'].unique())
    scenario_comp_params = []

    for _, dfg in df.groupby('Group'):
        # relevant_scenarios.update(set(dfg['Scenario'].unique()))
        sc = dict()
        sc['scenarios'] = dfg['Scenario'].tolist()
        sc['result_map'] = dict(zip(dfg['Scenario'], dfg['Result Variable']))
        sc['year_map'] = dict(zip(dfg['Scenario'], dfg['Year']))
        sc['name_map'] = dict(zip(dfg['Scenario'], dfg['Naming']))
        sc['line_map'] = dict(zip(dfg['Scenario'], dfg['Line']))
        sc['line_map_name'] = dict(zip(dfg['Naming'], dfg['Line']))
        sc['color_map'] = dict(zip(dfg['Scenario'], dfg['Color']))
        sc['color_map_name'] = dict(zip(dfg['Naming'], dfg['Color']))

        scenario_comp_params.append(sc)

    df = pd.read_excel(CONTROLLER_PATH / 'controller.xlsx', sheet_name="individual_load_shapes")
    relevant_scenarios.update(df['Scenario'].unique())
    individual_load_params = dict(zip(df['Scenario'], df['Naming']))

    return list(relevant_scenarios), scenario_comp_params, individual_load_params


def load_individual_scenarios():
    """ Function to load scenario comparisons as dictated in controller """
    df = pd.read_excel(CONTROLLER_PATH / 'controller.xlsx', sheet_name="individual_scenario_graphs")

    individual_scenario_graphs = {
        'scenarios': df['Scenario'].tolist(),
        'id_map': dict(zip(df['Scenario'], df['id'])),
        'name_map': dict(zip(df['Scenario'], df['Naming'])),
        'marginal_costs_by_sector': dict(zip(df['Scenario'], df['marginal costs by sector'])),
        'marginal_emissions_by_sector': dict(zip(df['Scenario'], df['marginal emissions by sector'])),
        'emissions_by_sector': dict(zip(df['Scenario'], df['emissions by sector'])),
        'egen_by_resource': dict(zip(df['Scenario'], df['egen by resource'])),
        'cumulative_egen_capacity_added': dict(zip(df['Scenario'], df['cumulative egen capacity added'])),
    }

    return individual_scenario_graphs


def load_color_maps():
    """ Function to load color maps from controller """

    df = pd.read_excel(CONTROLLER_PATH / 'controller.xlsx', sheet_name="egen_resource_colors")
    egen_resource_color_map_long = dict(zip(df[df['Length'] == 'Long']['Resource'], df[df['Length'] == 'Long']['Color']))
    egen_resource_color_map_short = dict(zip(df[df['Length'] == 'Short']['Resource'], df[df['Length'] == 'Short']['Color']))

    df = pd.read_excel(CONTROLLER_PATH / 'controller.xlsx', sheet_name="sector_colors")
    sector_color_map = dict(zip(df['Sector'], df['Color']))

    return egen_resource_color_map_long, egen_resource_color_map_short, sector_color_map


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


def discount_it(df):
    yrs = np.sort(df['Year'].unique())
    base_yr = yrs[0]

    # discount all costs
    for key, dfg in df.groupby(by=['Scenario', 'Subgroup', 'Year']):
        sce, subg, yr = key
        mask = np.array(
            (df['Scenario'] == sce) &
            (df['Subgroup'] == subg) &
            (df['Year'] == yr)
        )
        ids = list(np.where(mask)[0])
        df.iloc[ids, df.columns.get_loc('Value')] = dfg['Value'] / (1 + DISCOUNT_RATE) ** (yr - base_yr)

    return df


def cumsum_it(df):
    df = df.sort_values(by='Year', axis=0)
    for key, dfg in df.groupby(by=['Scenario', 'Subgroup']):
        sce, subg = key
        mask = np.array(
            (df['Scenario'] == sce) &
            (df['Subgroup'] == subg)
        )
        ids = list(np.where(mask)[0])
        df.iloc[ids, df.columns.get_loc('Value')] = dfg['Value'].cumsum(axis=0)

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
        if sc['emissions_over_time']:
            fig = plot_line_scenario_comparison_over_time(
                df, 'Scenario Emissions', 'Annual Emissions (Mt CO2e)', '', sc,
            )
            fig.write_image(FIGURES_PATH / f"emissions_over_time{i}.pdf")


def graph_marginal_cost_over_time_scenario_comparisons(
        df_in, scenario_comparisons, relative_to='LEAP Version CARB Reference_0_nan'):
    id_cols = ["Year", "Scenario", "Result Variable", "Fuel"]
    result_cols = list(set(df_in.columns) - set(id_cols))
    subgroup_dict = {
        'all_branches': result_cols,
    }

    df = calculate_annual_result_by_subgroup(df_in, COST_RESULT_STRING, subgroup_dict)
    df = marginalize_it(df, relative_to)
    df['Value'] = df['Value'] / 1e9

    for i, sc in enumerate(scenario_comparisons):
        if sc['marginal_cost_over_time']:
            fig = plot_line_scenario_comparison_over_time(
                df, 'Scenario Marginal Costs', '$/yr (Billion)', '', sc,
            )
            fig.write_image(FIGURES_PATH / f"marginal_cost_over_time{i}.pdf")


def evaluate_cumulative_marginal_emissions_cumulative_marginal_cost(df_in, subgroup_dict, relative_to='LEAP Version CARB Reference_0_nan'):
    df_cost = calculate_annual_result_by_subgroup(df_in, COST_RESULT_STRING, subgroup_dict)
    df_cost = marginalize_it(df_cost, relative_to)
    df_cost = discount_it(df_cost)
    df_cost = cumsum_it(df_cost)
    df_cost = df_cost.rename(columns={'Value': 'cumulative_marginal_cost'})

    df_emissions = calculate_annual_result_by_subgroup(df_in, EMISSIONS_RESULT_STRING, subgroup_dict)
    df_emissions = marginalize_it(df_emissions, relative_to)
    df_emissions = cumsum_it(df_emissions)
    df_emissions = df_emissions.rename(columns={'Value': 'cumulative_marginal_emissions'})

    df = df_emissions.merge(df_cost, how='outer', on=['Scenario', 'Subgroup', 'Year'])

    return df


def graph_marginal_emissions_vs_marginal_cost_scatter_scenario_comparison(df_in, scenario_comparisons, relative_to='LEAP Version CARB Reference_0_nan'):
    id_cols = ["Year", "Scenario", "Result Variable", "Fuel"]
    result_cols = list(set(df_in.columns) - set(id_cols))
    subgroup_dict = {
        'all_branches': result_cols,
    }

    df = evaluate_cumulative_marginal_emissions_cumulative_marginal_cost(df_in, subgroup_dict, relative_to)
    df = df[df['Year'] == df['Year'].max()].copy()
    df['cumulative_marginal_cost'] = df['cumulative_marginal_cost'] / 1e9
    df['cumulative_marginal_abated_emissions'] = -1 * df['cumulative_marginal_emissions'] / 1e9
    df = df.rename(columns={
        'cumulative_marginal_cost': 'yval',
        'cumulative_marginal_abated_emissions': 'xval'
    })

    for i, sc in enumerate(scenario_comparisons):
        if sc['marginal_emissions_vs_marginal_cost']:
            fig = plot_scatter_scenario_comparison(
                df, 'Emissions vs Cost', 'Marginal Abated Emissions (Gt CO2e)', 'Marginal Cost ($B)', sc,
            )
            fig.write_image(FIGURES_PATH / f"marginal_emissions_vs_marginal_cost{i}.pdf")


def graph_cost_of_co2_abatement_bar(df_in, scenario_comparisons, relative_to='LEAP Version CARB Reference_0_nan'):
    id_cols = ["Year", "Scenario", "Result Variable", "Fuel"]
    result_cols = list(set(df_in.columns) - set(id_cols))
    subgroup_dict = {
        'all_branches': result_cols,
    }

    df = evaluate_cumulative_marginal_emissions_cumulative_marginal_cost(df_in, subgroup_dict, relative_to)
    df = df[df['Year'] == df['Year'].max()].copy()    # filter to end year
    df['Value'] = -1 * df['cumulative_marginal_cost'] / df['cumulative_marginal_emissions']

    for i, sc in enumerate(scenario_comparisons):
        if sc['cost_of_co2_abatement']:
            df_graph = df[df['Scenario'].isin(sc['scenarios'])].copy()
            df_graph = df_graph.replace({'Scenario': sc['name_map']})
            # color_map = dict(zip([sc['name_map'][sce] for sce in sc['scenarios']],
            #                      [sc['color_map'][sce] for sce in sc['scenarios']]))

            fig = plot_bar_scenario_comparison(
                df=df_graph,
                title='Cost of Carbon Abatement',
                xaxis_title='$/t CO2e',
                yaxis_title='',
                color_dict=sc['color_map_name'],
                color_column='Scenario',
                include_legend=False,
            )
            fig.write_image(FIGURES_PATH / f"cost_of_co2_abatement{i}.pdf")


def graph_egen_by_resource_scenario_comparison(df_in, scenario_comparisons, color_map, branch_map, file_suffix, year=2045):

    df = calculate_annual_result_by_subgroup(df_in, GENERATION_STRING, branch_map)
    df = df[df['Year'] == year].copy()
    df['Value'] = df['Value'] / 1e9

    for i, sc in enumerate(scenario_comparisons):
        if sc['egen_by_resource']:
            df_graph = df[df['Scenario'].isin(sc['scenarios'])].copy()
            df_graph = df_graph.replace({'Scenario': sc['name_map']})

            fig = plot_bar_scenario_comparison(
                df=df_graph,
                title=f'Electricity Generation by Resource in {year}',
                xaxis_title='EJ',
                yaxis_title='',
                color_dict=color_map,
                color_column='Subgroup',
                include_legend=True,
            )
            fig.write_image(FIGURES_PATH / f"egen_by_resource_{file_suffix}_{i}.pdf")


def graph_cumulative_marginal_costs_by_sector_scenario_comparison(df_in, scenario_comparisons, color_map, branch_map,
                                                                  year=2045, relative_to='LEAP Version CARB Reference_0_nan'):

    df = evaluate_cumulative_marginal_emissions_cumulative_marginal_cost(df_in, branch_map, relative_to)
    df = df.rename(columns={'cumulative_marginal_cost': 'Value'})
    df = df[df['Year'] == year].copy()
    df['Value'] = df['Value'] / 1e9

    for i, sc in enumerate(scenario_comparisons):
        if sc['cumulative_marginal_cost_by_sector']:
            df_graph = df[df['Scenario'].isin(sc['scenarios'])].copy()
            df_graph = df_graph.replace({'Scenario': sc['name_map']})
            fig = plot_bar_scenario_comparison(
                df=df_graph,
                title=f'Cumulative Marginal Cost',
                xaxis_title='$B',
                yaxis_title='',
                color_dict=color_map,
                color_column='Subgroup',
                include_legend=True,
            )
            fig.write_image(FIGURES_PATH / f"cumulative_marginal_cost_by_sector_{i}.pdf")


def graph_cumulative_marginal_abated_emissions_by_sector_scenario_comparison(df_in, scenario_comparisons, color_map, branch_map,
                                                                  year=2045, relative_to='LEAP Version CARB Reference_0_nan'):

    df = evaluate_cumulative_marginal_emissions_cumulative_marginal_cost(df_in, branch_map, relative_to)
    df['Value'] = -1 * df['cumulative_marginal_emissions'] / 1e9
    df = df[df['Year'] == year].copy()

    for i, sc in enumerate(scenario_comparisons):
        if sc['cumulative_marginal_abated_emissions_by_sector']:
            df_graph = df[df['Scenario'].isin(sc['scenarios'])].copy()
            df_graph = df_graph.replace({'Scenario': sc['name_map']})

            fig = plot_bar_scenario_comparison(
                df=df_graph,
                title=f'Cumulative Marginal Abated Emissions',
                xaxis_title='Gt CO2e',
                yaxis_title='',
                color_dict=color_map,
                color_column='Subgroup',
                include_legend=True,
            )
            fig.write_image(FIGURES_PATH / f"cumulative_marginal_emissions_by_sector_{i}.pdf")


def graph_annual_marginal_abated_emissions_by_sector_scenario_comparison(df_in, scenario_comparisons, color_map, branch_map,
                                                                  year=2045, relative_to='LEAP Version CARB Reference_0_nan'):
    df = calculate_annual_result_by_subgroup(df_in, EMISSIONS_RESULT_STRING, branch_map)
    df = marginalize_it(df, relative_to)
    df = df[df['Year'] == year].copy()

    df['Value'] = -1 * df['Value'] / 1e6

    for i, sc in enumerate(scenario_comparisons):
        if sc['annual_marginal_abated_emissions_by_sector']:
            df_graph = df[df['Scenario'].isin(sc['scenarios'])].copy()
            df_graph = df_graph.replace({'Scenario': sc['name_map']})

            fig = plot_bar_scenario_comparison(
                df=df_graph,
                title=f'Annual Marginally Abated Emissions in {year}',
                xaxis_title='Mt CO2e',
                yaxis_title='',
                color_dict=color_map,
                color_column='Subgroup',
                include_legend=True,
            )
            fig.write_image(FIGURES_PATH / f"annual_marginal_emissions_by_sector_{i}.pdf")


def graph_marginal_costs_by_sector_over_time(df_in, individual_sce_graph_params,
                                             color_map, branch_map, relative_to='LEAP Version CARB Reference_0_nan'):
    df = calculate_annual_result_by_subgroup(df_in, COST_RESULT_STRING, branch_map)
    df = marginalize_it(df, relative_to)
    df['Value'] = df['Value'] / 1e9

    for sce in individual_sce_graph_params['scenarios']:
        if individual_sce_graph_params['marginal_costs_by_sector'][sce]:
            name = individual_sce_graph_params['name_map'][sce]
            fig = plot_bar_subgroup_over_time(
                df=df[df['Scenario'] == sce],
                title=f'Marginal Cost by Sector<br>{name}',
                xaxis_title='',
                yaxis_title='$B',
                color_map=color_map,
                include_sum=True,
            )
            fig.write_image(FIGURES_PATH / f"marginal_cost_over_time_by_sector_{individual_sce_graph_params['id_map'][sce]}.pdf")


def graph_marginal_emissions_by_sector_over_time(df_in, individual_sce_graph_params,
                                                 color_map, branch_map, relative_to='LEAP Version CARB Reference_0_nan'):

    df = calculate_annual_result_by_subgroup(df_in, EMISSIONS_RESULT_STRING, branch_map)
    df = marginalize_it(df, relative_to)
    df['Value'] = df['Value'] / 1e6

    for sce in individual_sce_graph_params['scenarios']:
        if individual_sce_graph_params['marginal_emissions_by_sector'][sce]:
            name = individual_sce_graph_params['name_map'][sce]
            fig = plot_bar_subgroup_over_time(
                df=df[df['Scenario'] == sce],
                title=f'Marginal Emissions by Sector<br>{name}',
                xaxis_title='',
                yaxis_title='Mt CO2e',
                color_map=color_map,
                include_sum=True,
            )
            fig.write_image(FIGURES_PATH / f"marginal_emissions_by_sector_over_time_{individual_sce_graph_params['id_map'][sce]}.pdf")


def graph_emissions_by_sector_over_time(df_in, individual_sce_graph_params,
                                        color_map, branch_map, relative_to='LEAP Version CARB Reference_0_nan'):

    df = calculate_annual_result_by_subgroup(df_in, EMISSIONS_RESULT_STRING, branch_map)
    df['Value'] = df['Value'] / 1e6

    for sce in individual_sce_graph_params['scenarios']:
        if individual_sce_graph_params['emissions_by_sector'][sce]:
            name = individual_sce_graph_params['name_map'][sce]
            fig = plot_bar_subgroup_over_time(
                df=df[df['Scenario'] == sce],
                title=f'Emissions by Sector<br>{name}',
                xaxis_title='',
                yaxis_title='Mt CO2e',
                color_map=color_map,
                include_sum=True,
            )
            fig.write_image(FIGURES_PATH / f"emissions_by_sector_over_time_{individual_sce_graph_params['id_map'][sce]}.pdf")


def graph_egen_by_resource_over_time(df_in, individual_sce_graph_params,
                                     color_map, branch_map, suffix):
    df = calculate_annual_result_by_subgroup(df_in, GENERATION_STRING, branch_map)
    df['Value'] = df['Value'] / 1e9

    for sce in individual_sce_graph_params['scenarios']:
        if individual_sce_graph_params['egen_by_resource'][sce]:
            name = individual_sce_graph_params['name_map'][sce]
            fig = plot_bar_subgroup_over_time(
                df=df[df['Scenario'] == sce],
                title=f'Electricity Generation by Resource<br>{name}',
                xaxis_title='',
                yaxis_title='EJ',
                color_map=color_map,
                include_sum=True,
            )
            fig.write_image(FIGURES_PATH / f"egen_by_resource_over_time_{suffix}_{individual_sce_graph_params['id_map'][sce]}.pdf")


def graph_cumulative_egen_capacity_added_over_time(df_in, individual_sce_graph_params,
                                                   color_map, branch_map, suffix):
    df = calculate_annual_result_by_subgroup(df_in, CAPACITY_ADDED_STRING, branch_map)
    df = cumsum_it(df)
    df['Value'] = df['Value'] / 1e3

    for sce in individual_sce_graph_params['scenarios']:
        if individual_sce_graph_params['cumulative_egen_capacity_added'][sce]:
            name = individual_sce_graph_params['name_map'][sce]
            fig = plot_bar_subgroup_over_time(
                df=df[df['Scenario'] == sce],
                title=f'Egen Capacity Added by Resource<br>{name}',
                xaxis_title='',
                yaxis_title='GW',
                color_map=color_map,
                include_sum=True,
            )
            fig.write_image(FIGURES_PATH / f"egen_capacity_added_over_time_{suffix}_{individual_sce_graph_params['id_map'][sce]}.pdf")


def load_tech_choice_graph_params():
    """ Function to load scenario comparisons as dictated in controller """
    df = pd.read_excel(CONTROLLER_PATH / 'controller.xlsx', sheet_name="tech_choice_plots")

    relevant_scenarios = set()
    scenario_comp_params = []

    for _, dfg in df.groupby('Plot'):
        relevant_scenarios.update(set(dfg['Scenario'].unique()))
        relevant_scenarios.update(set(dfg['relative_to'].unique()))
        sc = dict()
        sc['scenarios'] = dfg['Scenario'].tolist()
        sc['relative_to'] = dfg['relative_to'].unique()[0]
        sc['name_map'] = dict(zip(dfg['Scenario'], dfg['Naming']))
        sc['sector_map'] = dict(zip(dfg['Scenario'], dfg['Sector']))
        scenario_comp_params.append(sc)

    return list(relevant_scenarios), scenario_comp_params


def graph_tech_choice_emissions(df_in, tech_choice_graph_params, color_map, year=2045):
    id_cols = ["Year", "Scenario", "Result Variable", "Fuel"]
    result_cols = list(set(df_in.columns) - set(id_cols))
    subgroup_dict = {
        'all_branches': result_cols,
    }

    df = calculate_annual_result_by_subgroup(df_in, EMISSIONS_RESULT_STRING, subgroup_dict)
    df = df[df['Year'] == year]
    df = df.reset_index()    # need to reset index for the marginalize_it function to work
    df['Value'] = -1 * df['Value'] / 1e6

    for i, tc in enumerate(tech_choice_graph_params):
        df_graph = marginalize_it(df, tc['relative_to'])
        df_graph = df_graph[df_graph['Scenario'].isin(tc['scenarios'])].copy()
        df_graph['Sector'] = df_graph['Scenario'].map(tc['sector_map'])
        df_graph = df_graph.replace({'Scenario': tc['name_map']})
        df_graph = df_graph.sort_values(by=['Sector', 'Scenario'])

        fig = plot_bar_scenario_comparison(
            df=df_graph,
            title=f'Abated Annual Emissions Contribution, {year}',
            xaxis_title='Mt CO2e',
            yaxis_title='',
            color_dict=color_map,
            color_column='Sector',
            include_legend=True,
        )

        fig = update_to_tall_fig(fig)
        fig.write_image(FIGURES_PATH / f"tech_choice_emissions_{i}.pdf")


def graph_tech_choice_marginal_cost(df_in, tech_choice_graph_params, color_map, year=2045):
    id_cols = ["Year", "Scenario", "Result Variable", "Fuel"]
    result_cols = list(set(df_in.columns) - set(id_cols))
    subgroup_dict = {
        'all_branches': result_cols,
    }

    df = calculate_annual_result_by_subgroup(df_in, COST_RESULT_STRING, subgroup_dict)
    df = discount_it(df)
    df = cumsum_it(df)
    df = df[df['Year'] == year]
    df = df.reset_index()    # need to reset index for the marginalize_it function to work
    df['Value'] = df['Value'] / 1e9

    for i, tc in enumerate(tech_choice_graph_params):
        df_graph = marginalize_it(df, tc['relative_to'])
        df_graph = df_graph[df_graph['Scenario'].isin(tc['scenarios'])].copy()
        df_graph['Sector'] = df_graph['Scenario'].map(tc['sector_map'])
        df_graph = df_graph.replace({'Scenario': tc['name_map']})
        df_graph = df_graph.sort_values(by=['Sector', 'Scenario'])

        fig = plot_bar_scenario_comparison(
            df=df_graph,
            title='Marginal Cost Contribution',
            xaxis_title='$B',
            yaxis_title='',
            color_dict=color_map,
            color_column='Sector',
            include_legend=True,
        )
        fig = update_to_tall_fig(fig)
        fig.write_image(FIGURES_PATH / f"tech_choice_costs_{i}.pdf")


def graph_tech_choice_cost_of_abatement(df_in, tech_choice_graph_params, color_map):
    id_cols = ["Year", "Scenario", "Result Variable", "Fuel"]
    result_cols = list(set(df_in.columns) - set(id_cols))
    subgroup_dict = {
        'all_branches': result_cols,
    }

    for i, params in enumerate(tech_choice_graph_params):
        # relative_to = params['relative_to']
        relative_to = RELATIVE_TO
        df_graph = evaluate_cumulative_marginal_emissions_cumulative_marginal_cost(df_in, subgroup_dict, relative_to)
        df_graph = df_graph[
            (df_graph['Scenario'].isin(params['scenarios'])) &
            (df_graph['Year'] == df_graph['Year'].max())
             ].copy()
        df_graph['Value'] = -1 * df_graph['cumulative_marginal_cost'] / df_graph['cumulative_marginal_emissions']
        df_graph['Sector'] = df_graph['Scenario'].map(params['sector_map'])
        df_graph = df_graph.replace({'Scenario': params['name_map']})
        df_graph = df_graph.sort_values(by=['Sector', 'Scenario'])
        fig = plot_bar_scenario_comparison(
            df=df_graph,
            title='Cost of Abatement',
            xaxis_title='$ / tonne CO2e',
            yaxis_title='',
            color_dict=color_map,
            color_column='Sector',
            include_legend=True,
        )
        fig = update_to_tall_fig(fig)
        # fig.update_layout(xaxis_range=[-500, 1000])
        fig.write_image(FIGURES_PATH / f"tech_choice_cost_of_abatement_{i}.pdf")


def graph_load_by_sector(df_in, params, color_map):
    df = df_in[df_in['Result Variable'] == 'Load Shape'].copy()
    df['Value'] = df['Value'] / 1e3

    for i, (key, dfg) in enumerate(df.groupby(['Year', 'Scenario'])):
        yr, sce = key
        fig = plot_area_subgroup_over_time(
            df=dfg[
                (dfg['Scenario'] == sce) &
                (dfg['Year'] == yr)
            ],
            title=f"Electric Load by Sector<br>{params[sce]}",
            xaxis_title="Representative Day",
            yaxis_title="GW",
            color_map=color_map,
            yaxis_col='Value',
            xaxis_col='Hour',
            color_col='Branch',
            include_sum=True,
        )
        fig = update_to_load_shape_layout(fig)
        fig.write_image(FIGURES_PATH / f"load_shape_by_sector_{i}.pdf")


def graph_load_comparison(df_in, comp_params):
    df = sum_load_across_branches(df_in)
    df['Value'] = df['Value'] / 1e3

    for i, comp in enumerate(comp_params):
        df_graph = pd.DataFrame(columns=df.columns)
        for sce in comp['scenarios']:
            df_graph = pd.concat([
                df_graph,
                df[
                    (df['Scenario'] == sce) &
                    (df['Year'] == comp['year_map'][sce]) &
                    (df['Result Variable'] == comp['result_map'][sce])
                ]
            ], axis=0, ignore_index=True)
        df_graph = df_graph.replace({'Scenario': comp['name_map']})
        fig = plot_load_comparison(
            df=df_graph,
            color_col='Scenario',
            dash_col='Scenario',
            color_dict=comp['color_map_name'],
            line_dict=comp['line_map_name'],
            title='Load Shape',
            xaxis_title='Representative Day of Month',
            yaxis_title='GW',
        )
        fig.write_image(FIGURES_PATH / f"load_shape_comparison_{i}.pdf")


def plot_load_comparison(df, color_col, dash_col, color_dict, line_dict, title, xaxis_title, yaxis_title):
    fig = px.line(
        df,
        x='Hour',
        y='Value',
        color=color_col,
        color_discrete_map=color_dict,
        line_dash=dash_col,
        line_dash_map=line_dict,
    )
    fig = update_titles(fig, title, xaxis_title, yaxis_title)
    fig = update_legend_layout(fig, xaxis_title)
    fig = update_plot_size(fig)
    fig = update_to_load_shape_layout(fig)
    return fig


def update_to_load_shape_layout(fig):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=np.arange(12, 289, 24),
            ticktext=months,
            showgrid=False,
            minor=dict(
                tickvals=np.arange(0, 289, 24),
                showgrid=True,
                gridcolor='#FFFFFF'
            )
        )
    )
    month_ends = np.arange(0, 289, 24)
    for i, (x0, x1) in enumerate(zip(month_ends, month_ends[1:])):
        if i % 2 == 0:
            continue
        else:
            fig.add_vrect(x0=x0, x1=x1, line_width=0, fillcolor='grey', opacity=0.1)
    return fig


def sum_load_across_branches(df_in):
    df = pd.DataFrame(columns=['Year', 'Hour', 'Scenario', 'Result Variable', 'Value'])

    yrs = df_in['Year'].unique()
    hrs = df_in['Hour'].unique()
    scenarios = df_in['Scenario'].unique()
    results_vars = df_in['Result Variable'].unique()

    for yr, hr, sce, res, in itertools.product(yrs, hrs, scenarios, results_vars):
        df_to_add = df_in[
            (df_in['Year'] == yr) &
            (df_in['Hour'] == hr) &
            (df_in['Scenario'] == sce) &
            (df_in['Result Variable'] == res)
        ]
        if len(df_to_add.index) > 0:
            df.loc[len(df.index), :] = yr, hr, sce, res, df_to_add['Value'].sum(axis=0)

    return df


def form_egen_branch_maps(df):
    egen_res_map = {
        'Biogas': [],
        'Biomass': [],
        'Coal': [],
        'Geothermal': [],
        'H2 Fuel Cell': [],
        'Hydro': [],
        'Li Ion': [],
        'Natural Gas': [],
        'Natural Gas CCS': [],
        'Nuclear': [],
        'Solar': [],
        'Unspecified': [],
        'Wind': [],
    }

    egen_branches = [col for col in df.columns if "Transformation\Electricity Production" in col]

    for branch in egen_branches:
        if ('landfill' in branch.lower()) or ('manure' in branch.lower()) or ('wwtp' in branch.lower()) or (
                'food' in branch.lower()):
            egen_res_map['Biogas'].append(branch)
        elif ('biomass' in branch.lower()) or ('solid waste' in branch.lower()):
            egen_res_map['Biomass'].append(branch)
        elif 'coal' in branch.lower():
            egen_res_map['Coal'].append(branch)
        elif 'geothermal' in branch.lower():
            egen_res_map['Geothermal'].append(branch)
        elif 'hydrogen fuel cell' in branch.lower():
            egen_res_map['H2 Fuel Cell'].append(branch)
        elif 'hydro' in branch.lower():
            egen_res_map['Hydro'].append(branch)
        elif 'li ion' in branch.lower():
            egen_res_map['Li Ion'].append(branch)
        elif 'gas css' in branch.lower():
            egen_res_map['Natural Gas CCS'].append(branch)
        elif ('natural gas' in branch.lower()) or ('ng' in branch.lower()):
            egen_res_map['Natural Gas'].append(branch)
        elif 'nuclear' in branch.lower():
            egen_res_map['Nuclear'].append(branch)
        elif 'solar' in branch.lower():
            egen_res_map['Solar'].append(branch)
        elif 'unspecified' in branch.lower():
            egen_res_map['Unspecified'].append(branch)
        elif 'wind' in branch.lower():
            egen_res_map['Wind'].append(branch)
        else:
            print(f"Branch: {branch} not assigned")

    egen_res_map_short = {
        'Other': egen_res_map['Coal'] + egen_res_map['Geothermal'] + egen_res_map['Nuclear'] + egen_res_map[
            'Unspecified'] + egen_res_map['Biogas'] + egen_res_map['Biomass'],
        'H2 Fuel Cell': egen_res_map['H2 Fuel Cell'],
        'Li Ion': egen_res_map['Li Ion'],
        'Hydro': egen_res_map['Hydro'],
        'Natural Gas': egen_res_map['Natural Gas'],
        'Natural Gas CCS': egen_res_map['Natural Gas CCS'],
        'Solar': egen_res_map['Solar'],
        'Wind': egen_res_map['Wind'],
    }

    return egen_res_map, egen_res_map_short


def form_sector_branch_map(df):
    id_cols = ["Year", "Scenario", "Result Variable", "Fuel"]

    sector_map = {
        'Industry': [],
        'Electricity': [],
        'Buildings': [],
        'Agriculture': [],
        'Transportation': [],
        'Resources': [],
        'Incentives': [],
    }

    for branch in list(set(df.columns) - set(id_cols)):
        if (
                ('Demand\Residential' in branch) or
                ('Demand\Commercial' in branch) or
                ('Non Energy\Residential' in branch) or
                ('Non Energy\Commercial' in branch)
        ):
            sector_map['Buildings'].append(branch)
        elif (
                ('Demand\Transportation' in branch) or
                ('Non Energy\Transportation' in branch)
        ):
            sector_map['Transportation'].append(branch)
        elif (
                ('Demand\Agriculture' in branch) or
                ('Non Energy\Agriculture' in branch)
        ):
            sector_map['Agriculture'].append(branch)
        elif (
                ('Demand\Industry' in branch) or
                ('Transformation\Ethanol' in branch) or
                ('Transformation\Biodiesel' in branch) or
                ('Transformation\Refinery' in branch) or
                ('Transformation\Renewable Diesel' in branch) or
                ('Transformation\Crude Oil' in branch) or
                ('Transformation\Steam Gen' in branch) or
                ('Transformation\Hydrogen' in branch) or
                ('Transformation\CNG' in branch) or
                ('Transformation\CRNG' in branch) or
                ('Transformation\RNG' in branch) or
                ('NG Compressors' in branch) or
                ('Non Energy\Industry' in branch) or
                ('Non Energy\Carbon Removal\Industry' in branch) or
                ('Non Energy\Carbon Removal\DAC' in branch)
        ):
            sector_map['Industry'].append(branch)

        elif (
                ('Transformation\Electricity' in branch) or
                ('Non Energy\Electricity' in branch) or
                ('Transformation\Distributed PV' in branch) or
                ('Carbon Removal\Electricity Production' in branch)
        ):
            sector_map['Electricity'].append(branch)
        elif (
                ('Resources\\' in branch)
        ):
            sector_map['Resources'].append(branch)
        elif (
                ('Non Energy\Incentives' in branch)
        ):
            sector_map['Incentives'].append(branch)
        else:
            print(f"branch: {branch} not added to mapping")

    return sector_map


def plot_bar_subgroup_over_time(df, title, xaxis_title, yaxis_title, color_map, yaxis_col='Value',
                                xaxis_col='Year', color_col='Subgroup', include_sum=True):
    fig = px.bar(
        df,
        x=xaxis_col,
        y=yaxis_col,
        color=color_col,
        color_discrete_map=color_map,
    )

    # xaxis is a unit of time
    # yaxis is a value (eg emissions)
    if include_sum:
        df_sum = pd.DataFrame(columns=[xaxis_col, yaxis_col])
        for t in df[xaxis_col].unique():
            sum_in_t = df[df[xaxis_col] == t][yaxis_col].sum()
            df_sum.loc[len(df_sum.index)] = [t, sum_in_t]
        # add line to graph showing sum
        fig.add_trace(go.Scatter(
            mode='lines',
            x=df_sum[xaxis_col],
            y=df_sum[yaxis_col],
            name="Total",
            showlegend=True,
            line=dict(
                color='black',
                dash='solid',
            )
        ))

    fig = update_titles(fig, title, xaxis_title, yaxis_title)
    fig = update_legend_layout(fig, xaxis_title)
    fig = update_plot_size(fig)
    return fig


def plot_area_subgroup_over_time(df, title, xaxis_title, yaxis_title, color_map, yaxis_col='Value',
                                xaxis_col='Year', color_col='Subgroup', include_sum=True):
    fig = px.area(
        df,
        x=xaxis_col,
        y=yaxis_col,
        color=color_col,
        color_discrete_map=color_map,
    )

    # xaxis is a unit of time
    # yaxis is a value (eg emissions)
    if include_sum:
        df_sum = pd.DataFrame(columns=[xaxis_col, yaxis_col])
        for t in df[xaxis_col].unique():
            sum_in_t = df[df[xaxis_col] == t][yaxis_col].sum()
            df_sum.loc[len(df_sum.index)] = [t, sum_in_t]
        # add line to graph showing sum
        fig.add_trace(go.Scatter(
            mode='lines',
            x=df_sum[xaxis_col],
            y=df_sum[yaxis_col],
            name="Total",
            showlegend=True,
            line=dict(
                color='black',
                dash='solid',
            )
        ))

    fig = update_titles(fig, title, xaxis_title, yaxis_title)
    fig = update_legend_layout(fig, xaxis_title)
    fig = update_plot_size(fig)
    return fig


def plot_bar_scenario_comparison(df, title, xaxis_title, yaxis_title, color_dict, color_column='Subgroup',
                                 include_legend=False):

    fig = px.bar(
        df,
        x='Value',
        y='Scenario',
        color=color_column,
        color_discrete_map=color_dict,
    )
    if include_legend:
        fig = update_legend_layout(fig, xaxis_title)
    else:
        fig.update_layout(showlegend=False)
    fig = update_titles(fig, title, xaxis_title, yaxis_title)
    fig = update_plot_size(fig)
    return fig


def plot_scatter_scenario_comparison(df, title, xaxis_title, yaxis_title, sce_comp):
    fig = go.Figure()

    for sce in sce_comp['scenarios']:
        df_sce = df[df['Scenario'] == sce].copy()
        fig.add_trace(go.Scatter(
            mode='markers',
            x=df_sce['xval'],
            y=df_sce['yval'],
            name=sce_comp['name_map'][sce],
            showlegend=sce_comp['legend_map'][sce],
            marker_symbol=sce_comp['marker_map'][sce],
            marker_color=sce_comp['color_map'][sce],
        ))

    fig.update_traces(marker={'size': 10})
    fig = update_titles(fig, title, xaxis_title, yaxis_title)
    fig = update_legend_layout(fig, xaxis_title)
    fig = update_plot_size(fig)
    return fig


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

    fig = update_titles(fig, title, xaxis_title, yaxis_title)
    fig = update_legend_layout(fig, xaxis_title)
    fig = update_plot_size(fig)
    return fig


def update_titles(fig, title, xaxis_title, yaxis_title):
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )
    return fig


def update_to_tall_fig(fig, include_legend=True):
    fig.update_layout(
        autosize=False,
        height=1500,
    )

    if include_legend:
        fig.update_layout(
            legend=dict(
                orientation='h',
                yanchor='top',
                y=-0.05,
                xanchor='left',
                x=0,
            )
        )
    fig.update_yaxes(automargin=True)
    return fig


def update_legend_layout(fig, xaxis_title):
    if xaxis_title == '':
        y = -.08
    else:
        y=-0.2

    fig.update_layout(
        legend_title='',
        legend=dict(
            orientation='h',
            yanchor='top',
            y=y,
            xanchor='left',
            x=0,
        )
    )
    return fig


def update_plot_size(fig):
    fig.update_layout(
        autosize=False,
        width=800,
        height=500,
    )
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)
    return fig


if __name__ == "__main__":
    CLEAN_RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    main()