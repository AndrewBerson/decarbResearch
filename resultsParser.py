import pandas as pd
import itertools
from pathlib import Path
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from tqdm import tqdm

# Paths
INPUT_PATH = Path("resultsFiles/apr27_2023")
CONTROLLER_PATH = INPUT_PATH / "results_controller"
CLEAN_RESULTS_PATH = INPUT_PATH / "clean_results"
FIGURES_PATH = INPUT_PATH / "figures"

# Years, discount rate, capital recovery factor
BASE_YEAR = 2018
END_YEAR = 2045
TOTAL_YEARS = END_YEAR - BASE_YEAR + 1
DISCOUNT_RATE = 0.05
CAPITAL_RECOVERY_FACTOR = (DISCOUNT_RATE * ((1 + DISCOUNT_RATE) ** TOTAL_YEARS)) / \
                          (((1 + DISCOUNT_RATE) ** TOTAL_YEARS) - 1)

# LEAP result strings
EMISSIONS_RESULT_STRING = "One_Hundred Year GWP Direct At Point of Emissions"
COST_RESULT_STRING = "Social Costs"
GENERATION_STRING = "Outputs by Output Fuel"
CAPACITY_ADDED_STRING = "Capacity Added"
ENERGY_DEMAND_STRING = "Energy Demand Final Units"
INPUTS_STRING = "Inputs"

# Fuels
FUELS_TO_COMBINE = {
    "CRNG": "RNG",
    "CNG": "NG",
    "Hydrogen Transmitted": "Hydrogen"
}
RESOURCE_PROXY = {
    'Scenario': ['Today', 'Today', 'Resource Proxy', 'Resource Proxy'],
    'Fuel': ['RNG', 'Renewable Diesel', 'RNG', 'Renewable Diesel'],
    'Value': [55, 95, 455, 285],
}


def main():
    reload_results = False    # set to True if using a new raw results excel document

    # load data and make copies of scenarios specified in the controller
    df = load_data(reload=reload_results)
    df = create_scenario_copies(df)
    df_loads = load_load_shapes(reload=reload_results)
    df_loads = create_scenario_copies(df_loads)

    # create color and branch maps
    color_maps = load_color_maps()        # keys are egen_resources, sectors, fuels
    branch_maps = form_branch_maps(df)    # keys are the column names in the controller

    # make graphs of comparisons between scenarios
    base_scenario_comparison_graphs(df, color_maps, branch_maps)

    # make graphs exploring individual scenarios
    individual_scenario_graphs(df, color_maps, branch_maps)

    # load shape graphs
    load_shape_graphs(df_loads, color_maps)

    # Sensitivity graphs
    sensitivity_graphs(df, color_maps, branch_maps)

    # TODO: RPS graphs


def base_scenario_comparison_graphs(df, color_maps, branch_maps):
    """ Function to generate the graphs of all of the 'base' comparisons between scenarios """

    relevant_scenarios, scenario_comparison_params = load_sce_comps()
    df_scenario_comp = df[df['Scenario'].isin(relevant_scenarios)].copy()

    graph_emissions_over_time_scenario_comparisons(
        df_in=df_scenario_comp,
        scenario_comparison_params=scenario_comparison_params,
        branch_map=branch_maps['all_branches_together'],
        # branch_map=branch_maps['buildings_only'],
    )
    graph_marginal_emissions_over_time_scenario_comparisons(
        df_in=df_scenario_comp,
        scenario_comparison_params=scenario_comparison_params,
        branch_map=branch_maps['all_branches_together'],
    )
    graph_marginal_cost_over_time_scenario_comparisons(
        df_in=df_scenario_comp,
        scenario_comparison_params=scenario_comparison_params,
        branch_map=branch_maps['all_branches_together'],
    )
    graph_marginal_emissions_vs_marginal_cost_scatter_scenario_comparison(
        df_in=df_scenario_comp,
        scenario_comparison_params=scenario_comparison_params,
        branch_map=branch_maps['all_branches_together'],
    )
    graph_cost_of_co2_abatement_bar(
        df_in=df_scenario_comp,
        scenario_comparison_params=scenario_comparison_params,
        branch_map=branch_maps['all_branches_together'],
    )
    graph_egen_by_resource_scenario_comparison(
        df_in=df_scenario_comp,
        scenario_comparison_params=scenario_comparison_params,
        color_map=color_maps['egen_resources'],
        branch_map=branch_maps['egen_resources_long'],
        file_suffix='long',
    )
    graph_egen_by_resource_scenario_comparison(
        df_in=df_scenario_comp,
        scenario_comparison_params=scenario_comparison_params,
        color_map=color_maps['egen_resources'],
        branch_map=branch_maps['egen_resources_short'],
        file_suffix='short',
    )
    graph_cumulative_marginal_costs_by_sector_scenario_comparison(
        df_in=df_scenario_comp,
        scenario_comparison_params=scenario_comparison_params,
        color_map=color_maps['sectors'],
        branch_map=branch_maps['sectors'],
        year=2045,
    )
    graph_cumulative_marginal_abated_emissions_by_sector_scenario_comparison(
        df_in=df_scenario_comp,
        scenario_comparison_params=scenario_comparison_params,
        color_map=color_maps['sectors'],
        branch_map=branch_maps['sectors'],
        year=2045,
    )
    graph_annual_marginal_abated_emissions_by_sector_scenario_comparison(
        df_in=df_scenario_comp,
        scenario_comparison_params=scenario_comparison_params,
        color_map=color_maps['sectors'],
        branch_map=branch_maps['sectors'],
        year=2045,
    )
    graph_energy_demand_by_fuel_scenario_comparison_bar(
        df_in=df_scenario_comp,
        scenario_comparison_params=scenario_comparison_params,
        color_map=color_maps['fuels'],
        branch_map=branch_maps['all_branches_together'],
        fuels=["Renewable Diesel", "RNG"],
        year=2045,
        include_proxy=True,
    )


def individual_scenario_graphs(df, color_maps, branch_maps):

    relevant_scenarios, individual_sce_graph_params = load_individual_scenarios()
    df_individual_scenarios = df[df['Scenario'].isin(relevant_scenarios)]

    graph_marginal_costs_by_sector_over_time(
        df_in=df_individual_scenarios,
        individual_sce_graph_params=individual_sce_graph_params,
        color_map=color_maps['sectors'],
        branch_map=branch_maps['sectors'],
    )
    graph_marginal_emissions_by_sector_over_time(
        df_in=df_individual_scenarios,
        individual_sce_graph_params=individual_sce_graph_params,
        color_map=color_maps['sectors'],
        branch_map=branch_maps['sectors'],
        # branch_map=branch_maps['buildings_only_fgas_separate']
    )
    graph_emissions_by_sector_over_time(
        df_in=df_individual_scenarios,
        individual_sce_graph_params=individual_sce_graph_params,
        color_map=color_maps['sectors'],
        branch_map=branch_maps['sectors'],
        # branch_map=branch_maps['sectors_dac_separate_no_incentives_no_resources'],
        # branch_map=branch_maps['buildings_only_fgas_separate']
    )
    graph_egen_by_resource_over_time(
        df_in=df_individual_scenarios,
        individual_sce_graph_params=individual_sce_graph_params,
        color_map=color_maps['egen_resources'],
        branch_map=branch_maps['egen_resources_short'],
        suffix='short'
    )
    graph_cumulative_egen_capacity_added_over_time(
        df_in=df_individual_scenarios,
        individual_sce_graph_params=individual_sce_graph_params,
        color_map=color_maps['egen_resources'],
        branch_map=branch_maps['egen_resources_short'],
        suffix='short'
    )
    graph_energy_demand_by_fuel_over_time(
        df_in=df_individual_scenarios,
        individual_sce_graph_params=individual_sce_graph_params,
        color_map=color_maps['fuels'],
        branch_map=branch_maps['fuel_use'],
        fuels=["Renewable Diesel", "RNG"],
        # fuels=None    # set Fuels to none if you want to include all fuels
    )


def load_shape_graphs(df_loads, color_maps):
    load_scenarios_to_compare, load_scenario_comparison_params = load_load_comps()
    df_load_comparison = df_loads[df_loads['Scenario'].isin(load_scenarios_to_compare)]
    graph_load_comparison(
        df_in=df_load_comparison,
        comp_params=load_scenario_comparison_params,
    )

    individual_load_scenarios, individual_load_params = load_individual_load_params()
    df_individual_loads = df_loads[df_loads['Scenario'].isin(individual_load_scenarios)]
    graph_load_by_sector(
        df_in=df_individual_loads,
        params=individual_load_params,
        color_map=color_maps['sectors'],
    )


def sensitivity_graphs(df, color_maps, branch_maps):
    tech_choice_relevant_scenarios, tech_choice_graph_params = load_tech_choice_graph_params()
    graph_tech_choice_emissions(
        df_in=df[df['Scenario'].isin(tech_choice_relevant_scenarios)],
        tech_choice_graph_params=tech_choice_graph_params,
        stacked=False,
        branch_map=branch_maps['all_branches_together'],
        color_map=color_maps['sectors'],
        year=2045,
    )
    # stacked bar chart (stacked by sector
    graph_tech_choice_emissions(
        df_in=df[df['Scenario'].isin(tech_choice_relevant_scenarios)],
        tech_choice_graph_params=tech_choice_graph_params,
        stacked=True,
        branch_map=branch_maps['sectors'],
        color_map=color_maps['sectors'],
        year=2045,
    )
    graph_tech_choice_marginal_cost(
        df_in=df[df['Scenario'].isin(tech_choice_relevant_scenarios)],
        tech_choice_graph_params=tech_choice_graph_params,
        stacked=False,
        branch_map=branch_maps['all_branches_together'],
        color_map=color_maps['sectors'],
        year=2045,
    )
    # stacked bar chart (stacked by sector)
    graph_tech_choice_marginal_cost(
        df_in=df[df['Scenario'].isin(tech_choice_relevant_scenarios)],
        tech_choice_graph_params=tech_choice_graph_params,
        stacked=True,
        branch_map=branch_maps['sectors'],
        color_map=color_maps['sectors'],
        year=2045,
    )
    graph_tech_choice_cost_of_abatement(
        df_in=df[df['Scenario'].isin(tech_choice_relevant_scenarios)],
        tech_choice_graph_params=tech_choice_graph_params,
        branch_map=branch_maps['all_branches_together'],
        color_map=color_maps['sectors'],
    )


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
    i = 0
    for fname in os.listdir(input_path):
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


def create_scenario_copies(df):
    """ Function to create copies of specified scenarios under a new name"""
    df_excel = pd.read_excel(CONTROLLER_PATH / 'controller.xlsx', sheet_name="scenario_copies")

    sce_copy_dict = dict(zip(df_excel['Copy Name'], df_excel['Original Scenario']))
    sce_instate_incentive_on_off_dict = dict(zip(df_excel['Copy Name'], df_excel['In State Incentives']))

    for new_name, original_name in sce_copy_dict.items():
        df_to_add = df[df['Scenario'] == original_name].copy()
        df_to_add['Scenario'] = new_name
        df = pd.concat([df, df_to_add], axis=0)

    return remove_instate_incentives(df, sce_instate_incentive_on_off_dict)


def remove_instate_incentives(df, scenario_dict):

    scenarios_to_remove_incentives = [sce for sce, on_off in scenario_dict.items() if on_off.lower() == 'off']
    relevant_columns = [col for col in df.columns if 'Non Energy\\Incentives' in col]

    df.loc[
        (df['Scenario'].isin(scenarios_to_remove_incentives)) &
        (df['Result Variable'] == COST_RESULT_STRING),
        relevant_columns
    ] = 0

    return df


def form_branch_maps(df_results):

    # for set of all branches included in the results
    id_cols = ["Year", "Scenario", "Result Variable", "Fuel"]
    all_branches = set(df_results.columns) - set(id_cols)

    df = pd.read_excel(CONTROLLER_PATH / 'controller.xlsx', sheet_name="branch_maps")

    # check if there are any branches missing from the controller
    missing_branches = list(all_branches - set(df['Branch'].unique()))
    if len(missing_branches) > 0:
        print(f"Branches not included in controller: {missing_branches}")

    # form maps of branches
    branch_maps = dict()
    maps = df.columns.tolist()
    maps.remove('Branch')

    # iterate through columns in the controller
    for map_name in maps:
        branch_maps[map_name] = dict()

        df_map = df[['Branch'] + [map_name]].copy()

        # map unique sector (or other variable) to relevant branches
        for key, dfg in df_map.groupby(map_name):
            if key == False:
                continue
            branch_maps[map_name][key] = dfg['Branch'].tolist()

    return branch_maps


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


def load_sce_comps():
    """ Function to load scenario comparisons as dictated in controller """
    df = pd.read_excel(CONTROLLER_PATH / 'controller.xlsx', sheet_name="base_scenario_comparisons")

    relevant_scenarios = set(df['Scenario'].unique())
    relevant_scenarios.update(set(df['Relative to'].unique()))

    # Generate list of dictionaries storing parameters for each group of scenarios that will be graphed
    scenario_comp_params = []
    for _, dfg in df.groupby('Group'):
        params = dict()

        # scenarios included in the group
        params['scenarios'] = dfg['Scenario'].tolist()
        params['relevant_scenarios'] = list(set(dfg['Scenario'].tolist() + dfg['Relative to'].tolist()))

        # dictionaries mapping scenario --> name, line, etc.
        params['name_map'] = dict(zip(dfg['Scenario'], dfg['Naming']))
        params['line_map'] = dict(zip(dfg['Scenario'], dfg['Line']))
        params['color_map'] = dict(zip(dfg['Scenario'], dfg['Color']))
        params['legend_map'] = dict(zip(dfg['Scenario'], dfg['Include in legend']))
        params['marker_map'] = dict(zip(dfg['Scenario'], dfg['Marker']))

        # dictionaries mapping name --> color, line, etc.
        params['line_map_by_name'] = dict(zip(dfg['Naming'], dfg['Line']))
        params['color_map_by_name'] = dict(zip(dfg['Naming'], dfg['Color']))
        params['legend_map_by_name'] = dict(zip(dfg['Naming'], dfg['Include in legend']))
        params['marker_map_by_name'] = dict(zip(dfg['Naming'], dfg['Marker']))

        # scenario to marginalize relative to
        params['relative_to_map'] = dict(zip(dfg['Scenario'], dfg['Relative to']))

        # Load parameter dictating whether each specific graph is generated
        params['emissions_over_time'] = dfg['emissions over time'].unique()[0]
        params['marginal_emissions_over_time'] = dfg['marginal emissions over time'].unique()[0]
        params['marginal_cost_over_time'] = dfg['marginal cost over time'].unique()[0]
        params['marginal_emissions_vs_marginal_cost'] = dfg['marginal emissions vs marginal cost'].unique()[0]
        params['cost_of_co2_abatement'] = dfg['cost of co2 abatement'].unique()[0]
        params['egen_by_resource'] = dfg['egen by resource'].unique()[0]
        params['cumulative_marginal_cost_by_sector'] = dfg['cumulative marginal cost by sector'].unique()[0]
        params['cumulative_marginal_abated_emissions_by_sector'] = dfg['cumulative marginal abated emissions by sector'].unique()[0]
        params['annual_marginal_abated_emissions_by_sector'] = dfg['annual marginal abated emissions by sector'].unique()[0]
        params['energy_demand_by_fuel'] = dfg['energy demand by fuel'].unique()[0]

        scenario_comp_params.append(params)

    return list(relevant_scenarios), scenario_comp_params


def load_individual_scenarios():
    """ Function to load scenario comparisons as dictated in controller """
    df = pd.read_excel(CONTROLLER_PATH / 'controller.xlsx', sheet_name="individual_scenario_graphs")

    relevant_scenarios = set(df['Scenario'].unique())
    relevant_scenarios.update(set(df['Relative to'].unique()))

    individual_graph_params = []
    for key, dfg in df.groupby('id'):
        params = {
            'scenario': dfg['Scenario'].unique()[0],
            'relevant_scenarios': [dfg['Scenario'].unique()[0], dfg['Relative to'].unique()[0]],
            'id': dfg['id'].unique()[0],
            'name': dfg['Naming'].unique()[0],
            'relative_to_map': dict(zip(dfg['Scenario'], dfg['Relative to'])),

            # which graphs to generate
            'marginal_costs_by_sector': dfg['marginal costs by sector'].unique()[0],
            'marginal_emissions_by_sector': dfg['marginal emissions by sector'].unique()[0],
            'emissions_by_sector': dfg['emissions by sector'].unique()[0],
            'egen_by_resource': dfg['egen by resource'].unique()[0],
            'cumulative_egen_capacity_added': dfg['cumulative egen capacity added'].unique()[0],
            'energy_demand_by_fuel': dfg['energy demand by fuel'].unique()[0],
            'marginal_energy_demand_by_fuel': dfg['marginal energy demand by fuel'].unique()[0],
        }
        individual_graph_params.append(params)

    return relevant_scenarios, individual_graph_params


def load_load_comps():
    """ Function to load scenario comparisons as dictated in controller """

    # load data related to comparisons between different scenarios
    df = pd.read_excel(CONTROLLER_PATH / 'controller.xlsx', sheet_name="load_shape_comparisons")
    relevant_scenarios = set(df['Scenario'].unique())

    scenario_comp_params = []
    for _, dfg in df.groupby('Group'):
        params = dict()

        # list of scenarios being compared
        params['scenarios'] = dfg['Scenario'].tolist()

        # dicts of scenarios --> color, name...
        params['result_map'] = dict(zip(dfg['Scenario'], dfg['Result Variable']))
        params['year_map'] = dict(zip(dfg['Scenario'], dfg['Year']))
        params['name_map'] = dict(zip(dfg['Scenario'], dfg['Naming']))
        params['line_map'] = dict(zip(dfg['Scenario'], dfg['Line']))
        params['color_map'] = dict(zip(dfg['Scenario'], dfg['Color']))

        # dicts of name --> line, color...
        params['line_map_by_name'] = dict(zip(dfg['Naming'], dfg['Line']))
        params['color_map_by_name'] = dict(zip(dfg['Naming'], dfg['Color']))

        scenario_comp_params.append(params)

    return list(relevant_scenarios), scenario_comp_params,


def load_individual_load_params():

    # load data for graphs about single scenarios
    df = pd.read_excel(CONTROLLER_PATH / 'controller.xlsx', sheet_name="individual_load_shapes")
    relevant_scenarios = set(df['Scenario'].unique())

    individual_load_params = []
    for _, dfg in df.groupby('id'):
        params = {
            'scenario': dfg['Scenario'].unique()[0],
            'name': dfg['Naming'].unique()[0],
            'year': dfg['Year'].unique()[0],
            'result_var': dfg['Result Variable'].unique()[0],
            'name_map': dict(zip(dfg['Scenario'], dfg['Naming'])),
        }
        individual_load_params.append(params)

    return list(relevant_scenarios), individual_load_params


def load_tech_choice_graph_params():
    """ Function to load scenario comparisons as dictated in controller """
    df = pd.read_excel(CONTROLLER_PATH / 'controller.xlsx', sheet_name="tech_choice_plots")

    relevant_scenarios = set(df['Scenario'].unique())
    relevant_scenarios.update(set(df['Relative to'].unique()))

    scenario_comp_params = []
    for _, dfg in df.groupby('Plot'):
        params = dict()
        params['scenarios'] = dfg['Scenario'].tolist()
        params['relevant_scenarios'] = list(set(dfg['Scenario'].tolist() + dfg['Relative to'].tolist()))
        params['relative_to_map'] = dict(zip(dfg['Scenario'], dfg['Relative to']))
        params['name_map'] = dict(zip(dfg['Scenario'], dfg['Naming']))
        params['sector_map'] = dict(zip(dfg['Scenario'], dfg['Sector']))
        scenario_comp_params.append(params)

    return list(relevant_scenarios), scenario_comp_params


def load_color_maps():
    """ Function to load color maps from controller """
    color_maps = dict()

    df = pd.read_excel(CONTROLLER_PATH / 'controller.xlsx', sheet_name="egen_resource_colors")
    color_maps['egen_resources'] = dict(zip(df['Resource'], df['Color']))

    df = pd.read_excel(CONTROLLER_PATH / 'controller.xlsx', sheet_name="sector_colors")
    color_maps['sectors'] = dict(zip(df['Sector'], df['Color']))

    df = pd.read_excel(CONTROLLER_PATH / 'controller.xlsx', sheet_name="fuel_colors")
    color_maps['fuels'] = dict(zip(df['Fuel'], df['Color']))

    return color_maps


def calculate_annual_result_by_subgroup(df_in, result_str, subgroup_dict):
    """ Function to calculate annual result summed branch subgroups for all scenarios """

    df_out = pd.DataFrame(columns=['Year', 'Scenario', 'Fuel', 'Subgroup', 'Value'])

    # convert result_str to list so that multiple result_strings can be passed into the function as a list
    # note this is useful for energy demand and inputs
    if type(result_str) == str:
        result_str = [result_str]

    for key, dfg in df_in[df_in['Result Variable'].isin(result_str)].groupby(by=['Year', 'Scenario', 'Fuel']):
        yr, sce, fuel = key
        mask = np.array(
            (dfg['Year'] == yr) &
            (dfg['Scenario'] == sce) &
            (dfg['Fuel'] == fuel)
        )
        row_ids = list(np.where(mask)[0])
        for subgroup, branches in subgroup_dict.items():
            value = dfg[branches].iloc[row_ids].sum(axis=1).sum()
            df_out.loc[len(df_out.index)] = [yr, sce, fuel, subgroup, value]

    return df_out


def marginalize_it(df_in, relative_to_dict):

    df_out = df_in.copy()

    # iterate through scenarios and what they're being marginalized against in the relative_to_dict
    for sce, relative_to in relative_to_dict.items():

        # find all relevant subgroups, years, and fuels for the scenario
        subgroups = df_out[df_out['Scenario'] == sce]['Subgroup'].unique()
        years = df_out[df_out['Scenario'] == sce]['Year'].unique()
        fuels = df_out[df_out['Scenario'] == sce]['Fuel'].unique()

        for subg, yr, fuel in itertools.product(subgroups, years, fuels):

            # subtract out the scenario that it's being marginalized relative to
            df_out.loc[
                (df_out['Scenario'] == sce) &
                (df_out['Subgroup'] == subg) &
                (df_out['Fuel'] == fuel) &
                (df_out['Year'] == yr),
                'Value'
            ] -= float(
                df_in.loc[
                    (df_in['Scenario'] == relative_to) &
                    (df_in['Subgroup'] == subg) &
                    (df_in['Fuel'] == fuel) &
                    (df_in['Year'] == yr),
                    'Value'
                ]
            )

    return df_out


def discount_it(df_in):
    df = df_in.copy()
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


def cumsum_it(df_in):
    df = df_in.copy()
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


def evaluate_cumulative_marginal_emissions_cumulative_marginal_cost(df_in, subgroup_dict, relative_to_map):
    df_cost = calculate_annual_result_by_subgroup(df_in, COST_RESULT_STRING, subgroup_dict)
    df_cost = marginalize_it(df_cost, relative_to_map)
    df_cost = discount_it(df_cost)
    df_cost = cumsum_it(df_cost)
    df_cost = df_cost.rename(columns={'Value': 'cumulative_marginal_cost'})

    df_emissions = calculate_annual_result_by_subgroup(df_in, EMISSIONS_RESULT_STRING, subgroup_dict)
    df_emissions = marginalize_it(df_emissions, relative_to_map)
    df_emissions = cumsum_it(df_emissions)
    df_emissions = df_emissions.rename(columns={'Value': 'cumulative_marginal_emissions'})

    df = df_emissions.merge(df_cost, how='outer', on=['Scenario', 'Subgroup', 'Year'])

    return df


def evaluate_dollar_per_ton_abated(df_in, subgroup_dict, relative_to_map):
    df = evaluate_cumulative_marginal_emissions_cumulative_marginal_cost(df_in, subgroup_dict, relative_to_map)
    df = df[df['Year'] == END_YEAR].copy()
    df['annualized_cost'] = df['cumulative_marginal_cost'] * CAPITAL_RECOVERY_FACTOR
    df['annualized_emissions_reduction'] = -1 * df['cumulative_marginal_emissions'] / TOTAL_YEARS
    df['cost_of_abatement'] = df['annualized_cost'] / df['annualized_emissions_reduction']

    return df


def graph_emissions_over_time_scenario_comparisons(df_in, scenario_comparison_params, branch_map):

    df = calculate_annual_result_by_subgroup(df_in, EMISSIONS_RESULT_STRING, branch_map)
    df['Value'] = df['Value'] / 1e6

    for i, params in enumerate(scenario_comparison_params):
        if params['emissions_over_time']:
            fig = plot_line_scenario_comparison_over_time(
                df=df,
                title='Scenario Emissions',
                yaxis_title='Annual Emissions (Mt CO2e)',
                xaxis_title='',
                sce_comp=params,
            )
            fig.update_yaxes(rangemode="tozero")
            fig.write_image(FIGURES_PATH / f"emissions_over_time{i}.pdf")


def graph_marginal_emissions_over_time_scenario_comparisons(df_in, scenario_comparison_params, branch_map):

    df = calculate_annual_result_by_subgroup(df_in, EMISSIONS_RESULT_STRING, branch_map)
    df['Value'] = df['Value'] / 1e6

    for i, params in enumerate(scenario_comparison_params):
        if params['marginal_emissions_over_time']:
            df_graph = marginalize_it(df, params['relative_to_map'])
            fig = plot_line_scenario_comparison_over_time(
                df=df_graph,
                title='',
                yaxis_title='Change in Annual Emissions (Mt CO2e)',
                xaxis_title='',
                sce_comp=params,
            )
            fig.write_image(FIGURES_PATH / f"marginal_emissions_over_time_{i}.pdf")


def graph_marginal_cost_over_time_scenario_comparisons(df_in, scenario_comparison_params, branch_map):

    # Evaluate cost in Billions of Dollars
    df = calculate_annual_result_by_subgroup(df_in, COST_RESULT_STRING, branch_map)
    df['Value'] = df['Value'] / 1e9

    for i, params in enumerate(scenario_comparison_params):
        if params['marginal_cost_over_time']:

            # marginalize the cost
            df_graph = marginalize_it(df, params['relative_to_map'])
            fig = plot_line_scenario_comparison_over_time(
                df=df_graph,
                title='Scenario Marginal Costs',
                yaxis_title='$/yr (Billion)',
                xaxis_title='',
                sce_comp=params,
            )
            fig.write_image(FIGURES_PATH / f"marginal_cost_over_time_{i}.pdf")


def graph_marginal_emissions_vs_marginal_cost_scatter_scenario_comparison(df_in, scenario_comparison_params,
                                                                          branch_map):

    for i, params in enumerate(scenario_comparison_params):
        if params['marginal_emissions_vs_marginal_cost']:
            # evaluate cumulative marginal emissions and cumulative marginal cost
            df = evaluate_cumulative_marginal_emissions_cumulative_marginal_cost(
                df_in=df_in[df_in['Scenario'].isin(params['relevant_scenarios'])],
                subgroup_dict=branch_map,
                relative_to_map=params['relative_to_map'],
            )

            # Only use the final year
            df = df[df['Year'] == END_YEAR].copy()

            # Units of Billion tonnes and Billion dollars
            df['cumulative_marginal_cost'] = df['cumulative_marginal_cost'] / 1e9
            df['cumulative_marginal_abated_emissions'] = -1 * df['cumulative_marginal_emissions'] / 1e9
            df = df.rename(columns={
                'cumulative_marginal_cost': 'yval',
                'cumulative_marginal_abated_emissions': 'xval'
            })

            fig = plot_scatter_scenario_comparison(
                df=df,
                title='Emissions vs Cost',
                xaxis_title='Marginal Abated Emissions (Gt CO2e)',
                yaxis_title='Marginal Cost ($B)',
                sce_comp=params,
            )
            fig.write_image(FIGURES_PATH / f"marginal_emissions_vs_marginal_cost{i}.pdf")


def graph_cost_of_co2_abatement_bar(df_in, scenario_comparison_params, branch_map):


    for i, params in enumerate(scenario_comparison_params):
        if params['cost_of_co2_abatement']:

            df = evaluate_dollar_per_ton_abated(
                df_in=df_in[df_in['Scenario'].isin(params['relevant_scenarios'])],
                subgroup_dict=branch_map,
                relative_to_map=params['relative_to_map']
            )

            df['Value'] = df['cost_of_abatement']

            # filter out irrelevant scenarios and rename scenarios
            df = df[df['Scenario'].isin(params['scenarios'])].copy()
            df = df.replace({'Scenario': params['name_map']})

            fig = plot_bar_scenario_comparison(
                df=df,
                title='Cost of Carbon Abatement',
                xaxis_title='$/t CO2e',
                yaxis_title='',
                color_dict=params['color_map_by_name'],
                color_column='Scenario',
                include_legend=False,
            )
            fig.write_image(FIGURES_PATH / f"cost_of_co2_abatement{i}.pdf")


def graph_egen_by_resource_scenario_comparison(df_in, scenario_comparison_params, color_map, branch_map,
                                               file_suffix, year=2045):

    df = calculate_annual_result_by_subgroup(df_in, GENERATION_STRING, branch_map)
    df = df[df['Year'] == year].copy()
    df['Value'] = df['Value'] / 1e9

    for i, params in enumerate(scenario_comparison_params):
        if params['egen_by_resource']:
            df_graph = df[df['Scenario'].isin(params['scenarios'])].copy()
            df_graph = df_graph.replace({'Scenario': params['name_map']})

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


def graph_cumulative_marginal_costs_by_sector_scenario_comparison(df_in, scenario_comparison_params,
                                                                  color_map, branch_map, year=END_YEAR):

    for i, params in enumerate(scenario_comparison_params):
        if params['cumulative_marginal_cost_by_sector']:

            # git rid of extraneous scenarios to cut down on compute time
            df = df_in[df_in['Scenario'].isin(params['relevant_scenarios'])].copy()

            # evaluate marginal cost
            df = evaluate_cumulative_marginal_emissions_cumulative_marginal_cost(df, branch_map, params['relative_to_map'])
            df = df.rename(columns={'cumulative_marginal_cost': 'Value'})

            df = df[df['Year'] == year].copy()          # only graphing result from a specified year
            df['Value'] = df['Value'] / 1e9             # units of Billions

            # only graph specified scenarios (does not necessarily include relative_to)
            df = df[df['Scenario'].isin(params['scenarios'])].copy()

            # rename scenarios
            df = df.replace({'Scenario': params['name_map']})

            # create figure
            fig = plot_bar_scenario_comparison(
                df=df,
                title=f'Cumulative Marginal Cost',
                xaxis_title='$B',
                yaxis_title='',
                color_dict=color_map,
                color_column='Subgroup',
                include_legend=True,
            )
            fig.write_image(FIGURES_PATH / f"cumulative_marginal_cost_by_sector_{i}.pdf")


def graph_cumulative_marginal_abated_emissions_by_sector_scenario_comparison(df_in, scenario_comparison_params,
                                                                             color_map, branch_map, year=END_YEAR):

    for i, params in enumerate(scenario_comparison_params):
        if params['cumulative_marginal_abated_emissions_by_sector']:

            df = evaluate_cumulative_marginal_emissions_cumulative_marginal_cost(
                df_in[df_in['Scenario'].isin(params['relevant_scenarios'])].copy(),
                branch_map,
                params['relative_to_map']
            )
            df['Value'] = -1 * df['cumulative_marginal_emissions'] / 1e9
            df = df[df['Year'] == year].copy()

            df = df[df['Scenario'].isin(params['scenarios'])].copy()
            df = df.replace({'Scenario': params['name_map']})

            fig = plot_bar_scenario_comparison(
                df=df,
                title='Cumulative Marginal Abated Emissions',
                xaxis_title='Gt CO2e',
                yaxis_title='',
                color_dict=color_map,
                color_column='Subgroup',
                include_legend=True,
            )
            fig.write_image(FIGURES_PATH / f"cumulative_marginal_emissions_by_sector_{i}.pdf")


def graph_annual_marginal_abated_emissions_by_sector_scenario_comparison(df_in, scenario_comparison_params, color_map,
                                                                         branch_map, year=END_YEAR):
    df = calculate_annual_result_by_subgroup(df_in, EMISSIONS_RESULT_STRING, branch_map)
    df['Value'] = -1 * df['Value'] / 1e6    # units of millions

    for i, params in enumerate(scenario_comparison_params):
        if params['annual_marginal_abated_emissions_by_sector']:
            df_graph = marginalize_it(df, params['relative_to_map'])
            df_graph = df_graph[df_graph['Year'] == year].copy()

            df_graph = df_graph[df_graph['Scenario'].isin(params['scenarios'])].copy()
            df_graph = df_graph.replace({'Scenario': params['name_map']})

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


def graph_energy_demand_by_fuel_scenario_comparison_bar(df_in, scenario_comparison_params, color_map, branch_map, fuels,
                                                        year=2045, include_proxy=True):

    df = calculate_annual_result_by_subgroup(df_in, [ENERGY_DEMAND_STRING, INPUTS_STRING], branch_map)
    df['Value'] = df['Value'] / 1e6
    df = df[df['Year'] == year]
    df = df.replace({"Fuel": FUELS_TO_COMBINE})

    if include_proxy:
        df = pd.concat([df, pd.DataFrame.from_dict(RESOURCE_PROXY)], axis=0, sort=True)

    if fuels == None:
        df = df[df['Fuel'] != "Other"].copy()
    else:
        df = df[df['Fuel'].isin(fuels)]

    df = df.drop(columns=['Year'])
    df = df.groupby(by=['Scenario', 'Fuel'], as_index=False).sum()

    for i, params in enumerate(scenario_comparison_params):
        if params['energy_demand_by_fuel']:
            relevant_scenarios = params['scenarios']
            if include_proxy:
                relevant_scenarios = relevant_scenarios + RESOURCE_PROXY['Scenario']
            df_graph = df[df['Scenario'].isin(relevant_scenarios)].copy()
            df_graph = df_graph.replace({'Scenario': params['name_map']})
            df_graph = df_graph.sort_values(by=['Scenario'])

            fig = plot_grouped_bar_scenario_comparison(
                df=df_graph,
                title=f'Energy Demand in {year}',
                xaxis_title='Scenario',
                yaxis_title='PJ',
                grouping_column='Scenario',
                color_column='Fuel',
                color_dict=color_map,
                include_legend=True,
            )
            fig.write_image(FIGURES_PATH / f"energy_demand_by_fuel_{i}.pdf")



def graph_marginal_costs_by_sector_over_time(df_in, individual_sce_graph_params,
                                             color_map, branch_map):

    df = calculate_annual_result_by_subgroup(df_in, COST_RESULT_STRING, branch_map)

    for params in individual_sce_graph_params:
        if params['marginal_costs_by_sector']:
            df_graph = df[df['Scenario'].isin(params['relevant_scenarios'])].copy()

            df_graph = marginalize_it(df_graph, params['relative_to_map'])
            df_graph['Value'] = df_graph['Value'] / 1e9

            name = params['name']

            df_graph = df_graph[df_graph['Scenario'] == params['scenario']].copy()
            fig = plot_bar_subgroup_over_time(
                df=df_graph,
                title=f'Marginal Cost by Sector<br>{name}',
                xaxis_title='',
                yaxis_title='$B',
                color_map=color_map,
                include_sum=True,
            )
            fig.write_image(FIGURES_PATH / f"marginal_cost_over_time_by_sector_{params['id']}.pdf")


def graph_marginal_emissions_by_sector_over_time(df_in, individual_sce_graph_params, color_map, branch_map):

    df = calculate_annual_result_by_subgroup(df_in, EMISSIONS_RESULT_STRING, branch_map)

    for params in individual_sce_graph_params:
        if params['marginal_emissions_by_sector']:

            df_graph = df[df['Scenario'].isin(params['relevant_scenarios'])].copy()

            df_graph = marginalize_it(df_graph, params['relative_to_map'])
            df_graph = df_graph[df_graph['Scenario'] == params['scenario']].copy()

            df_graph['Value'] = df_graph['Value'] / 1e6

            name = params['name']

            fig = plot_bar_subgroup_over_time(
                df=df_graph,
                title=f'Marginal Emissions by Sector<br>{name}',
                xaxis_title='',
                yaxis_title='Mt CO2e',
                color_map=color_map,
                include_sum=True,
            )
            fig.write_image(FIGURES_PATH / f"marginal_emissions_by_sector_over_time_{params['id']}.pdf")


def graph_emissions_by_sector_over_time(df_in, individual_sce_graph_params, color_map, branch_map):

    df = calculate_annual_result_by_subgroup(df_in, EMISSIONS_RESULT_STRING, branch_map)
    df['Value'] = df['Value'] / 1e6

    for params in individual_sce_graph_params:
        if params['emissions_by_sector']:
            name = params['name']
            fig = plot_bar_subgroup_over_time(
                df=df[df['Scenario'] == params['scenario']],
                title=f'Emissions by Sector<br>{name}',
                xaxis_title='',
                yaxis_title='Mt CO2e',
                color_map=color_map,
                include_sum=True,
            )
            fig.write_image(FIGURES_PATH / f"emissions_by_sector_over_time_{params['id']}.pdf")


def graph_egen_by_resource_over_time(df_in, individual_sce_graph_params, color_map, branch_map, suffix):
    df = calculate_annual_result_by_subgroup(df_in, GENERATION_STRING, branch_map)
    df['Value'] = df['Value'] / 1e9

    for params in individual_sce_graph_params:
        if params['egen_by_resource']:
            name = params['name']
            fig = plot_bar_subgroup_over_time(
                df=df[df['Scenario'] == params['scenario']],
                title=f'Electricity Generation by Resource<br>{name}',
                xaxis_title='',
                yaxis_title='EJ',
                color_map=color_map,
                include_sum=True,
            )
            fig.write_image(FIGURES_PATH / f"egen_by_resource_over_time_{suffix}_{params['id']}.pdf")


def graph_cumulative_egen_capacity_added_over_time(df_in, individual_sce_graph_params,
                                                   color_map, branch_map, suffix):
    df = calculate_annual_result_by_subgroup(df_in, CAPACITY_ADDED_STRING, branch_map)
    df = cumsum_it(df)
    df['Value'] = df['Value'] / 1e3

    for params in individual_sce_graph_params:
        if params['cumulative_egen_capacity_added']:
            name = params['name']
            fig = plot_bar_subgroup_over_time(
                df=df[df['Scenario'] == params['scenario']],
                title=f'Cumulative New Electricity Generation Capacity <br>{name}',
                xaxis_title='',
                yaxis_title='GW',
                color_map=color_map,
                include_sum=True,
            )
            fig.write_image(FIGURES_PATH / f"egen_capacity_added_over_time_{suffix}_{params['id']}.pdf")


def graph_energy_demand_by_fuel_over_time(df_in, individual_sce_graph_params, color_map, branch_map,
                                          fuels=None):

    df = calculate_annual_result_by_subgroup(df_in, [ENERGY_DEMAND_STRING, INPUTS_STRING], branch_map)
    df = df.replace({"Fuel": FUELS_TO_COMBINE})

    if fuels is None:
        df = df[df['Fuel'] != "Other"].copy()
    else:
        df = df[df['Fuel'].isin(fuels)]

    for params in individual_sce_graph_params:
        if params['energy_demand_by_fuel']:
            name = params['name']
            fig = plot_bar_subgroup_over_time(
                df=df[df['Scenario'] == params['scenario']],
                title=f'Energy Demand by Fuel<br>{name}',
                xaxis_title='',
                yaxis_title='GJ',
                color_map=color_map,
                yaxis_col='Value',
                xaxis_col='Year',
                color_col='Fuel',
                include_sum=True,
            )
            fig.write_image(FIGURES_PATH / f"energy_demand_by_fuel_over_time_{params['id']}.pdf")


def graph_tech_choice_emissions(df_in, tech_choice_graph_params, stacked, branch_map, color_map, year=END_YEAR):

    if not stacked:
        color_col = 'Sector'
    else:
        color_col = 'Subgroup'

    df = calculate_annual_result_by_subgroup(df_in, EMISSIONS_RESULT_STRING, branch_map)
    df = df[df['Year'] == year]
    df = df.reset_index()    # need to reset index for the marginalize_it function to work
    df['Value'] = -1 * df['Value'] / 1e6

    for i, params in enumerate(tech_choice_graph_params):
        df_graph = marginalize_it(df, params['relative_to_map'])
        df_graph = df_graph[df_graph['Scenario'].isin(params['scenarios'])].copy()
        df_graph['Sector'] = df_graph['Scenario'].map(params['sector_map'])
        df_graph = df_graph.replace({'Scenario': params['name_map']})
        df_graph = df_graph.sort_values(by=['Sector', 'Scenario'], ascending=False)

        fig = plot_bar_scenario_comparison(
            df=df_graph,
            title=f'Abated Annual Emissions Contribution, {year}',
            xaxis_title='Mt CO2e',
            yaxis_title='',
            color_dict=color_map,
            color_column=color_col,
            include_legend=True,
        )
        if len(params['scenarios']) > 30:
            fig = update_to_tall_fig(fig)

        if stacked:
            fig.write_image(FIGURES_PATH / f"tech_choice_emissions_stacked_by_sector_{i}.pdf")
        else:
            fig.write_image(FIGURES_PATH / f"tech_choice_emissions{i}.pdf")


def graph_tech_choice_marginal_cost(df_in, tech_choice_graph_params, stacked, branch_map, color_map, year=2045):

    if not stacked:
        color_col = 'Sector'
    else:
        color_col = 'Subgroup'

    df = calculate_annual_result_by_subgroup(df_in, COST_RESULT_STRING, branch_map)
    df = discount_it(df)
    df = cumsum_it(df)
    df = df[df['Year'] == year]
    df = df.reset_index()    # need to reset index for the marginalize_it function to work
    df['Value'] = df['Value'] / 1e9

    for i, params in enumerate(tech_choice_graph_params):
        df_graph = marginalize_it(df, params['relative_to_map'])
        df_graph = df_graph[df_graph['Scenario'].isin(params['scenarios'])].copy()
        df_graph['Sector'] = df_graph['Scenario'].map(params['sector_map'])
        df_graph = df_graph.replace({'Scenario': params['name_map']})
        df_graph = df_graph.sort_values(by=['Sector', 'Scenario'])

        fig = plot_bar_scenario_comparison(
            df=df_graph,
            title='Marginal Cost Contribution',
            xaxis_title='$B',
            yaxis_title='',
            color_dict=color_map,
            color_column=color_col,
            include_legend=True,
        )
        if len(params['scenarios']) > 30:
            fig = update_to_tall_fig(fig)

        if stacked:
            fig.write_image(FIGURES_PATH / f"tech_choice_costs_stacked_by_sector_{i}.pdf")
        else:
            fig.write_image(FIGURES_PATH / f"tech_choice_costs_{i}.pdf")


def graph_tech_choice_cost_of_abatement(df_in, tech_choice_graph_params, color_map, branch_map):

    for i, params in enumerate(tech_choice_graph_params):

        df_graph = evaluate_dollar_per_ton_abated(df_in, branch_map, params['relative_to_map'])
        df_graph['Value'] = df_graph['cost_of_abatement']
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
        if len(params['scenarios']) > 30:
            fig = update_to_tall_fig(fig)
        # fig.update_layout(xaxis_range=[-500, 500])
        fig.write_image(FIGURES_PATH / f"tech_choice_cost_of_abatement_{i}.pdf")


def graph_load_by_sector(df_in, params, color_map):
    df = df_in.copy()
    df['Value'] = df['Value'] / 1e3

    for i, param in enumerate(params):
        name = param['name']
        fig = plot_area_subgroup_over_time(
            df=df[
                (df['Scenario'] == param['scenario']) &
                (df['Year'] == param['year']) &
                (df['Result Variable'] == param['result_var'])
            ],
            title=f"Electric Load by Sector<br>{name}",
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

    for i, params in enumerate(comp_params):
        df_graph = pd.DataFrame(columns=df.columns)
        for sce in params['scenarios']:
            df_graph = pd.concat([
                df_graph,
                df[
                    (df['Scenario'] == sce) &
                    (df['Year'] == params['year_map'][sce]) &
                    (df['Result Variable'] == params['result_map'][sce])
                ]
            ], axis=0, ignore_index=True)
        df_graph = df_graph.replace({'Scenario': params['name_map']})
        fig = plot_load_comparison(
            df=df_graph,
            color_col='Scenario',
            dash_col='Scenario',
            color_dict=params['color_map_by_name'],
            line_dict=params['line_map_by_name'],
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


def plot_grouped_bar_scenario_comparison(df, title, xaxis_title, yaxis_title, color_dict, grouping_column,
                                         color_column, include_legend=False):
    fig = px.bar(
        df,
        x=grouping_column,
        y='Value',
        color=color_column,
        barmode='group',
        color_discrete_map=color_dict,
    )
    if include_legend:
        pass
        # fig = update_legend_layout(fig, xaxis_title)
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
    # fig = update_legend_layout(fig, xaxis_title)
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