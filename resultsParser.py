import pandas as pd
import itertools
from pathlib import Path
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

import seaborn as sns

DISCOUNT_RATE = 0.05
INPUT_PATH = Path("resultsFiles/mar6_2023")
CONTROLLER_PATH = INPUT_PATH / "results_controller"
CLEAN_RESULTS_PATH = INPUT_PATH / "clean_results"
CARB_PATH = INPUT_PATH / "carb_results"
FIGURES_PATH = INPUT_PATH / "figures"
EMISSIONS_RESULT_STRING = "One_Hundred Year GWP Direct At Point of Emissions"
COST_RESULT_STRING = "Social Costs"
GENERATION_STRING = "Outputs by Output Fuel"


def main():

    df = load_data(reload=False)
    scenario_comparisons = load_sce_comps()
    egen_resource_color_map, sector_color_map = load_color_maps()
    egen_branch_map_long, egen_branch_map_short = form_egen_branch_maps(df)
    sector_branch_map = form_sector_branch_map(df)

    # Graphs that compare various scenarios
    # graph_emissions_over_time_scenario_comparisons(df, scenario_comparisons)
    # graph_marginal_cost_over_time_scenario_comparisons(df, scenario_comparisons)
    # graph_marginal_emissions_vs_marginal_cost_scatter_scenario_comparison(df, scenario_comparisons)
    # graph_cost_of_co2_abatement_bar(df, scenario_comparisons)
    # graph_egen_by_resource_scenario_comparison(df, scenario_comparisons, egen_resource_color_map['long'],
    #                                            egen_branch_map_long, 'long')
    # graph_egen_by_resource_scenario_comparison(df, scenario_comparisons, egen_resource_color_map['short'],
    #                                            egen_branch_map_short, 'short')
    # graph_cumulative_marginal_costs_by_sector_scenario_comparison(df, scenario_comparisons, sector_color_map, sector_branch_map,
    #                                                               year=2045, relative_to='LEAP Version CARB Reference_0_nan')
    # graph_cumulative_marginal_abated_emissions_by_sector_scenario_comparison(df, scenario_comparisons, sector_color_map, sector_branch_map,
    #                                                               year=2045, relative_to='LEAP Version CARB Reference_0_nan')
    # graph_annual_marginal_abated_emissions_by_sector_scenario_comparison(df, scenario_comparisons, sector_color_map, sector_branch_map,
    #                                                               year=2045, relative_to='LEAP Version CARB Reference_0_nan')

    # Graphs that only look at one scenario
    individual_sce_graph_params = load_individual_scenarios()
    graph_marginal_costs_by_sector_over_time(df, individual_sce_graph_params, sector_color_map, sector_branch_map)
    # graph_marginal_emissions_by_sector_over_time()
    # graph_egen_by_resource_over_time()
    # graph_cumulative_egen_capacity_added_over_time()

    # Load shape graphs

    # RPS graphs

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
        sc['marker_map'] = dict(zip(dfg['Scenario'], dfg['Marker']))
        scenario_comps.append(sc)

    return scenario_comps


def load_individual_scenarios():
    """ Function to load scenario comparisons as dictated in controller """
    df = pd.read_excel(CONTROLLER_PATH / 'controller.xlsx', sheet_name="individual_scenario_graphs")

    individual_scenario_graphs = {
        'scenarios': df['Scenario'].tolist(),
        'name_map': dict(zip(df['Scenario'], df['Naming']))
    }

    return individual_scenario_graphs


def load_color_maps():
    """ Function to load color maps from controller """

    df = pd.read_excel(CONTROLLER_PATH / 'controller.xlsx', sheet_name="egen_resource_colors")
    egen_resource_color_map = {
        'long': dict(zip(df[df['Length'] == 'Long']['Resource'], df[df['Length'] == 'Long']['Color'])),
        'short': dict(zip(df[df['Length'] == 'Short']['Resource'], df[df['Length'] == 'Short']['Color'])),
    }

    df = pd.read_excel(CONTROLLER_PATH / 'controller.xlsx', sheet_name="sector_colors")
    sector_color_map = dict(zip(df['Sector'], df['Color']))

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
        fig = plot_scatter_scenario_comparison(
            df, 'Emissions vs Cost', 'Marginal Abated Emissions (Gt CO2e)', 'Marginal Cost ($B)', sc,
        )
        fig.write_image(FIGURES_PATH / f"emissions_v_cost_{i}.pdf")


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
        df_graph = df[df['Scenario'].isin(sc['scenarios'])].copy()
        df_graph = df_graph.replace({'Scenario': sc['name_map']})
        color_map = dict(zip([sc['name_map'][sce] for sce in sc['scenarios']],
                             [sc['color_map'][sce] for sce in sc['scenarios']]))

        fig = plot_bar_scenario_comparison(
            df=df_graph,
            title='Cost of Carbon Abatement',
            xaxis_title='$/t CO2e',
            yaxis_title='',
            color_dict=color_map,
            color_column='Scenario',
            include_legend=False,
        )
        fig.write_image(FIGURES_PATH / f"cost_of_abatement_{i}.pdf")


def graph_egen_by_resource_scenario_comparison(df_in, scenario_comparisons, color_map, branch_map, file_suffix, year=2045):

    df = calculate_annual_result_by_subgroup(df_in, GENERATION_STRING, branch_map)
    df = df[df['Year'] == year].copy()
    df['Value'] = df['Value'] / 1e9

    for i, sc in enumerate(scenario_comparisons):
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

    for i, sce in enumerate(individual_sce_graph_params['scenarios']):
        name = individual_sce_graph_params['name_map'][sce]
        fig = plot_bar_subgroup_over_time(
            df=df[df['Scenario'] == sce],
            title=f'Marginal Cost by Sector\n{name}',
            xaxis_title='',
            yaxis_title='$B',
            color_map=color_map,
            include_sum=True,
        )
        fig.write_image(FIGURES_PATH / f"marginal_cost_over_time_by_sector_{i}.pdf")


def graph_marginal_emissions_by_sector_over_time():
    pass


def graph_egen_by_resource_over_time():
    pass


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


def plot_bar_subgroup_over_time(df, title, xaxis_title, yaxis_title, color_map, include_sum=True):
    fig = px.bar(
        df,
        x='Year',
        y='Value',
        color='Subgroup',
        color_discrete_map=color_map,
    )

    if include_sum:
        df_sum = pd.DataFrame(columns=['Year', 'Value'])
        for yr in df['Year'].unique():
            annual_sum = df[df['Year'] == yr]['Value'].sum()
            df_sum.loc[len(df_sum.index)] = [yr, annual_sum]
        # add line to graph showing cumulative sum
        fig.add_trace(go.Scatter(
            mode='lines',
            x=df_sum['Year'],
            y=df_sum['Value'],
            name="Annual Total",
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


def plot_bar_scenario_comparison(df, title, xaxis_title, yaxis_title, color_dict, color_column='Subgroup', include_legend=False):

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