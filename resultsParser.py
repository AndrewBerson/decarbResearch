import pandas as pd
import itertools
from pathlib import Path
import numpy as np
import plotly.express as px
import os
import seaborn as sns

DISCOUNT_RATE = 0.1


def main():
    input_path = Path("resultsFiles/mar6_2023")

    # folder to store combined csv of all results
    csv_path = input_path / Path('combinedCSV')
    csv_path.mkdir(parents=True, exist_ok=True)

    # folder to store graphs created
    figures_path = input_path / Path('figures')
    figures_path.mkdir(parents=True, exist_ok=True)

    reload = True
    if reload:
        df = load_all_files(input_path)
        df = reformat(df)
        df.to_csv(csv_path / 'combined_results.csv')
    else:
        df = pd.read_csv(csv_path / 'combined_results.csv', header=0, index_col=0)

    # scenarios = df.loc[:, "Scenario"].unique().tolist()
    #
    # scenario_comparisons = [
    #     [s for s in scenarios if ("Baseline" in s) or ("CARB" in s)],
    #     [s for s in scenarios if ("Baseline" in s) or ("CARB" in s) or ("_0_" in s)],
    #     [s for s in scenarios if "Baseline" in s],
    #     [s for s in scenarios if "CARB" in s],
    #     [s for s in scenarios if ("CARB" in s) or ("_0_" in s)],
    #     [s for s in scenarios if "High Elec" in s] + ["Stanford Baseline_0_NA"],
    #     [s for s in scenarios if "High Biofuels" in s] + ["Stanford Baseline_0_NA"],
    #     [s for s in scenarios if "High CCS" in s] + ["Stanford Baseline_0_NA"],
    #     [s for s in scenarios if "High H2" in s] + ["Stanford Baseline_0_NA"],
    # ]
    # emissions_cost_v_time(df, figures_path, scenario_comparisons)
    # # cost_v_emissions(df, figures_path, scenario_comparisons)
    # # emissions_makeup_v_time(df, figures_path)
    # # energy_demand(df, figures_path)


def load_all_files(input_path):
    df = pd.DataFrame
    added_scenarios = set()
    i = 0
    for fname in os.listdir(input_path):
        f = os.path.join(input_path, fname)
        if os.path.isfile(f) and (fname[0] not in [".", "~"]):
            df_excel = pd.read_excel(f, sheet_name="Results")

            # ensure that the same scenario doesn't get added multiple times
            # eg: if baseline gets run multiple times and has results in multiple excel files
            row_ids = [i for i, sce in enumerate(df_excel.loc[:, "Scenario"]) if sce not in added_scenarios]
            if i == 0:
                df = df_excel.iloc[row_ids, :].copy()
            else:
                df = pd.concat([df, df_excel.iloc[row_ids, :].copy()], sort=True)
            added_scenarios = set(df.loc[:, "Scenario"].unique())
            i += 1

    return df


def reformat(df_excel):
    """
    Function to take LEAP script output and convert it to long dataframe
    Dataframe indices are years 2018-2045 and columns contain both id and results cols
        id cols - Scenario, Result Variable, Fuel
        result cols - each branch output
    :param f - file path:
    :return: dataframe of results
    """
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

    return df.fillna(0)


def emissions_cost_v_time(df, output_path, comparisons, relative_to="Stanford Baseline_0_NA"):
    id_cols = ['Scenario', 'Result Variable', 'Fuel']
    result_cols = list(set(df.columns) - set(id_cols))
    emissions_result_str = 'One_Hundred Year GWP Direct At Point of Emissions'
    cost_result_str = 'Social Costs'

    df_emissions = pd.DataFrame(index=df.index.unique())
    for sce, df_group in df[df['Result Variable'] == emissions_result_str].groupby('Scenario'):
        df_emissions = df_emissions.join(pd.Series(df_group[result_cols].sum(axis=1), name=sce))

    df_costs = pd.DataFrame(index=df.index.unique())
    for sce, df_group in df[df['Result Variable'] == cost_result_str].groupby('Scenario'):
        df_costs = df_costs.join(pd.Series(df_group[result_cols].sum(axis=1), name=sce))

    # Marginalize the costs
    for col in df_costs.columns:
        df_costs[col] = df_costs[col] - df_costs[relative_to]

    # setup legend names
    short_names_dict = dict(zip(
        df_emissions.columns,
        [col.split("_")[0] for col in df_emissions.columns],
    ))
    long_names_dict = dict()
    line_styles = dict()
    for col in df_emissions.columns:
        if col.split("_")[2] == "NA":
            long_names_dict[col] = col.split("_")[0]
        else:
            long_names_dict[col] = col.split("_")[0] + " (" + col.split("_")[2] + ")"
        if col.split("_")[1] == "0":
            line_styles[col] = "solid"
        else:
            line_styles[col] = "dash"

    for i, comp in enumerate(comparisons):
        # Emissions graph
        fig_emis = px.line(
            df_emissions[comp],
            title="Scenario Emissions (tonnes CO2e)",
            line_dash=0,
            # line_dash_map=line_styles,
        )
        fig_emis.update_layout(
            xaxis_title="",
            yaxis_title="",
            legend_title=""
        )
        fig_emis.for_each_trace(
            lambda t: t.update(
                name=short_names_dict[t.name],
                legendgroup=short_names_dict[t.name],
                hovertemplate = t.hovertemplate.replace(t.name, short_names_dict[t.name]),
            )
        )
        fig_emis.write_image(output_path / f"scenario_emissions_{i}.pdf")

        # marginal cost graph
        fig_cost = px.line(
            df_costs[comp],
            title="Scenario Costs"
        )
        fig_cost.update_layout(
            title="Marginal Cost (USD)",
            xaxis_title="",
            yaxis_title="",
            legend_title=""
        )
        fig_cost.write_image(output_path / f"scenario_marg_cost_{i}.pdf")


def cost_v_emissions(df, output_path, comparisons, relative_to="Stanford Baseline_0_NA"):
    id_cols = ['Scenario', 'Result Variable', 'Fuel']
    result_cols = list(set(df.columns) - set(id_cols))
    emissions_result_str = 'One_Hundred Year GWP Direct At Point of Emissions'
    cost_result_str = 'Social Costs'

    df_emissions_cost = pd.DataFrame(
        columns=['Scenario', 'Cumulative_emissions', 'Cumulative_Marginal_costs']
    )
    for sce, df_group in df.groupby('Scenario'):
        emissions = df_group.loc[:, result_cols][df_group.loc[:, 'Result Variable'] == emissions_result_str].sum(axis=0).sum()
        annual_costs = df_group.loc[:, result_cols][df_group.loc[:, 'Result Variable'] == cost_result_str].sum(axis=1)
        npv = 0
        base_year = annual_costs.index.min()
        for yr in annual_costs.index:
            npv += annual_costs.loc[yr] / (1 + DISCOUNT_RATE)**(yr - base_year)
        df_emissions_cost.loc[len(df_emissions_cost.index)] = [sce, emissions, npv]

    # set index to be the scenario
    df_emissions_cost = df_emissions_cost.set_index('Scenario')

    # calculate cost relative to stanford baseline
    df_emissions_cost.loc[:, 'Cumulative_Marginal_costs'] -= \
        df_emissions_cost.loc[relative_to, 'Cumulative_Marginal_costs']

    df_emissions_cost.loc[:, 'Cost_of_Abatement'] = \
        df_emissions_cost.loc[:, 'Cumulative_Marginal_costs'] / \
        (df_emissions_cost.loc[relative_to, 'Cumulative_emissions'] - df_emissions_cost.loc[:, 'Cumulative_emissions'])

    for i, comp in enumerate(comparisons):
        fig = px.scatter(
            df_emissions_cost.loc[comp, :],
            color=df_emissions_cost.loc[comp, :].index,
            x="Cumulative_emissions",
            y="Cumulative_Marginal_costs"
        )
        fig.update_layout(
            title=f"Emissions vs Costs (years {df.index.min()} - {df.index.max()})",
            xaxis_title="Cumulative Emissions (tonnes CO2e)",
            yaxis_title=f"Cumulative Marginal Cost (NPV, rate = {DISCOUNT_RATE:.2f})",
            legend_title=""
        )
        fig.update_traces(marker={'size': 10})
        fig.write_image(output_path / f"scenario_emissions_cost_{i}.pdf")

    # TODO: add color groupings for high CCS, high elec...
    fig = px.bar(df_emissions_cost, x=df_emissions_cost.index, y="Cost_of_Abatement")
    fig.write_image(output_path / f"abatement_cost.pdf")


def emissions_makeup_v_time(df, output_path):
    id_cols = ['Scenario', 'Result Variable', 'Fuel']
    result_cols = list(set(df.columns) - set(id_cols))
    branches = set([col.split("\\")[1] for col in result_cols])
    emissions_result_str = 'One_Hundred Year GWP Direct At Point of Emissions'

    sce_id = 0
    for sce, df_group in df.groupby("Scenario"):
        df_emissions = pd.DataFrame(columns=branches, index=df.index.unique())
        for branch in branches:
            col_ids = []
            for i, col in enumerate(df_group.columns):
                if len(col.split("\\")) >= 2:
                    if col.split("\\")[1] == branch:
                        col_ids.append(i)
            df_emissions.loc[:, branch] = df_group.iloc[:, col_ids][df_group.loc[:, "Result Variable"] == \
                                                              emissions_result_str].sum(axis=1)

        # df_emissions = df_emissions.sort_values(by=df.index[0], axis=1)

        # TODO: set it up so the graph only includes the top x variables, and the rest are grouped into "other"
        included_cols = set(df_emissions.sort_values(by=df_emissions.index[0], axis=1, ascending=False).columns[0:5])
        # included_cols += df_emissions.sort_values(by=df_emissions.index[-1], axis=1, ascending=False).columns[0:5]
        # included_cols.add([col for i, col in enumerate(df_emissions.columns) if df_emissions.iloc[-1, i] < 0])
        excluded_cols = list(set(df_emissions.columns) - included_cols)
        df_emissions.loc[:, "Other"] = df_emissions.loc[:, excluded_cols].sum(axis=1)
        df_emissions = df_emissions.drop(columns=excluded_cols)
        df_emissions = df_emissions.sort_values(by=df_emissions.index[-1], axis=1)

        fig = px.area(df_emissions, x=df_emissions.index, y=df_emissions.columns)
        fig.update_layout(
            title=sce,
            yaxis_title="Emissions (tonnes CO2e)"
        )
        fig.write_image(output_path / f"scenario_emissions_strip_{sce_id}.pdf")
        sce_id += 1


def energy_demand(df, output_path):
    id_cols = ['Scenario', 'Result Variable', 'Fuel']
    result_cols = list(set(df.columns) - set(id_cols))
    branches = set([col.split("\\")[1] for col in result_cols])
    energy_result_str = 'Energy Demand Final Units'
    fuels = df[df.loc[:, "Result Variable"] == energy_result_str].loc[:, "Fuel"].unique()
    scenarios = df.loc[:, "Scenario"].unique()
    yr_min = df.index.unique().min()
    yr_max = df.index.unique().max()

    df_energy_yr_min = pd.DataFrame(index=df.loc[:, "Scenario"].unique(), columns=fuels)
    df_energy_yr_max = pd.DataFrame(index=df.loc[:, "Scenario"].unique(), columns=fuels)

    for s, f in itertools.product(scenarios, fuels):
        df_energy_yr_min.loc[s, f] = df[
            (df.loc[:, "Scenario"] == s) &
            (df.loc[:, "Fuel"] == f) &
            (df.loc[:, "Result Variable"] == energy_result_str) &
            (df.index == yr_min)
        ].loc[:, result_cols].sum(axis=1)

        df_energy_yr_max.loc[s, f] = df[
            (df.loc[:, "Scenario"] == s) &
            (df.loc[:, "Fuel"] == f) &
            (df.loc[:, "Result Variable"] == energy_result_str) &
            (df.index == yr_max)
        ].loc[:, result_cols].sum(axis=1)


if __name__ == "__main__":
    main()