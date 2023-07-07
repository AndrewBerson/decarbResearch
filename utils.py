import pandas as pd
import itertools
import numpy as np
from typing import Union
from pathlib import Path

CONTROLLER_PATH = Path("resultsFiles/results_controller")

# LEAP result strings
EMISSIONS_RESULT_STRING = "One_Hundred Year GWP Direct At Point of Emissions"
COST_RESULT_STRING = "Social Costs"

# Fuels that are equivalent
FUELS_TO_COMBINE = {"CRNG": "RNG", "CNG": "NG", "Hydrogen Transmitted": "Hydrogen"}

# Years and discount rate
BASE_YEAR = 2018
END_YEAR = 2045
DISCOUNT_RATE = 0.05


def form_df_graph(
    df_in: pd.DataFrame,
    sce_group_params: dict,
    result: list,
    multiplier: float,
    marginalize: bool,
    cumulative: bool,
    discount: bool,
    filter_yrs: bool,
    branch_map: dict,
    fuel_filter: Union[list, None],
    groupby: list,
) -> pd.DataFrame:
    """
    Function to make dataframe for graphing according to instructions in controller
    :param df_in: DataFrame of results
    :param sce_group_params: info contained in "scenario group params" tab of controller
    :param result: what result(s) to evaluate
    :param multiplier: value to scale results
    :param marginalize: T/F whether to marginalize results
    :param cumulative: T/F if results are cumulative sum
    :param discount: T/F if results should be discounted by discount rate
    :param filter_yrs: T/F if scenarios have 1 relevant yr, or if all yrs are relevant
    :param branch_map: map from LEAP branch --> Subgroup
    :param fuel_filter: list of fuels to filter for
    :param groupby: List of params that should be grouped together
    :return: DataFrame of results that will be used to make the graph
    """
    # filter out irrelevant scenarios
    df_graph = df_in[
        df_in["Scenario"].isin(sce_group_params["relevant_scenarios"])
    ].copy()

    # Calculate result (special function for cost of abatement)
    if result == ["cost of abatement"]:
        # calculate cost of abatement
        df_graph_1 = evaluate_dollar_per_ton_abated(
            df_in=df_graph,
            subgroup_dict=branch_map,
            relative_to_map=sce_group_params["relative_to_map"],
        )
        # get any cost of abatements published by CARB
        df_graph_2 = calculate_annual_result_by_subgroup(df_graph, result, branch_map)
        # combine results
        df_graph = pd.concat([df_graph_1, df_graph_2], ignore_index=True, sort=True)
        df_graph["Value"] = df_graph["Value"] * multiplier
    else:
        # calculate result
        df_graph = calculate_annual_result_by_subgroup(df_graph, result, branch_map)
        df_graph["Value"] = df_graph["Value"] * multiplier

        # combine fuels
        df_graph = df_graph.replace({"Fuel": FUELS_TO_COMBINE})

        # if specified, filter for specific fuels
        if fuel_filter is not None:
            df_graph = df_graph[df_graph["Fuel"].isin(fuel_filter)].copy()

        # discount, marginalize, cumsum
        if discount:
            df_graph = discount_it(df_graph)
        if marginalize:
            df_graph = marginalize_it(df_graph, sce_group_params["relative_to_map"])
        if cumulative:
            df_graph = cumsum_it(df_graph)

    # get rid of years not specified to be included
    if filter_yrs:
        for sce, yr in sce_group_params["specified_year_map"].items():
            df_graph = df_graph.reset_index(drop=True)
            rows_to_drop = np.array(
                (df_graph["Scenario"] == sce) & (df_graph["Year"] != yr)
            )
            row_ids_to_drop = list(np.where(rows_to_drop)[0])
            df_graph = df_graph.drop(index=row_ids_to_drop)

    # get rid of unneeded scenarios
    df_graph = df_graph[df_graph["Scenario"].isin(sce_group_params["scenarios"])].copy()

    # add columns based on the relevant maps (name_map, color_map...)
    for k, v in sce_group_params.items():
        if k.endswith("_map"):
            df_graph[k[:-4]] = df_graph["Scenario"].map(v)

    # sum values within the same year, scenario, specified color
    df_graph = df_graph.groupby(by=groupby, as_index=False)["Value"].sum()

    return df_graph


def form_df_graph_load(
    df_in: pd.DataFrame,
    sce_group_params: dict,
    graph_params: dict,
    sum_across_branches: bool = False,
) -> pd.DataFrame:
    """
    Form DataFrame for load shape graphs
    :param df_in: DataFrame of load shape results
    :param sce_group_params: info from graph group in controller tab "scenario group params"
    :param graph_params: info from row of controller controlling the specific graph
    :param sum_across_branches: bool whether or not to aggregate sub-branches (eg: transport, industry...)
    :return: DataFrame ready to be used for graphing
    """
    # filter for relevant scenarios
    df_graph = df_in[df_in["Scenario"].isin(sce_group_params["scenarios"])].copy()

    # filter for correct result
    df_graph = df_graph[df_graph["Result Variable"] == graph_params["result"]].copy()

    # scale result
    df_graph["Value"] = df_graph["Value"] * graph_params["multiplier"]

    # sum loads across branches
    if sum_across_branches:
        df_graph = sum_load_across_branches(df_graph)

    # add columns based on the relevant maps (name_map, color_map...)
    for k, v in sce_group_params.items():
        if k.endswith("_map"):
            df_graph[k[:-4]] = df_graph["Scenario"].map(v)

    # get rid of years not specified to be included
    for sce, yr in sce_group_params["load_shape_yr_map"].items():
        df_graph = df_graph.reset_index(drop=True)
        rows_to_drop = np.array(
            (df_graph["Scenario"] == sce) & (df_graph["Year"] != yr)
        )
        row_ids_to_drop = list(np.where(rows_to_drop)[0])
        df_graph = df_graph.drop(index=row_ids_to_drop)

    return df_graph


def calculate_annual_result_by_subgroup(
    df_in: pd.DataFrame,
    result: Union[str, list],
    subgroup_dict: dict,
) -> pd.DataFrame:
    """
    Function to sum the result variable in each year for the branches in each key/value pairing of subgroup dict
    :param df_in: dataframe containing all relevant results
    :param result: either a string or list of the relevant results (eg: Output by Output Fuel)
    :param subgroup_dict: dictionary mapping groups to their relevant branches (Eg: 'buildings' --> [Demand\Residential...]
    :return: dataframe with cols Year, Scenario, Fuel, Subgroup, Value. Subgroups are the keys of the subgroup_dict
    """

    df_out = pd.DataFrame(columns=["Year", "Scenario", "Fuel", "Subgroup", "Value"])

    # convert result_str to list so that multiple result_strings can be passed into the function as a list
    # note this is useful for energy demand and inputs
    if type(result) == str:
        result = [result]

    for key, dfg in df_in[df_in["Result Variable"].isin(result)].groupby(
        by=["Year", "Scenario", "Fuel"]
    ):
        yr, sce, fuel = key
        mask = np.array(
            (dfg["Year"] == yr) & (dfg["Scenario"] == sce) & (dfg["Fuel"] == fuel)
        )
        row_ids = list(np.where(mask)[0])
        for subgroup, branches in subgroup_dict.items():
            # eliminate branches that do not appear in dfg
            branches = list(set(branches).intersection(set(dfg.columns)))
            value = dfg[branches].iloc[row_ids].sum(axis=1).sum()
            df_out.loc[len(df_out.index)] = [yr, sce, fuel, subgroup, value]

    return df_out


def sum_load_across_branches(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Function to sum electric load across branches that belong together
    :param df_in: load shape results
    :return: updated load shape results
    """
    df = pd.DataFrame(columns=["Year", "Hour", "Scenario", "Result Variable", "Value"])

    yrs = df_in["Year"].unique()
    hrs = df_in["Hour"].unique()
    scenarios = df_in["Scenario"].unique()
    results_vars = df_in["Result Variable"].unique()

    for (
        yr,
        hr,
        sce,
        res,
    ) in itertools.product(yrs, hrs, scenarios, results_vars):
        df_to_add = df_in[
            (df_in["Year"] == yr)
            & (df_in["Hour"] == hr)
            & (df_in["Scenario"] == sce)
            & (df_in["Result Variable"] == res)
        ]
        if len(df_to_add.index) > 0:
            df.loc[len(df.index), :] = yr, hr, sce, res, df_to_add["Value"].sum(axis=0)

    return df


def evaluate_cumulative_marginal_emissions_cumulative_marginal_cost(
    df_in: pd.DataFrame,
    subgroup_dict: dict,
    relative_to_map: dict,
) -> pd.DataFrame:
    """
    Function to evaluate cumulative marginal emissions and cumulative marginal costs
    :param df_in: raw results from LEAP script
    :param subgroup_dict: dict mapping subgroup --> list of relevant branches
    :param relative_to_map: dict mapping scenario --> scenario to marginalize against
    :return: dataframe containing cols 'cumulative_marginal_cost' and 'cumulative_marginal_emissions'
    """
    df_cost = calculate_annual_result_by_subgroup(
        df_in, COST_RESULT_STRING, subgroup_dict
    )
    df_cost = marginalize_it(df_cost, relative_to_map)
    df_cost = discount_it(df_cost)
    df_cost = cumsum_it(df_cost)
    df_cost = df_cost.rename(columns={"Value": "cumulative_marginal_cost"})

    df_emissions = calculate_annual_result_by_subgroup(
        df_in, EMISSIONS_RESULT_STRING, subgroup_dict
    )
    df_emissions = marginalize_it(df_emissions, relative_to_map)
    df_emissions = cumsum_it(df_emissions)
    df_emissions = df_emissions.rename(
        columns={"Value": "cumulative_marginal_emissions"}
    )

    df = df_emissions.merge(df_cost, how="outer", on=["Scenario", "Subgroup", "Year"])

    return df


def evaluate_dollar_per_ton_abated(
    df_in: pd.DataFrame,
    subgroup_dict: dict,
    relative_to_map: dict,
) -> pd.DataFrame:
    """
    Function to evalute the cost of abatement
    :param df_in: dataframe containing results
    :param subgroup_dict: dict mapping subgroup --> list of branches
    :param relative_to_map: dict mapping scenario --> scenario it should be marginalized against
    :return: df containing col 'cost_of_abatement' for year == End year
    """

    total_yrs = END_YEAR - BASE_YEAR + 1
    capital_recovery_factor = (DISCOUNT_RATE * ((1 + DISCOUNT_RATE) ** total_yrs)) / (
        ((1 + DISCOUNT_RATE) ** total_yrs) - 1
    )

    df = evaluate_cumulative_marginal_emissions_cumulative_marginal_cost(
        df_in, subgroup_dict, relative_to_map
    )
    df = df[df["Year"] == END_YEAR].copy()
    df["annualized_cost"] = df["cumulative_marginal_cost"] * capital_recovery_factor
    df["annualized_emissions_reduction"] = (
        -1 * df["cumulative_marginal_emissions"] / total_yrs
    )
    df["cost_of_abatement"] = (
        df["annualized_cost"] / df["annualized_emissions_reduction"]
    )
    df["Value"] = df["cost_of_abatement"]

    return df


def marginalize_it(
    df_in: pd.DataFrame,
    relative_to_dict: dict,
) -> pd.DataFrame:
    """
    Function to calculate marginal result
    :param df_in: Dataframe containing the following cols: Year, Scenario, Fuel, Subgroup, Value. The 'Value' column
    is marginalized against other columns that contain the same year, fuel, and subgroup
    :param relative_to_dict: dictionary where the keys are scenarios and the values are that the key should be
    marginalized against
    :return: dataframe where all of the scenarios in relative_to_dict have been marginalized
    """

    df_out = df_in.copy()

    # iterate through scenarios and what they're being marginalized against in the relative_to_dict
    for sce, relative_to in relative_to_dict.items():

        # find all relevant subgroups, years, and fuels for the scenario
        subgroups = df_out[df_out["Scenario"] == sce]["Subgroup"].unique()
        years = df_out[df_out["Scenario"] == sce]["Year"].unique()
        fuels = df_out[df_out["Scenario"] == sce]["Fuel"].unique()

        for subg, yr, fuel in itertools.product(subgroups, years, fuels):
            # subtract out the scenario that it's being marginalized relative to
            df_out.loc[
                (df_out["Scenario"] == sce)
                & (df_out["Subgroup"] == subg)
                & (df_out["Fuel"] == fuel)
                & (df_out["Year"] == yr),
                "Value",
            ] -= float(
                df_in.loc[
                    (df_in["Scenario"] == relative_to)
                    & (df_in["Subgroup"] == subg)
                    & (df_in["Fuel"] == fuel)
                    & (df_in["Year"] == yr),
                    "Value",
                ]
            )

    return df_out


def discount_it(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Function to discount all costs
    :param df_in: dataframe containing results with cols Year, Scenario, Fuel, Subgroup, Value
    :return: dataframe with discounted costs
    """
    df = df_in.copy()
    yrs = np.sort(df["Year"].unique())
    base_yr = yrs[0]

    # discount all costs
    for key, dfg in df.groupby(by=["Scenario", "Subgroup", "Year"]):
        sce, subg, yr = key
        mask = np.array(
            (df["Scenario"] == sce) & (df["Subgroup"] == subg) & (df["Year"] == yr)
        )
        ids = list(np.where(mask)[0])
        df.iloc[ids, df.columns.get_loc("Value")] = dfg["Value"] / (
            1 + DISCOUNT_RATE
        ) ** (yr - base_yr)

    return df


def cumsum_it(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Function to calculate cumulative sum of 'Value' column across years and separated by Scenario, subgroup, and fuel
    :param df_in: dataframe containing results with cols Year, Scenario, Fuel, Subgroup, Value
    :return: dataframe with 'Value' column now containing the cumulative sum beginning from the base year
    """
    df = df_in.copy()
    df = df.sort_values(by="Year", axis=0)
    for key, dfg in df.groupby(by=["Scenario", "Subgroup", "Fuel"]):
        sce, subg, fuel = key
        mask = np.array(
            (df["Scenario"] == sce) & (df["Subgroup"] == subg) & (df["Fuel"] == fuel)
        )
        ids = list(np.where(mask)[0])
        df.iloc[ids, df.columns.get_loc("Value")] = dfg["Value"].cumsum(axis=0)

    return df


def create_scenario_copies(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Make copies of scenarios under new name.
    :param df: results
    :return: results including new scenario copies
    """
    df_excel = pd.read_excel(
        CONTROLLER_PATH / "controller.xlsm", sheet_name="scenario_copies"
    )

    sce_copy_dict = dict(zip(df_excel["Copy Name"], df_excel["Original Scenario"]))
    sce_instate_incentive_on_off_dict = dict(
        zip(df_excel["Copy Name"], df_excel["In State Incentives"])
    )

    for new_name, original_name in sce_copy_dict.items():
        df_to_add = df[df["Scenario"] == original_name].copy()
        df_to_add["Scenario"] = new_name
        df = pd.concat([df, df_to_add], axis=0)

    return remove_instate_incentives(df, sce_instate_incentive_on_off_dict)


def remove_instate_incentives(
    df: pd.DataFrame,
    scenario_dict: dict,
) -> pd.DataFrame:
    """
    Function to zero-out instate incentives from scenarios
    :param df: results
    :param scenario_dict: dict indicating which scenarios should have incentives removed
    :return: updated results
    """
    scenarios_to_remove_incentives = [
        sce for sce, on_off in scenario_dict.items() if on_off.lower() == "off"
    ]
    relevant_columns = [col for col in df.columns if "Non Energy\\Incentives" in col]

    df.loc[
        (df["Scenario"].isin(scenarios_to_remove_incentives))
        & (df["Result Variable"] == COST_RESULT_STRING),
        relevant_columns,
    ] = 0

    return df


def form_branch_maps(df_results: pd.DataFrame) -> dict:
    """
    Function to make branch maps per branch_maps tab in controller
    :param df_results: results (used to check if any branches are missing in controller)
    :return: dict of branch maps
    """
    # for set of all branches included in the results
    id_cols = ["Year", "Scenario", "Result Variable", "Fuel"]
    all_branches = set(df_results.columns) - set(id_cols)

    df = pd.read_excel(CONTROLLER_PATH / "controller.xlsm", sheet_name="branch_maps")

    # check if there are any branches missing from the controller
    missing_branches = list(all_branches - set(df["Branch"].unique()))
    if len(missing_branches) > 0:
        print(f"Branches not included in controller: {missing_branches}")

    # form maps of branches
    branch_maps = dict()
    map_names = df.columns.tolist()
    map_names.remove("Branch")

    # iterate through columns in the controller
    for map_name in map_names:
        branch_maps[map_name] = dict()

        df_map = df[["Branch"] + [map_name]].copy()

        # map unique sector (or other variable) to relevant branches
        for key, dfg in df_map.groupby(map_name):
            if key == False:
                continue
            branch_maps[map_name][key] = dfg["Branch"].tolist()

    return branch_maps


def form_sce_group_params() -> tuple:
    """
    function to make dicts of parameters associated with each scenarios within each group
    :return: set of scenarios used, dict of parameters
    """
    df = pd.read_excel(
        CONTROLLER_PATH / "controller.xlsm", sheet_name="scenario_group_params"
    )

    relevant_scenarios = set(df["scenario"].unique())
    relevant_scenarios.update(set(df["relative_to"].unique()))

    map_val_cols = [
        col for col in df.columns.tolist() if col not in ["group_id", "scenario"]
    ]
    map_val_cols_by_name = [
        col
        for col in df.columns.tolist()
        if col not in ["group_id", "scenario", "name"]
    ]

    sce_group_params = dict()
    for group_id, dfg in df.groupby(by=["group_id"]):
        sce_group_params[group_id] = dict()

        sce_group_params[group_id]["scenarios"] = dfg["scenario"].tolist()
        sce_group_params[group_id]["relevant_scenarios"] = list(
            set(dfg["scenario"].tolist() + dfg["relative_to"].tolist())
        )

        for col in map_val_cols:
            sce_group_params[group_id][col + "_map"] = dict(
                zip(dfg["scenario"], dfg[col])
            )

        for col in map_val_cols_by_name:
            sce_group_params[group_id][col + "_map_by_name"] = dict(
                zip(dfg["name"], dfg[col])
            )

    return relevant_scenarios, sce_group_params


def load_map(sheet_name: str) -> dict:
    """
    Function to create mapping based on key / value pairings in controller
    :param sheet_name: sheet in controller
    :return: dict of k,v per controller
    """
    df = pd.read_excel(CONTROLLER_PATH / "controller.xlsm", sheet_name=sheet_name)

    return dict(zip(df["key"], df["value"]))
