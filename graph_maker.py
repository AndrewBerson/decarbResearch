import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from utils import *

FIGURES_PATH = Path("resultsFiles/new_figures")
IMAGE_FORMAT = ".pdf"
IMAGE_SCALE = 1


def main():
    paths_and_folder_names = (
        (
            Path("resultsFiles/112_28results/clean_results"),
            "112_28results",
        ),
        (
            Path("resultsFiles/112_31results/clean_results"),
            "112_31results",
        ),
        (
            Path("resultsFiles/112_35results/clean_results"),
            "112_35results",
        ),
        (
            Path("resultsFiles/Leverable_NotLeverable/clean_results"),
            "Leverable_NotLeverable",
        ),
        (
            Path("resultsFiles/112_37results/clean_results"),
            "112_37results",
        ),
    )

    # form df of proxy results
    df_proxies = load_results(Path("resultsFiles/proxy_results/clean_results"))

    # create color and branch maps
    color_map = load_map("color_map")
    active_graph_map = load_map("active_graph_map")
    branch_maps = form_branch_maps(
        pd.concat(
            [load_results(p) for p, _ in paths_and_folder_names],
            ignore_index=True,
            sort=True,
        )
    )

    # read in scenario group parameters
    _, all_sce_group_params = form_sce_group_params()

    # results - ((graphing function, controller sheet name, T/F if graph is active))
    result_fns_sheets_active = (
        (
            lines_over_time,
            "lines_over_time",
            active_graph_map["lines_over_time"],
        ),
        (
            bars_over_time,
            "bars_over_time",
            active_graph_map["bars_over_time"],
        ),
        (
            bars_over_scenarios,
            "bars_over_scenarios",
            active_graph_map["bars_over_scenarios"],
        ),
        (
            diff_xaxis_lines,
            "diff_xaxis_lines",
            active_graph_map["diff_xaxis_lines"],
        ),
        (
            diff_xaxis_bars,
            "diff_xaxis_bars",
            active_graph_map["diff_xaxis_bars"],
        ),
        (
            x_y_scatter,
            "x_y_scatter",
            active_graph_map["x_y_scatter"],
        ),
        (
            tornado,
            "tornado",
            active_graph_map["tornado"],
        ),
        (
            macc,
            "macc",
            active_graph_map["macc"],
        ),
    )

    # load shapes - ((graphing function, controller sheet name, T/F if graph is active))
    load_fns_sheets_active = (
        (
            load_shape_area,
            "load_shape_area",
            active_graph_map["load_shape_area"],
        ),
        (
            load_shape_disaggregated,
            "load_shape_disaggregated",
            active_graph_map["load_shape_disaggregated"],
        ),
        (
            multiple_load_shapes,
            "multiple_load_shapes",
            active_graph_map["multiple_load_shapes"],
        ),
    )

    # create result and load shape graphs
    for p, folder in paths_and_folder_names:
        df_result = pd.concat(
            [load_results(p), df_proxies], ignore_index=True, sort=True
        ).fillna(0)
        make_graphs(
            df=df_result,
            folder=folder,
            color_map=color_map,
            branch_maps=branch_maps,
            all_sce_group_params=all_sce_group_params,
            fns_sheets_active=result_fns_sheets_active,
        )

        df_shape = load_shapes(p)
        make_graphs(
            df=df_shape,
            folder=folder,
            color_map=color_map,
            branch_maps=branch_maps,
            all_sce_group_params=all_sce_group_params,
            fns_sheets_active=load_fns_sheets_active,
        )


def load_results(input_path: Path) -> pd.DataFrame:
    """
    function to load parsed results
    :param input_path: where the results are stored
    :return: DataFrame of results
    """
    df = pd.read_csv(input_path / "combined_results.csv", header=0, index_col=0)
    df = create_scenario_copies(df)
    return df


def load_shapes(input_path: Path) -> pd.DataFrame:
    """
    Function to load parsed load shapes
    :param input_path: where the results are stored
    :return: DataFrame of load shapes
    """
    df_loads = pd.read_csv(input_path / "shapes.csv", header=0, index_col=0)
    df_loads = create_scenario_copies(df_loads)
    return df_loads


def make_graphs(
    df: pd.DataFrame,
    folder: str,
    color_map: dict,
    branch_maps: dict,
    all_sce_group_params: dict,
    fns_sheets_active: tuple,
) -> None:
    """
    Function to make graphs from results
    :param df: DataFrame of results
    :param folder: name of folder the results are contained in (needs to align with folder col in controller)
    :param color_map: dict of keys to hex color values
    :param branch_maps: numerous dicts of LEAP branches --> groupings
    :param all_sce_group_params: info found in tab "scenario_group_params" of controller
    :param fns_sheets_active: tuple of tuples, where the inner tuple contains (fn, sheet name, on/off switch)
    :return: NA
    """

    # iterate through all graphing functions
    for fn, sheet, active in fns_sheets_active:
        if active:

            # read graph params from excel controller tab
            df_graphs = pd.read_excel(
                CONTROLLER_PATH / "controller.xlsm", sheet_name=sheet
            )
            df_graphs = df_graphs.fillna("")

            # make a graph for each row
            for _, row in df_graphs.iterrows():
                if (row["make_graph"]) and (row["folder"] == folder):
                    fn(
                        df_in=df,
                        color_map=color_map,
                        branch_maps=branch_maps,
                        sce_group_params=all_sce_group_params[row["group_id"]],
                        graph_params=row.to_dict(),
                    )


def lines_over_time(
    df_in: pd.DataFrame,
    color_map: dict,
    branch_maps: dict,
    sce_group_params: dict,
    graph_params: dict,
) -> None:
    """
    Graph result from a single scenario over time
    :param df_in: DataFrame of results
    :param color_map: dict of key --> hex color codes
    :param branch_maps: maps of LEAP branches --> subgroups
    :param sce_group_params: info from graph group in controller tab "scenario group params"
    :param graph_params: info from row of graphing tab of controller
    :return: NA - generates a graph
    """

    # establish fuel filter
    if graph_params["fuel_filter"] == "":
        fuel_filter = None
    else:
        fuel_filter = [fuel.strip() for fuel in graph_params["fuel_filter"].split(",")]

    # Evaluate result
    df_graph = form_df_graph(
        df_in=df_in,
        sce_group_params=sce_group_params,
        result=[result.strip() for result in graph_params["result"].split(",")],
        multiplier=graph_params["multiplier"],
        marginalize=graph_params["marginalize"],
        cumulative=graph_params["cumulative"],
        discount=graph_params["discount"],
        filter_yrs=False,
        branch_map=branch_maps[graph_params["branch_map_name"]],
        fuel_filter=fuel_filter,
        groupby=list(
            {"Scenario", graph_params["xcol"], graph_params["ycol"]} - {"Value"}
        ),
    )

    # Create graphic
    fig = go.Figure()
    for sce in sce_group_params["scenarios"]:
        df_sce = df_graph[df_graph["Scenario"] == sce].copy()
        fig.add_trace(
            go.Scatter(
                mode="lines",
                x=df_sce[graph_params["xcol"]],
                y=df_sce[graph_params["ycol"]],
                name=sce_group_params["name_map"][sce],
                showlegend=sce_group_params["include_in_legend_map"][sce],
                line=dict(
                    color=color_map[
                        sce_group_params[graph_params["color_col"] + "_map"][sce]
                    ],
                    dash=sce_group_params["line_map"][sce],
                ),
            )
        )

    fig = update_fig_styling(fig, graph_params)
    fig.write_image(
        FIGURES_PATH
        / f"{graph_params['fname']}_{graph_params['group_id']}{IMAGE_FORMAT}",
        scale=IMAGE_SCALE,
    )


def bars_over_time(
    df_in: pd.DataFrame,
    color_map: dict,
    branch_maps: dict,
    sce_group_params: dict,
    graph_params: dict,
) -> None:
    """
    Graph result from a single scenario. Can have stacked or grouped bars for each year
    :param df_in: DataFrame of result
    :param color_map: dict of key --> hex color codes
    :param branch_maps: maps of LEAP branches --> subgroups
    :param sce_group_params: info from graph group in controller tab "scenario group params"
    :param graph_params: info from row of graphing tab of controller
    :return: NA - generates a graph
    """

    if graph_params["fuel_filter"] == "":
        fuel_filter = None
    else:
        fuel_filter = [fuel.strip() for fuel in graph_params["fuel_filter"].split(",")]

    df_graph = form_df_graph(
        df_in=df_in,
        sce_group_params=sce_group_params,
        result=[result.strip() for result in graph_params["result"].split(",")],
        multiplier=graph_params["multiplier"],
        marginalize=graph_params["marginalize"],
        cumulative=graph_params["cumulative"],
        discount=graph_params["discount"],
        filter_yrs=False,
        branch_map=branch_maps[graph_params["branch_map_name"]],
        fuel_filter=fuel_filter,
        groupby=list(
            {
                "Scenario",
                "Year",
                graph_params["xcol"],
                graph_params["ycol"],
                graph_params["color_col"],
            }
            - {"Value"}
        ),
    )

    if not graph_params["grouped"]:
        fig = px.bar(
            df_graph,
            x=graph_params["xcol"],
            y=graph_params["ycol"],
            color=graph_params["color_col"],
            color_discrete_map=color_map,
        )
    else:
        fig = px.bar(
            df_graph,
            x=graph_params["xcol"],
            y=graph_params["ycol"],
            color=graph_params["color_col"],
            barmode="group",
            color_discrete_map=color_map,
        )

    if graph_params["include_sum"]:
        df_sum = pd.DataFrame(columns=[graph_params["xcol"], graph_params["ycol"]])
        for time_pt in df_graph[graph_params["xcol"]].unique():
            sum_in_t = df_graph[df_graph[graph_params["xcol"]] == time_pt][
                graph_params["ycol"]
            ].sum()
            df_sum.loc[len(df_sum.index)] = [time_pt, sum_in_t]
        # add line to graph showing sum
        fig.add_trace(
            go.Scatter(
                mode="lines",
                x=df_sum[graph_params["xcol"]],
                y=df_sum[graph_params["ycol"]],
                name="Total",
                showlegend=True,
                line=dict(
                    color="black",
                    dash="solid",
                ),
            )
        )

    fig = update_fig_styling(fig, graph_params)
    fig.write_image(
        FIGURES_PATH
        / f"{graph_params['fname']}_{graph_params['group_id']}{IMAGE_FORMAT}",
        scale=IMAGE_SCALE,
    )


def bars_over_scenarios(
    df_in: pd.DataFrame,
    color_map: dict,
    branch_maps: dict,
    sce_group_params: dict,
    graph_params: dict,
) -> None:
    """
    Graph result comparing results across multiple scenarios (in a single year)
    :param df_in: DataFrame of result
    :param color_map: dict of key --> hex color codes
    :param branch_maps: maps of LEAP branches --> subgroups
    :param sce_group_params: info from graph group in controller tab "scenario group params"
    :param graph_params: info from row of graphing tab of controller
    :return: NA - generates a graph
    """

    if graph_params["fuel_filter"] == "":
        fuel_filter = None
    else:
        fuel_filter = [fuel.strip() for fuel in graph_params["fuel_filter"].split(",")]

    groupby = {
        "Scenario",
        graph_params["xcol"],
        graph_params["ycol"],
        graph_params["color_col"],
    } - {"Value"}
    if graph_params["sort_by"] != "":
        sort_by = [sort_col.strip() for sort_col in graph_params["sort_by"].split(",")]
        groupby.update(set(sort_by))
        if graph_params["include_error_bars"]:
            groupby = set(list(groupby) + ["error_group"])

    df_graph = form_df_graph(
        df_in=df_in,
        sce_group_params=sce_group_params,
        result=[result.strip() for result in graph_params["result"].split(",")],
        multiplier=graph_params["multiplier"],
        marginalize=graph_params["marginalize"],
        cumulative=graph_params["cumulative"],
        discount=graph_params["discount"],
        filter_yrs=True,
        branch_map=branch_maps[graph_params["branch_map_name"]],
        fuel_filter=fuel_filter,
        groupby=list(groupby),
    )

    # TODO: make defining error bars its own function
    error_up_y = None
    error_down_y = None
    error_up_x = None
    error_down_x = None
    if graph_params["include_error_bars"]:
        groupby.update({"error_up", "error_down"})
        df_graph_error = pd.DataFrame(columns=list(groupby - set(["Scenario"])))
        for key, dfg in df_graph.groupby("error_group"):
            dfg = dfg.reset_index(drop=True)
            median = dfg["Value"].median()
            dfg["median"] = median
            dfg["error_up"] = abs(dfg["Value"].max() - median)
            dfg["error_down"] = abs(dfg["Value"].min() - median)

            df_graph_error = pd.concat(
                [df_graph_error, dfg], ignore_index=True, sort=True
            )

        df_graph_error.drop(columns="Value", inplace=True)
        df_graph_error.rename(columns={"median": "Value"}, inplace=True)
        df_graph = df_graph_error

        if graph_params["xcol"] == "Value":
            error_up_x = "error_up"
            error_down_x = "error_down"
        else:
            error_up_y = "error_up"
            error_down_y = "error_down"

    # sort dataframe
    category_orders = dict()
    if graph_params["sort_by"] != "":
        sort_by = [sort_col.strip() for sort_col in graph_params["sort_by"].split(",")]
        df_graph = df_graph.sort_values(
            by=sort_by, ignore_index=True, ascending=graph_params["sort_ascending"]
        )
        if graph_params["xcol"] == "Value":
            category_orders = {
                graph_params["ycol"]: df_graph[graph_params["ycol"]].tolist()
            }
        else:
            category_orders = {
                graph_params["xcol"]: df_graph[graph_params["xcol"]].tolist()
            }

    # set up text annotations
    text_auto = False
    if graph_params["annotate"]:
        text_auto = graph_params["annotation_style"]

    if graph_params["markers_instead_of_bars"]:
        fig = px.scatter(
            df_graph,
            x=graph_params["xcol"],
            y=graph_params["ycol"],
            color=graph_params["color_col"],
            color_discrete_map=color_map,
            category_orders=category_orders,
            error_x=error_up_x,
            error_x_minus=error_down_x,
            error_y=error_up_y,
            error_y_minus=error_down_y,
        )
    elif not graph_params["grouped"]:
        fig = px.bar(
            df_graph,
            x=graph_params["xcol"],
            y=graph_params["ycol"],
            text_auto=text_auto,
            color=graph_params["color_col"],
            color_discrete_map=color_map,
            category_orders=category_orders,
            error_x=error_up_x,
            error_x_minus=error_down_x,
            error_y=error_up_y,
            error_y_minus=error_down_y,
        )
    else:
        fig = px.bar(
            df_graph,
            x=graph_params["xcol"],
            y=graph_params["ycol"],
            text_auto=text_auto,
            color=graph_params["color_col"],
            barmode="group",
            color_discrete_map=color_map,
            category_orders=category_orders,
            error_x=error_up_x,
            error_x_minus=error_down_x,
            error_y=error_up_y,
            error_y_minus=error_down_y,
        )

    fig = update_fig_styling(fig, graph_params)

    if graph_params["mark_sum"]:
        groupby = list({graph_params["xcol"], graph_params["ycol"]} - {"Value"})
        df_graph = df_graph.groupby(by=groupby, as_index=False)["Value"].sum()
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=df_graph[graph_params["xcol"]],
                y=df_graph[graph_params["ycol"]],
                name="Total",
                showlegend=True,
                marker=dict(
                    color="LightSkyBlue",
                    size=10,
                    line=dict(color="MediumPurple", width=2),
                ),
            )
        )

    if graph_params["annotate_sum"]:
        groupby = list({graph_params["xcol"], graph_params["ycol"]} - {"Value"})
        df_graph = df_graph.groupby(by=groupby, as_index=False)["Value"].sum()
        for _, row in df_graph.iterrows():
            fig.add_annotation(
                x=row[graph_params["xcol"]],
                y=row[graph_params["ycol"]],
                text=f"{row['Value']:{graph_params['annotation_style']}}",
                xshift=0,
                yshift=0,
                showarrow=False,
            )

    fig.write_image(
        FIGURES_PATH
        / f"{graph_params['fname']}_{graph_params['group_id']}{IMAGE_FORMAT}",
        scale=IMAGE_SCALE,
    )


def diff_xaxis_lines(
    df_in: pd.DataFrame,
    color_map: dict,
    branch_maps: dict,
    sce_group_params: dict,
    graph_params: dict,
) -> None:
    """
    Graph results from multiple scenarios where each scenario has its own x-axis value
    :param df_in: DataFrame of result
    :param color_map: dict of key --> hex color codes
    :param branch_maps: maps of LEAP branches --> subgroups
    :param sce_group_params: info from graph group in controller tab "scenario group params"
    :param graph_params: info from row of graphing tab of controller
    :return: NA - generates a graph
    """

    if graph_params["fuel_filter"] == "":
        fuel_filter = None
    else:
        fuel_filter = [fuel.strip() for fuel in graph_params["fuel_filter"].split(",")]

    groupby = {
        "Scenario",
        graph_params["xcol"],
        graph_params["ycol"],
        graph_params["color_col"],
    } - {"Value"}
    if graph_params["sort_by"] != "":
        sort_by = [sort_col.strip() for sort_col in graph_params["sort_by"].split(",")]
        groupby.update(set(sort_by))

    df_graph = form_df_graph(
        df_in=df_in,
        sce_group_params=sce_group_params,
        result=[result.strip() for result in graph_params["result"].split(",")],
        multiplier=graph_params["multiplier"],
        marginalize=graph_params["marginalize"],
        cumulative=graph_params["cumulative"],
        discount=graph_params["discount"],
        filter_yrs=True,
        branch_map=branch_maps[graph_params["branch_map_name"]],
        fuel_filter=fuel_filter,
        groupby=list(groupby),
    )

    if graph_params["sort_by"] != "":
        sort_by = [sort_col.strip() for sort_col in graph_params["sort_by"].split(",")]
        df_graph = df_graph.sort_values(
            by=sort_by, ascending=graph_params["sort_ascending"], ignore_index=True
        )

    fig = px.line(
        df_graph,
        x=graph_params["xcol"],
        y=graph_params["ycol"],
        color=graph_params["color_col"],
        color_discrete_map=color_map,
        markers=graph_params["include_markers"],
    )

    fig = update_fig_styling(fig, graph_params)
    fig.write_image(
        FIGURES_PATH
        / f"{graph_params['fname']}_{graph_params['group_id']}{IMAGE_FORMAT}",
        scale=IMAGE_SCALE,
    )


def diff_xaxis_bars(
    df_in: pd.DataFrame,
    color_map: dict,
    branch_maps: dict,
    sce_group_params: dict,
    graph_params: dict,
) -> None:
    """
    Graph result from multiple scenarios where each scenario is a bar at a different x axis value
    :param df_in: DataFrame of result
    :param color_map: dict of key --> hex color codes
    :param branch_maps: maps of LEAP branches --> subgroups
    :param sce_group_params: info from graph group in controller tab "scenario group params"
    :param graph_params: info from row of graphing tab of controller
    :return: NA - generates a graph
    """

    if graph_params["fuel_filter"] == "":
        fuel_filter = None
    else:
        fuel_filter = [fuel.strip() for fuel in graph_params["fuel_filter"].split(",")]

    df_graph = form_df_graph(
        df_in=df_in,
        sce_group_params=sce_group_params,
        result=[result.strip() for result in graph_params["result"].split(",")],
        multiplier=graph_params["multiplier"],
        marginalize=graph_params["marginalize"],
        cumulative=graph_params["cumulative"],
        discount=graph_params["discount"],
        filter_yrs=True,
        branch_map=branch_maps[graph_params["branch_map_name"]],
        fuel_filter=fuel_filter,
        groupby=list(
            {
                "Scenario",
                graph_params["xcol"],
                graph_params["ycol"],
                graph_params["color_col"],
            }
            - {"Value"}
        ),
    )

    # set up text annotations
    text_auto = False
    if graph_params["annotate"]:
        text_auto = graph_params["annotation_style"]

    if not graph_params["grouped"]:
        fig = px.bar(
            df_graph,
            x=graph_params["xcol"],
            y=graph_params["ycol"],
            text_auto=text_auto,
            color=graph_params["color_col"],
            color_discrete_map=color_map,
        )
    else:
        fig = px.bar(
            df_graph,
            x=graph_params["xcol"],
            y=graph_params["ycol"],
            text_auto=text_auto,
            color=graph_params["color_col"],
            barmode="group",
            color_discrete_map=color_map,
        )

    fig = update_fig_styling(fig, graph_params)
    fig.write_image(
        FIGURES_PATH
        / f"{graph_params['fname']}_{graph_params['group_id']}{IMAGE_FORMAT}",
        scale=IMAGE_SCALE,
    )


def x_y_scatter(
    df_in: pd.DataFrame,
    color_map: dict,
    branch_maps: dict,
    sce_group_params: dict,
    graph_params: dict,
) -> None:
    """
    Graph two results against eachother for multiple scenarios (one result on x axis, the other on y)
    :param df_in: DataFrame of result
    :param color_map: dict of key --> hex color codes
    :param branch_maps: maps of LEAP branches --> subgroups
    :param sce_group_params: info from graph group in controller tab "scenario group params"
    :param graph_params: info from row of graphing tab of controller
    :return: NA - generates a graph
    """

    if graph_params["fuel_filter"] == "":
        fuel_filter = None
    else:
        fuel_filter = [fuel.strip() for fuel in graph_params["fuel_filter"].split(",")]

    df_graph_x = form_df_graph(
        df_in=df_in,
        sce_group_params=sce_group_params,
        result=[result.strip() for result in graph_params["result_x"].split(",")],
        multiplier=graph_params["multiplier_x"],
        marginalize=graph_params["marginalize_x"],
        cumulative=graph_params["cumulative_x"],
        discount=graph_params["discount_x"],
        filter_yrs=True,
        branch_map=branch_maps[graph_params["branch_map_name"]],
        fuel_filter=fuel_filter,
        groupby=["Scenario"],
    )
    df_graph_x = df_graph_x.rename(columns={"Value": "Value_x"})

    df_graph_y = form_df_graph(
        df_in=df_in,
        sce_group_params=sce_group_params,
        result=[result.strip() for result in graph_params["result_y"].split(",")],
        multiplier=graph_params["multiplier_y"],
        marginalize=graph_params["marginalize_y"],
        cumulative=graph_params["cumulative_y"],
        discount=graph_params["discount_y"],
        filter_yrs=True,
        branch_map=branch_maps[graph_params["branch_map_name"]],
        fuel_filter=fuel_filter,
        groupby=["Scenario"],
    )
    df_graph_y = df_graph_y.rename(columns={"Value": "Value_y"})
    df_graph = df_graph_x.merge(df_graph_y, how="outer")

    fig = go.Figure()
    for sce in sce_group_params["scenarios"]:
        df_sce = df_graph[df_graph["Scenario"] == sce].copy()
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=df_sce["Value_x"],
                y=df_sce["Value_y"],
                name=sce_group_params["name_map"][sce],
                showlegend=sce_group_params["include_in_legend_map"][sce],
                marker_symbol=sce_group_params["marker_map"][sce],
                marker_color=color_map[
                    sce_group_params[graph_params["color_col"] + "_map"][sce]
                ],
            )
        )

    fig.update_traces(marker={"size": 10})

    fig = update_fig_styling(fig, graph_params)
    fig.write_image(
        FIGURES_PATH
        / f"{graph_params['fname']}_{graph_params['group_id']}{IMAGE_FORMAT}",
        scale=IMAGE_SCALE,
    )


def tornado(
    df_in: pd.DataFrame,
    color_map: dict,
    branch_maps: dict,
    sce_group_params: dict,
    graph_params: dict,
) -> None:
    """
    Graph results from multiple scenarios. Multiple scenarios can form each bar in the tornado.
    :param df_in: DataFrame of result
    :param color_map: dict of key --> hex color codes
    :param branch_maps: maps of LEAP branches --> subgroups
    :param sce_group_params: info from graph group in controller tab "scenario group params"
    :param graph_params: info from row of graphing tab of controller
    :return: NA - generates a graph
    """

    if graph_params["fuel_filter"] == "":
        fuel_filter = None
    else:
        fuel_filter = [fuel.strip() for fuel in graph_params["fuel_filter"].split(",")]

    df = form_df_graph(
        df_in=df_in,
        sce_group_params=sce_group_params,
        result=[result.strip() for result in graph_params["result"].split(",")],
        multiplier=graph_params["multiplier"],
        marginalize=graph_params["marginalize"],
        cumulative=graph_params["cumulative"],
        discount=graph_params["discount"],
        filter_yrs=True,
        branch_map=branch_maps[graph_params["branch_map_name"]],
        fuel_filter=fuel_filter,
        groupby=list(
            {
                "Scenario",
                "tornado_group_name",
                "tornado_order_id",
                graph_params["color_col"],
            }
            - {"Value"}
        ),
    )

    df_graph = pd.DataFrame(
        columns=[
            "tornado_group_name",
            "tornado_order_id",
            "bar_min",
            "bar_height",
            "bar_max",
            graph_params["color_col"],
        ]
    )
    for key, dfg in df.groupby(by=["tornado_group_name"]):
        df_graph.loc[len(df_graph.index)] = [
            key,  # tornado_group_name
            dfg["tornado_order_id"].unique()[0],  # tornado_order_id
            dfg["Value"].min(),  # bar_min
            dfg["Value"].max() - dfg["Value"].min(),  # bar_height
            dfg["Value"].max(),  # bar_max
            dfg[graph_params["color_col"]].unique()[0],  # color_id
        ]

    category_orders = dict()
    if graph_params["sort_by"] != "":
        sort_by = [sort_col.strip() for sort_col in graph_params["sort_by"].split(",")]
        df_graph = df_graph.sort_values(
            by=sort_by, ignore_index=True, ascending=graph_params["sort_ascending"]
        )
        if graph_params["xcol"] == "bar_height":
            category_orders = {
                graph_params["ycol"]: df_graph[graph_params["ycol"]].tolist()
            }
        else:
            category_orders = {
                graph_params["xcol"]: df_graph[graph_params["xcol"]].tolist()
            }

    fig = px.bar(
        df_graph,
        x=graph_params["xcol"],
        y=graph_params["ycol"],
        base="bar_min",
        color=graph_params["color_col"],
        color_discrete_map=color_map,
        category_orders=category_orders,
    )

    # add text labels to both side of the tornado bar
    if graph_params["annotate_tornado"]:
        if graph_params["xcol"] == "tornado_group_name":
            x_text_pos = np.array(
                list(
                    zip(df_graph["tornado_group_name"], df_graph["tornado_group_name"])
                )
            ).flatten()
            y_text_pos = np.array(
                list(
                    zip(
                        df_graph["bar_min"] + df_graph["bar_height"] * 0.2,
                        df_graph["bar_max"] - df_graph["bar_height"] * 0.2,
                    )
                )
            ).flatten()
            text = [
                f"{y:.1f}"
                for y in np.array(
                    list(zip(df_graph["bar_min"], df_graph["bar_max"]))
                ).flatten()
            ]
        else:
            y_text_pos = np.array(
                list(
                    zip(df_graph["tornado_group_name"], df_graph["tornado_group_name"])
                )
            ).flatten()
            x_text_pos = np.array(
                list(
                    zip(
                        df_graph["bar_min"] + df_graph["bar_height"] * 0.2,
                        df_graph["bar_max"] - df_graph["bar_height"] * 0.2,
                    )
                )
            ).flatten()
            text = [
                f"{x:.1f}"
                for x in np.array(
                    list(zip(df_graph["bar_min"], df_graph["bar_max"]))
                ).flatten()
            ]

        fig.add_trace(
            go.Scatter(
                x=x_text_pos,
                y=y_text_pos,
                text=text,
                mode="text",
                showlegend=False,
            )
        )

    fig = update_fig_styling(fig, graph_params)
    fig.write_image(
        FIGURES_PATH
        / f"{graph_params['fname']}_{graph_params['group_id']}{IMAGE_FORMAT}",
        scale=IMAGE_SCALE,
    )


def macc(
    df_in: pd.DataFrame,
    color_map: dict,
    branch_maps: dict,
    sce_group_params: dict,
    graph_params: dict,
) -> None:
    """
    Form MACC plot
    :param df_in: DataFrame of result
    :param color_map: dict of key --> hex color codes
    :param branch_maps: maps of LEAP branches --> subgroups
    :param sce_group_params: info from graph group in controller tab "scenario group params"
    :param graph_params: info from row of graphing tab of controller
    :return: NA - generates a graph
    """

    if graph_params["fuel_filter"] == "":
        fuel_filter = None
    else:
        fuel_filter = [fuel.strip() for fuel in graph_params["fuel_filter"].split(",")]

    df_graph_x = form_df_graph(
        df_in=df_in,
        sce_group_params=sce_group_params,
        result=[result.strip() for result in graph_params["result_x"].split(",")],
        multiplier=graph_params["multiplier_x"],
        marginalize=graph_params["marginalize_x"],
        cumulative=graph_params["cumulative_x"],
        discount=graph_params["discount_x"],
        filter_yrs=True,
        branch_map=branch_maps[graph_params["branch_map_name"]],
        fuel_filter=fuel_filter,
        groupby=["Scenario", graph_params["color_col"]],
    )
    df_graph_x = df_graph_x.rename(columns={"Value": "Value_x"})

    df_graph_y = form_df_graph(
        df_in=df_in,
        sce_group_params=sce_group_params,
        result=[result.strip() for result in graph_params["result_y"].split(",")],
        multiplier=graph_params["multiplier_y"],
        marginalize=graph_params["marginalize_y"],
        cumulative=graph_params["cumulative_y"],
        discount=graph_params["discount_y"],
        filter_yrs=True,
        branch_map=branch_maps[graph_params["branch_map_name"]],
        fuel_filter=fuel_filter,
        groupby=["Scenario", graph_params["color_col"]],
    )
    df_graph_y = df_graph_y.rename(columns={"Value": "Value_y"})
    df_graph = df_graph_x.merge(df_graph_y, how="outer")

    df_graph = df_graph.sort_values(by="Value_y", axis=0, ignore_index=True)
    df_graph["end_range_x"] = df_graph["Value_x"].cumsum()

    df_graph["start_range_x"] = 0
    for i in range(1, len(df_graph)):
        df_graph.loc[i, "start_range_x"] = df_graph.loc[i - 1, "end_range_x"]

    df_graph["width"] = df_graph["end_range_x"] - df_graph["start_range_x"]
    df_graph["mid_x"] = (df_graph["end_range_x"] + df_graph["start_range_x"]) / 2.0

    fig = go.Figure()
    legend_entries = []
    for sce in sce_group_params["scenarios"]:
        df_sce = df_graph[df_graph["Scenario"] == sce].copy()
        fig.add_trace(
            go.Bar(
                x=df_sce["mid_x"],
                width=df_sce["width"],
                y=df_sce["Value_y"],
                name=df_sce[graph_params["color_col"]].unique()[0],
                showlegend=df_sce[graph_params["color_col"]].unique()[0]
                not in legend_entries,
                marker=dict(
                    color=color_map[df_sce[graph_params["color_col"]].unique()[0]],
                ),
            )
        )
        legend_entries.append(df_sce[graph_params["color_col"]].unique()[0])

    # text annotations
    for sce in sce_group_params["scenarios"]:
        df_sce = df_graph[df_graph["Scenario"] == sce].copy()
        fig.add_annotation(
            x=df_sce["mid_x"].unique()[0],
            y=max(0, df_sce["Value_y"].unique()[0]) + 90,
            text=sce_group_params["name_map"][sce],
            textangle=90,
            showarrow=True,
            startarrowsize=0.3,
            yanchor="auto",
            yshift=0,
            font=dict(
                color="black",
                size=8.5,
            ),
        )

    # add in tick marks
    fig.update_xaxes(showgrid=True, ticks="outside", tickson="boundaries", ticklen=10)
    fig.update_yaxes(showgrid=True, ticks="outside", tickson="boundaries", ticklen=10)

    fig.update_traces(textfont_size=10, textposition="inside")
    fig.update_layout(uniformtext_minsize=6, uniformtext_mode="hide")

    fig = update_fig_styling(fig, graph_params)
    fig.write_image(
        FIGURES_PATH
        / f"{graph_params['fname']}_{graph_params['group_id']}{IMAGE_FORMAT}",
        scale=IMAGE_SCALE,
    )


def update_fig_styling(fig: go.Figure, graph_params: dict) -> go.Figure:
    """
    Update figure according to graph_parmas
    :param fig: figure
    :param graph_params: graph parameters
    :return: figure
    """

    # update title, xaxis title, and yaxis title
    fig = update_titles(
        fig,
        graph_params["title"],
        graph_params["xaxis_title"],
        graph_params["yaxis_title"],
    )

    # update plot size
    fig = update_plot_size(fig, graph_params["plot_width"], graph_params["plot_height"])

    # update legend title and positioning
    fig.update_layout(legend_title="")
    if graph_params["legend_position"] == "below":
        fig = place_legend_below(fig, graph_params["xaxis_title"])
    elif graph_params["legend_position"] == "hide":
        fig.update_layout(showlegend=False)

    # axis settings
    if graph_params["yaxis_to_zero"]:
        fig.update_yaxes(rangemode="tozero")

    if graph_params["xaxis_to_zero"]:
        fig.update_xaxes(rangemode="tozero")

    if graph_params["xaxis_lim"] != "":
        fig.update_xaxes(
            range=[float(val) for val in graph_params["xaxis_lim"].split(",")]
        )

    if graph_params["yaxis_lim"] != "":
        fig.update_yaxes(
            range=[float(val) for val in graph_params["yaxis_lim"].split(",")]
        )

    # update text annotations
    if "annotate" in graph_params:
        if graph_params["annotate"]:
            fig.update_traces(
                textfont_size=14,
                textposition=graph_params["annotation_position"],
                textangle=graph_params["annotation_angle"],
            )
            fig.update_layout(
                uniformtext_minsize=10,
                uniformtext_mode="hide",
            )

    return fig


def load_shape_area(
    df_in: pd.DataFrame,
    color_map: dict,
    branch_maps: dict,
    sce_group_params: dict,
    graph_params: dict,
) -> None:
    """
    Function to make a graph of load shape that is a stacked area plot (one color per sector)
    :param df_in: DataFrame of result
    :param color_map: dict of key --> hex color codes
    :param branch_maps: maps of LEAP branches --> subgroups
    :param sce_group_params: info from graph group in controller tab "scenario group params"
    :param graph_params: info from row of graphing tab of controller
    :return: NA - generates a graph
    """

    df_graph = form_df_graph_load(
        df_in, sce_group_params, graph_params, sum_across_branches=False
    )

    fig = px.area(
        df_graph,
        x=graph_params["xcol"],
        y=graph_params["ycol"],
        color=graph_params["color_col"],
        color_discrete_map=color_map,
        line_shape="spline",
    )
    fig = update_to_load_shape_layout(fig)
    fig = update_fig_styling(fig, graph_params)
    fig.write_image(
        FIGURES_PATH
        / f"{graph_params['fname']}_{graph_params['group_id']}{IMAGE_FORMAT}",
        scale=IMAGE_SCALE,
    )


def load_shape_disaggregated(
    df_in: pd.DataFrame,
    color_map: dict,
    branch_maps: dict,
    sce_group_params: dict,
    graph_params: dict,
) -> None:
    """
    Function to generate graph where each sector has its own line for its individual load
    :param df_in: DataFrame of result
    :param color_map: dict of key --> hex color codes
    :param branch_maps: maps of LEAP branches --> subgroups
    :param sce_group_params: info from graph group in controller tab "scenario group params"
    :param graph_params: info from row of graphing tab of controller
    :return: NA - generates a graph
    """

    df_graph = form_df_graph_load(
        df_in, sce_group_params, graph_params, sum_across_branches=False
    )

    fig = px.line(
        df_graph,
        x=graph_params["xcol"],
        y=graph_params["ycol"],
        color=graph_params["color_col"],
        color_discrete_map=color_map,
    )

    if graph_params["include_sum"]:
        df_sum = pd.DataFrame(columns=[graph_params["xcol"], graph_params["ycol"]])
        for time_pt in df_graph[graph_params["xcol"]].unique():
            sum_in_t = df_graph[df_graph[graph_params["xcol"]] == time_pt][
                graph_params["ycol"]
            ].sum()
            df_sum.loc[len(df_sum.index)] = [time_pt, sum_in_t]
        # add line to graph showing sum
        fig.add_trace(
            go.Scatter(
                mode="lines",
                x=df_sum[graph_params["xcol"]],
                y=df_sum[graph_params["ycol"]],
                name="Total",
                showlegend=True,
                line=dict(
                    color="black",
                    dash="solid",
                ),
            )
        )

    fig = update_to_load_shape_layout(fig)
    fig = update_fig_styling(fig, graph_params)
    fig.write_image(
        FIGURES_PATH
        / f"{graph_params['fname']}_{graph_params['group_id']}{IMAGE_FORMAT}",
        scale=IMAGE_SCALE,
    )


def multiple_load_shapes(
    df_in: pd.DataFrame,
    color_map: dict,
    branch_maps: dict,
    sce_group_params: dict,
    graph_params: dict,
) -> None:
    """
    Make graph with multiple load shapes compared
    :param df_in: DataFrame of result
    :param color_map: dict of key --> hex color codes
    :param branch_maps: maps of LEAP branches --> subgroups
    :param sce_group_params: info from graph group in controller tab "scenario group params"
    :param graph_params: info from row of graphing tab of controller
    :return: NA - generates a graph
    """

    df_graph = form_df_graph_load(
        df_in, sce_group_params, graph_params, sum_across_branches=True
    )

    fig = px.line(
        df_graph,
        x=graph_params["xcol"],
        y=graph_params["ycol"],
        color=graph_params["color_col"],
        color_discrete_map=color_map,
    )
    fig = update_to_load_shape_layout(fig)
    fig = update_fig_styling(fig, graph_params)
    fig.write_image(
        FIGURES_PATH
        / f"{graph_params['fname']}_{graph_params['group_id']}{IMAGE_FORMAT}",
        scale=IMAGE_SCALE,
    )


def update_to_load_shape_layout(fig: go.Figure) -> go.Figure:
    """
    Function to update figure to have load shape layout (months instead of years...)
    :param fig: figure
    :return: updated figure
    """
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=np.arange(12, 289, 24),
            ticktext=months,
            showgrid=False,
            minor=dict(
                tickvals=np.arange(0, 289, 24), showgrid=True, gridcolor="#FFFFFF"
            ),
        )
    )
    month_ends = np.arange(0, 289, 24)
    for i, (x0, x1) in enumerate(zip(month_ends, month_ends[1:])):
        if i % 2 == 0:
            continue
        else:
            fig.add_vrect(x0=x0, x1=x1, line_width=0, fillcolor="grey", opacity=0.1)
    return fig


def update_titles(
    fig: go.Figure,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
) -> go.Figure:
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )
    return fig


def place_legend_below(
    fig: go.Figure,
    xaxis_title: str,
) -> go.Figure:
    if xaxis_title == "":
        y = -0.08
    else:
        y = -0.2

    fig.update_layout(
        legend_title="",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=y,
            xanchor="left",
            x=0,
        ),
    )
    return fig


def update_plot_size(
    fig: go.Figure, width: float = 800, height: float = 500
) -> go.Figure:

    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
    )
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)
    return fig


if __name__ == "__main__":
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    main()
