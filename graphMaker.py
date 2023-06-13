import pandas as pd
import itertools
from pathlib import Path
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Paths
INPUT_PATH_112_28 = Path("resultsFiles/112_28results/clean_results")
INPUT_PATH_112_31 = Path("resultsFiles/112_31results/clean_results")
CONTROLLER_PATH = Path("resultsFiles/results_controller")
FIGURES_PATH = Path("resultsFiles/new_figures")

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

# Fuels that are equivalent
FUELS_TO_COMBINE = {
    "CRNG": "RNG",
    "CNG": "NG",
    "Hydrogen Transmitted": "Hydrogen"
}

# Image formatting
IMAGE_FORMAT = ".pdf"
IMAGE_SCALE = 1


def main():
    # load data and make copies of scenarios specified in the controller
    df_112_28, df_112_28_loads = load_data(INPUT_PATH_112_28)
    df_112_31, df_112_31_loads = load_data(INPUT_PATH_112_31)


    # create color and branch maps
    color_map = load_map('color_map')
    active_graph_map = load_map('active_graph_map')
    branch_maps = form_branch_maps(pd.concat([df_112_28, df_112_31], ignore_index=True, sort=True))

    # read in scenario group parameters
    _, all_sce_group_params = form_sce_group_params()

    result_fns_sheets_active = (
        (lines_over_time, 'lines_over_time', active_graph_map['lines_over_time']),
        (bars_over_time, 'bars_over_time', active_graph_map['bars_over_time']),
        (bars_over_scenarios, 'bars_over_scenarios', active_graph_map['bars_over_scenarios']),
        (diff_xaxis_lines, 'diff_xaxis_lines', active_graph_map['diff_xaxis_lines']),
        (diff_xaxis_bars, 'diff_xaxis_bars', active_graph_map['diff_xaxis_bars']),
        (x_y_scatter, 'x_y_scatter', active_graph_map['x_y_scatter']),
        (tornado, 'tornado', active_graph_map['tornado']),
        (macc, 'macc', active_graph_map['macc']),
    )

    load_fns_sheets_active = (
        (load_shape_area, 'load_shape_area', active_graph_map['load_shape_area']),
        (load_shape_disaggregated, 'load_shape_disaggregated', active_graph_map['load_shape_disaggregated']),
        (multiple_load_shapes, 'multiple_load_shapes', active_graph_map['multiple_load_shapes']),
    )

    # create result graphs
    result_graphs(
        df=df_112_28,
        folder='112_28results',
        color_map=color_map,
        branch_maps=branch_maps,
        all_sce_group_params=all_sce_group_params,
        fns_sheets_active=result_fns_sheets_active,
    )

    result_graphs(
        df=df_112_31,
        folder='112_31results',
        color_map=color_map,
        branch_maps=branch_maps,
        all_sce_group_params=all_sce_group_params,
        fns_sheets_active=result_fns_sheets_active,
    )

    # load shape graphs
    load_graphs(
        df=df_112_31_loads,
        folder='112_31results',
        color_map=color_map,
        all_sce_group_params=all_sce_group_params,
        fns_sheets_active=load_fns_sheets_active,
    )

    load_graphs(
        df=df_112_28_loads,
        folder='112_28results',
        color_map=color_map,
        all_sce_group_params=all_sce_group_params,
        fns_sheets_active=load_fns_sheets_active,
    )


def load_data(input_path):
    df = pd.read_csv(input_path / 'combined_results.csv', header=0, index_col=0)
    df = create_scenario_copies(df)
    df_loads = pd.read_csv(input_path / "shapes.csv", header=0, index_col=0)
    df_loads = create_scenario_copies(df_loads)

    return df, df_loads


def result_graphs(df, folder, color_map, branch_maps, all_sce_group_params, fns_sheets_active):
    """
    Function to make graphs from results
    :param df: DataFrame of results
    :param color_map: dict of keys to hex color values
    :param branch_maps: numerous dicts of LEAP branches --> groupings
    :param all_sce_group_params: info found in tab "scenario group params" of controller
    :param fns_sheets_active: tuple of tuples, where the inner tuple contains (fn, sheet name, on/off switch)
    :return: NA
    """

    # iterate through all graphing functions
    for fn, sheet, active in fns_sheets_active:
        if active:

            # read graph params from excel controller tab
            df_graphs = pd.read_excel(CONTROLLER_PATH / 'controller.xlsm', sheet_name=sheet)
            df_graphs = df_graphs.fillna('')

            # make a graph for each row
            for _, row in df_graphs.iterrows():
                if (row['make_graph']) and (row['folder'] == folder):
                    fn(
                        df_in=df,
                        color_map=color_map,
                        branch_maps=branch_maps,
                        sce_group_params=all_sce_group_params[row['group_id']],
                        graph_params=row.to_dict(),
                    )


def load_graphs(df, folder, color_map, all_sce_group_params, fns_sheets_active):
    """
    Function to make graphs of load shapes from results
    :param df: DataFrame of results
    :param color_map: dict of keys to hex color values
    :param all_sce_group_params: info found in tab "scenario group params" of controller
    :param fns_sheets_active: tuple of tuples, where the inner tuple contains (fn, sheet name, on/off switch)
    :return: NA
    """

    # iterate through all graphing functions
    for fn, sheet, active in fns_sheets_active:
        if active:

            # read graph params from excel controller tab
            df_graphs = pd.read_excel(CONTROLLER_PATH / 'controller.xlsm', sheet_name=sheet)
            df_graphs = df_graphs.fillna('')

            # make a graph for each row
            for _, row in df_graphs.iterrows():
                if (row['make_graph']) and (row['folder'] == folder):
                    fn(
                        df_in=df,
                        color_map=color_map,
                        sce_group_params=all_sce_group_params[row['group_id']],
                        graph_params=row.to_dict(),
                    )


def form_df_graph(df_in, sce_group_params, result, multiplier, marginalize, cumulative, discount, filter_yrs,
                  branch_map, fuel_filter, groupby):
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
    df_graph = df_in[df_in['Scenario'].isin(sce_group_params['relevant_scenarios'])].copy()

    # Calculate result (special function for cost of abatement)
    if result == ['cost of abatement']:
        # calculate cost of abatement
        df_graph_1 = evaluate_dollar_per_ton_abated(
            df_in=df_graph,
            subgroup_dict=branch_map,
            relative_to_map=sce_group_params['relative_to_map'],
        )
        # get any cost of abatements published by CARB
        df_graph_2 = calculate_annual_result_by_subgroup(df_graph, result, branch_map)
        # combine results
        df_graph = pd.concat([df_graph_1, df_graph_2], ignore_index=True, sort=True)
        df_graph['Value'] = df_graph['Value'] * multiplier
    else:
        # calculate result
        df_graph = calculate_annual_result_by_subgroup(df_graph, result, branch_map)
        df_graph['Value'] = df_graph['Value'] * multiplier

        # combine fuels
        df_graph = df_graph.replace({"Fuel": FUELS_TO_COMBINE})

        # if specified, filter for specific fuels
        if fuel_filter is not None:
            df_graph = df_graph[df_graph['Fuel'].isin(fuel_filter)].copy()

        # discount, marginalize, cumsum
        if discount:
            df_graph = discount_it(df_graph)
        if marginalize:
            df_graph = marginalize_it(df_graph, sce_group_params['relative_to_map'])
        if cumulative:
            df_graph = cumsum_it(df_graph)

    # get rid of years not specified to be included
    if filter_yrs:
        for sce, yr in sce_group_params['specified_year_map'].items():
            df_graph = df_graph.reset_index(drop=True)
            rows_to_drop = np.array(
                (df_graph['Scenario'] == sce) &
                (df_graph['Year'] != yr)
            )
            row_ids_to_drop = list(np.where(rows_to_drop)[0])
            df_graph = df_graph.drop(index=row_ids_to_drop)

    # get rid of unneeded scenarios
    df_graph = df_graph[df_graph['Scenario'].isin(sce_group_params['scenarios'])].copy()

    # add columns based on the relevant maps (name_map, color_map...)
    for k, v in sce_group_params.items():
        if k.endswith('_map'):
            df_graph[k[:-4]] = df_graph['Scenario'].map(v)

    # sum values within the same year, scenario, specified color
    df_graph = df_graph.groupby(by=groupby, as_index=False)['Value'].sum()

    return df_graph


def lines_over_time(df_in, color_map, branch_maps, sce_group_params, graph_params):
    """
    Graph result from a single scenario over time
    :param df_in: DataFrame of result
    :param color_map: dict of key --> hex color codes
    :param branch_maps: maps of LEAP branches --> subgroups
    :param sce_group_params: info from graph group in controller tab "scenario group params"
    :param graph_params: info from row of graphing tab of controller
    :return: NA - generates a graph
    """

    # establish fuel filter
    if graph_params['fuel_filter'] == '':
        fuel_filter = None
    else:
        fuel_filter = [fuel.strip() for fuel in graph_params['fuel_filter'].split(',')]

    # Evaluate result
    df_graph = form_df_graph(
        df_in=df_in,
        sce_group_params=sce_group_params,
        result=[result.strip() for result in graph_params['result'].split(',')],
        multiplier=graph_params['multiplier'],
        marginalize=graph_params['marginalize'],
        cumulative=graph_params['cumulative'],
        discount=graph_params['discount'],
        filter_yrs=False,
        branch_map=branch_maps[graph_params['branch_map_name']],
        fuel_filter=fuel_filter,
        groupby=list({'Scenario', graph_params['xcol'], graph_params['ycol']} - {'Value'})
    )

    # Create graphic
    fig = go.Figure()
    for sce in sce_group_params['scenarios']:
        df_sce = df_graph[df_graph['Scenario'] == sce].copy()
        fig.add_trace(go.Scatter(
            mode='lines',
            x=df_sce[graph_params['xcol']],
            y=df_sce[graph_params['ycol']],
            name=sce_group_params['name_map'][sce],
            showlegend=sce_group_params['include_in_legend_map'][sce],
            line=dict(
                color=color_map[sce_group_params[graph_params['color_col'] + '_map'][sce]],
                dash=sce_group_params['line_map'][sce],
            ),
        ))

    fig = update_fig_styling(fig, graph_params)
    fig.write_image(FIGURES_PATH / f"{graph_params['fname']}_{graph_params['group_id']}{IMAGE_FORMAT}",
                    scale=IMAGE_SCALE)


def bars_over_time(df_in, color_map, branch_maps, sce_group_params, graph_params):
    """
    Graph result from a single scenario over time
    :param df_in: DataFrame of result
    :param color_map: dict of key --> hex color codes
    :param branch_maps: maps of LEAP branches --> subgroups
    :param sce_group_params: info from graph group in controller tab "scenario group params"
    :param graph_params: info from row of graphing tab of controller
    :return: NA - generates a graph
    """


    if graph_params['fuel_filter'] == '':
        fuel_filter = None
    else:
        fuel_filter = [fuel.strip() for fuel in graph_params['fuel_filter'].split(',')]

    df_graph = form_df_graph(
        df_in=df_in,
        sce_group_params=sce_group_params,
        result=[result.strip() for result in graph_params['result'].split(',')],
        multiplier=graph_params['multiplier'],
        marginalize=graph_params['marginalize'],
        cumulative=graph_params['cumulative'],
        discount=graph_params['discount'],
        filter_yrs=False,
        branch_map=branch_maps[graph_params['branch_map_name']],
        fuel_filter=fuel_filter,
        groupby=list(
            {'Scenario', 'Year', graph_params['xcol'], graph_params['ycol'], graph_params['color_col']} - {'Value'}
        )
    )

    if not graph_params['grouped']:
        fig = px.bar(
            df_graph,
            x=graph_params['xcol'],
            y=graph_params['ycol'],
            color=graph_params['color_col'],
            color_discrete_map=color_map,
        )
    else:
        fig = px.bar(
            df_graph,
            x=graph_params['xcol'],
            y=graph_params['ycol'],
            color=graph_params['color_col'],
            barmode='group',
            color_discrete_map=color_map,
        )

    if graph_params['include_sum']:
        df_sum = pd.DataFrame(columns=[graph_params['xcol'], graph_params['ycol']])
        for time_pt in df_graph[graph_params['xcol']].unique():
            sum_in_t = df_graph[df_graph[graph_params['xcol']] == time_pt][graph_params['ycol']].sum()
            df_sum.loc[len(df_sum.index)] = [time_pt, sum_in_t]
        # add line to graph showing sum
        fig.add_trace(go.Scatter(
            mode='lines',
            x=df_sum[graph_params['xcol']],
            y=df_sum[graph_params['ycol']],
            name="Total",
            showlegend=True,
            line=dict(
                color='black',
                dash='solid',
            )
        ))

    fig = update_fig_styling(fig, graph_params)
    fig.write_image(FIGURES_PATH / f"{graph_params['fname']}_{graph_params['group_id']}{IMAGE_FORMAT}",
                    scale=IMAGE_SCALE)


def bars_over_scenarios(df_in, color_map, branch_maps, sce_group_params, graph_params):
    """
    Graph result from a single scenario over time
    :param df_in: DataFrame of result
    :param color_map: dict of key --> hex color codes
    :param branch_maps: maps of LEAP branches --> subgroups
    :param sce_group_params: info from graph group in controller tab "scenario group params"
    :param graph_params: info from row of graphing tab of controller
    :return: NA - generates a graph
    """

    if graph_params['fuel_filter'] == '':
        fuel_filter = None
    else:
        fuel_filter = [fuel.strip() for fuel in graph_params['fuel_filter'].split(',')]

    groupby = {'Scenario', graph_params['xcol'], graph_params['ycol'], graph_params['color_col']} - {'Value'}
    if graph_params['sort_by'] != '':
        sort_by = [sort_col.strip() for sort_col in graph_params['sort_by'].split(',')]
        groupby.update(set(sort_by))

    df_graph = form_df_graph(
        df_in=df_in,
        sce_group_params=sce_group_params,
        result=[result.strip() for result in graph_params['result'].split(',')],
        multiplier=graph_params['multiplier'],
        marginalize=graph_params['marginalize'],
        cumulative=graph_params['cumulative'],
        discount=graph_params['discount'],
        filter_yrs=True,
        branch_map=branch_maps[graph_params['branch_map_name']],
        fuel_filter=fuel_filter,
        groupby=list(groupby)
    )

    # sort dataframe
    category_orders = dict()
    if graph_params['sort_by'] != '':
        sort_by = [sort_col.strip() for sort_col in graph_params['sort_by'].split(',')]
        df_graph = df_graph.sort_values(by=sort_by, ignore_index=True, ascending=graph_params['sort_ascending'])
        if graph_params['xcol'] == 'Value':
            category_orders = {graph_params['ycol']: df_graph[graph_params['ycol']].tolist()}
        else:
            category_orders = {graph_params['xcol']: df_graph[graph_params['xcol']].tolist()}


    # set up text annotations
    text_auto = False
    if graph_params['annotate']:
        text_auto = graph_params['annotation_style']

    if not graph_params['grouped']:
        fig = px.bar(
            df_graph,
            x=graph_params['xcol'],
            y=graph_params['ycol'],
            text_auto=text_auto,
            color=graph_params['color_col'],
            color_discrete_map=color_map,
            category_orders=category_orders,
        )
    else:
        fig = px.bar(
            df_graph,
            x=graph_params['xcol'],
            y=graph_params['ycol'],
            text_auto=text_auto,
            color=graph_params['color_col'],
            barmode='group',
            color_discrete_map=color_map,
            category_orders=category_orders,
        )

    fig = update_fig_styling(fig, graph_params)

    if graph_params['mark_sum']:
        groupby = list({graph_params['xcol'], graph_params['ycol']} - {'Value'})
        df_graph = df_graph.groupby(by=groupby, as_index=False)['Value'].sum()
        fig.add_trace(go.Scatter(
            mode='markers',
            x=df_graph[graph_params['xcol']],
            y=df_graph[graph_params['ycol']],
            name='Total',
            showlegend=True,
            marker=dict(
                color='LightSkyBlue',
                size=10,
                line=dict(
                    color='MediumPurple',
                    width=2
                )
            ),
        ))

    if graph_params['annotate_sum']:
        groupby = list({graph_params['xcol'], graph_params['ycol']} - {'Value'})
        df_graph = df_graph.groupby(by=groupby, as_index=False)['Value'].sum()
        for _, row in df_graph.iterrows():
            fig.add_annotation(
                x=row[graph_params['xcol']],
                y=row[graph_params['ycol']],
                text=f"{row['Value']:{graph_params['annotation_style']}}",
                xshift=0,
                yshift=0,
                showarrow=False,
            )

    fig.write_image(FIGURES_PATH / f"{graph_params['fname']}_{graph_params['group_id']}{IMAGE_FORMAT}",
                    scale=IMAGE_SCALE)


def diff_xaxis_lines(df_in, color_map, branch_maps, sce_group_params, graph_params):
    """
    Graph result from a single scenario over time
    :param df_in: DataFrame of result
    :param color_map: dict of key --> hex color codes
    :param branch_maps: maps of LEAP branches --> subgroups
    :param sce_group_params: info from graph group in controller tab "scenario group params"
    :param graph_params: info from row of graphing tab of controller
    :return: NA - generates a graph
    """

    if graph_params['fuel_filter'] == '':
        fuel_filter = None
    else:
        fuel_filter = [fuel.strip() for fuel in graph_params['fuel_filter'].split(',')]

    groupby = {'Scenario', graph_params['xcol'], graph_params['ycol'], graph_params['color_col']} - {'Value'}
    if graph_params['sort_by'] != '':
        sort_by = [sort_col.strip() for sort_col in graph_params['sort_by'].split(',')]
        groupby.update(set(sort_by))

    df_graph = form_df_graph(
        df_in=df_in,
        sce_group_params=sce_group_params,
        result=[result.strip() for result in graph_params['result'].split(',')],
        multiplier=graph_params['multiplier'],
        marginalize=graph_params['marginalize'],
        cumulative=graph_params['cumulative'],
        discount=graph_params['discount'],
        filter_yrs=True,
        branch_map=branch_maps[graph_params['branch_map_name']],
        fuel_filter=fuel_filter,
        groupby=list(groupby)
    )

    if graph_params['sort_by'] != '':
        sort_by = [sort_col.strip() for sort_col in graph_params['sort_by'].split(',')]
        df_graph = df_graph.sort_values(by=sort_by, ascending=graph_params['sort_ascending'], ignore_index=True)

    fig = px.line(
        df_graph,
        x=graph_params['xcol'],
        y=graph_params['ycol'],
        color=graph_params['color_col'],
        color_discrete_map=color_map,
        markers=graph_params['include_markers'],
    )

    fig = update_fig_styling(fig, graph_params)
    fig.write_image(FIGURES_PATH / f"{graph_params['fname']}_{graph_params['group_id']}{IMAGE_FORMAT}",
                    scale=IMAGE_SCALE)


def diff_xaxis_bars(df_in, color_map, branch_maps, sce_group_params, graph_params):
    """
    Graph result from a single scenario over time
    :param df_in: DataFrame of result
    :param color_map: dict of key --> hex color codes
    :param branch_maps: maps of LEAP branches --> subgroups
    :param sce_group_params: info from graph group in controller tab "scenario group params"
    :param graph_params: info from row of graphing tab of controller
    :return: NA - generates a graph
    """

    if graph_params['fuel_filter'] == '':
        fuel_filter = None
    else:
        fuel_filter = [fuel.strip() for fuel in graph_params['fuel_filter'].split(',')]

    df_graph = form_df_graph(
        df_in=df_in,
        sce_group_params=sce_group_params,
        result=[result.strip() for result in graph_params['result'].split(',')],
        multiplier=graph_params['multiplier'],
        marginalize=graph_params['marginalize'],
        cumulative=graph_params['cumulative'],
        discount=graph_params['discount'],
        filter_yrs=True,
        branch_map=branch_maps[graph_params['branch_map_name']],
        fuel_filter=fuel_filter,
        groupby=list(
            {'Scenario', graph_params['xcol'], graph_params['ycol'], graph_params['color_col']} - {'Value'}
        )
    )

    # set up text annotations
    text_auto = False
    if graph_params['annotate']:
        text_auto = graph_params['annotation_style']

    if not graph_params['grouped']:
        fig = px.bar(
            df_graph,
            x=graph_params['xcol'],
            y=graph_params['ycol'],
            text_auto=text_auto,
            color=graph_params['color_col'],
            color_discrete_map=color_map,
        )
    else:
        fig = px.bar(
            df_graph,
            x=graph_params['xcol'],
            y=graph_params['ycol'],
            text_auto=text_auto,
            color=graph_params['color_col'],
            barmode='group',
            color_discrete_map=color_map,
        )

    fig = update_fig_styling(fig, graph_params)
    fig.write_image(FIGURES_PATH / f"{graph_params['fname']}_{graph_params['group_id']}{IMAGE_FORMAT}",
                    scale=IMAGE_SCALE)


def x_y_scatter(df_in, color_map, branch_maps, sce_group_params, graph_params):
    """
    Graph result from a single scenario over time
    :param df_in: DataFrame of result
    :param color_map: dict of key --> hex color codes
    :param branch_maps: maps of LEAP branches --> subgroups
    :param sce_group_params: info from graph group in controller tab "scenario group params"
    :param graph_params: info from row of graphing tab of controller
    :return: NA - generates a graph
    """

    if graph_params['fuel_filter'] == '':
        fuel_filter = None
    else:
        fuel_filter = [fuel.strip() for fuel in graph_params['fuel_filter'].split(',')]

    df_graph_x = form_df_graph(
        df_in=df_in,
        sce_group_params=sce_group_params,
        result=[result.strip() for result in graph_params['result_x'].split(',')],
        multiplier=graph_params['multiplier_x'],
        marginalize=graph_params['marginalize_x'],
        cumulative=graph_params['cumulative_x'],
        discount=graph_params['discount_x'],
        filter_yrs=True,
        branch_map=branch_maps[graph_params['branch_map_name']],
        fuel_filter=fuel_filter,
        groupby=['Scenario']
    )
    df_graph_x = df_graph_x.rename(columns={'Value': 'Value_x'})

    df_graph_y = form_df_graph(
        df_in=df_in,
        sce_group_params=sce_group_params,
        result=[result.strip() for result in graph_params['result_y'].split(',')],
        multiplier=graph_params['multiplier_y'],
        marginalize=graph_params['marginalize_y'],
        cumulative=graph_params['cumulative_y'],
        discount=graph_params['discount_y'],
        filter_yrs=True,
        branch_map=branch_maps[graph_params['branch_map_name']],
        fuel_filter=fuel_filter,
        groupby=['Scenario']
    )
    df_graph_y = df_graph_y.rename(columns={'Value': 'Value_y'})
    df_graph = df_graph_x.merge(df_graph_y, how='outer')

    fig = go.Figure()
    for sce in sce_group_params['scenarios']:
        df_sce = df_graph[df_graph['Scenario'] == sce].copy()
        fig.add_trace(go.Scatter(
            mode='markers',
            x=df_sce['Value_x'],
            y=df_sce['Value_y'],
            name=sce_group_params['name_map'][sce],
            showlegend=sce_group_params['include_in_legend_map'][sce],
            marker_symbol=sce_group_params['marker_map'][sce],
            marker_color=color_map[sce_group_params[graph_params['color_col']+'_map'][sce]],
        ))

    fig.update_traces(marker={'size': 10})

    fig = update_fig_styling(fig, graph_params)
    fig.write_image(FIGURES_PATH / f"{graph_params['fname']}_{graph_params['group_id']}{IMAGE_FORMAT}",
                    scale=IMAGE_SCALE)


def tornado(df_in, color_map, branch_maps, sce_group_params, graph_params):
    """
    Graph result from a single scenario over time
    :param df_in: DataFrame of result
    :param color_map: dict of key --> hex color codes
    :param branch_maps: maps of LEAP branches --> subgroups
    :param sce_group_params: info from graph group in controller tab "scenario group params"
    :param graph_params: info from row of graphing tab of controller
    :return: NA - generates a graph
    """

    if graph_params['fuel_filter'] == '':
        fuel_filter = None
    else:
        fuel_filter = [fuel.strip() for fuel in graph_params['fuel_filter'].split(',')]

    df = form_df_graph(
        df_in=df_in,
        sce_group_params=sce_group_params,
        result=[result.strip() for result in graph_params['result'].split(',')],
        multiplier=graph_params['multiplier'],
        marginalize=graph_params['marginalize'],
        cumulative=graph_params['cumulative'],
        discount=graph_params['discount'],
        filter_yrs=True,
        branch_map=branch_maps[graph_params['branch_map_name']],
        fuel_filter=fuel_filter,
        groupby=list(
            {'Scenario', 'tornado_group_name', 'tornado_order_id', graph_params['color_col']} - {'Value'}
        )
    )

    df_graph = pd.DataFrame(columns=['tornado_group_name', 'tornado_order_id', 'bar_min',
                                     'bar_height', 'bar_max', graph_params['color_col']])
    for key, dfg in df.groupby(by=['tornado_group_name']):
        df_graph.loc[len(df_graph.index)] = [
            key,                                            # tornado_group_name
            dfg['tornado_order_id'].unique()[0],            # tornado_order_id
            dfg['Value'].min(),                             # bar_min
            dfg['Value'].max() - dfg['Value'].min(),        # bar_height
            dfg['Value'].max(),                             # bar_max
            dfg[graph_params['color_col']].unique()[0]      # color_id
        ]

    category_orders = dict()
    if graph_params['sort_by'] != '':
        sort_by = [sort_col.strip() for sort_col in graph_params['sort_by'].split(',')]
        df_graph = df_graph.sort_values(by=sort_by, ignore_index=True, ascending=graph_params['sort_ascending'])
        if graph_params['xcol'] == 'bar_height':
            category_orders = {graph_params['ycol']: df_graph[graph_params['ycol']].tolist()}
        else:
            category_orders = {graph_params['xcol']: df_graph[graph_params['xcol']].tolist()}

    fig = px.bar(
        df_graph,
        x=graph_params['xcol'],
        y=graph_params['ycol'],
        base='bar_min',
        color=graph_params['color_col'],
        color_discrete_map=color_map,
        category_orders=category_orders,
    )

    # add text labels to both side of the tornado bar
    if graph_params['annotate_tornado']:
        if graph_params['xcol'] == 'tornado_group_name':
            x_text_pos = np.array(list(zip(df_graph['tornado_group_name'], df_graph['tornado_group_name']))).flatten()
            y_text_pos = np.array(list(zip(
                df_graph['bar_min'] + df_graph['bar_height'] * .2,
                df_graph['bar_max'] - df_graph['bar_height'] * .2
            ))).flatten()
            text = [f'{y:.1f}' for y in np.array(list(zip(df_graph['bar_min'], df_graph['bar_max']))).flatten()]
        else:
            y_text_pos = np.array(list(zip(df_graph['tornado_group_name'], df_graph['tornado_group_name']))).flatten()
            x_text_pos = np.array(list(zip(
                df_graph['bar_min'] + df_graph['bar_height'] * .2,
                df_graph['bar_max'] - df_graph['bar_height'] * .2
            ))).flatten()
            text = [f'{x:.1f}' for x in np.array(list(zip(df_graph['bar_min'], df_graph['bar_max']))).flatten()]

        fig.add_trace(go.Scatter(
            x=x_text_pos,
            y=y_text_pos,
            text=text,
            mode='text',
            showlegend=False,
        ))

    fig = update_fig_styling(fig, graph_params)
    fig.write_image(FIGURES_PATH / f"{graph_params['fname']}_{graph_params['group_id']}{IMAGE_FORMAT}",
                    scale=IMAGE_SCALE)


def macc(df_in, color_map, branch_maps, sce_group_params, graph_params):
    """
    Graph result from a single scenario over time
    :param df_in: DataFrame of result
    :param color_map: dict of key --> hex color codes
    :param branch_maps: maps of LEAP branches --> subgroups
    :param sce_group_params: info from graph group in controller tab "scenario group params"
    :param graph_params: info from row of graphing tab of controller
    :return: NA - generates a graph
    """

    if graph_params['fuel_filter'] == '':
        fuel_filter = None
    else:
        fuel_filter = [fuel.strip() for fuel in graph_params['fuel_filter'].split(',')]

    df_graph_x = form_df_graph(
        df_in=df_in,
        sce_group_params=sce_group_params,
        result=[result.strip() for result in graph_params['result_x'].split(',')],
        multiplier=graph_params['multiplier_x'],
        marginalize=graph_params['marginalize_x'],
        cumulative=graph_params['cumulative_x'],
        discount=graph_params['discount_x'],
        filter_yrs=True,
        branch_map=branch_maps[graph_params['branch_map_name']],
        fuel_filter=fuel_filter,
        groupby=['Scenario', graph_params['color_col']]
    )
    df_graph_x = df_graph_x.rename(columns={'Value': 'Value_x'})

    df_graph_y = form_df_graph(
        df_in=df_in,
        sce_group_params=sce_group_params,
        result=[result.strip() for result in graph_params['result_y'].split(',')],
        multiplier=graph_params['multiplier_y'],
        marginalize=graph_params['marginalize_y'],
        cumulative=graph_params['cumulative_y'],
        discount=graph_params['discount_y'],
        filter_yrs=True,
        branch_map=branch_maps[graph_params['branch_map_name']],
        fuel_filter=fuel_filter,
        groupby=['Scenario', graph_params['color_col']]
    )
    df_graph_y = df_graph_y.rename(columns={'Value': 'Value_y'})
    df_graph = df_graph_x.merge(df_graph_y, how='outer')

    df_graph = df_graph.sort_values(by='Value_y', axis=0, ignore_index=True)
    df_graph['end_range_x'] = df_graph['Value_x'].cumsum()

    df_graph['start_range_x'] = 0
    for i in range(1, len(df_graph)):
        df_graph.loc[i, 'start_range_x'] = df_graph.loc[i-1, 'end_range_x']

    df_graph['width'] = df_graph['end_range_x'] - df_graph['start_range_x']
    df_graph['mid_x'] = (df_graph['end_range_x'] + df_graph['start_range_x']) / 2.0

    fig = go.Figure()
    legend_entries = []
    for sce in sce_group_params['scenarios']:
        df_sce = df_graph[df_graph['Scenario'] == sce].copy()
        fig.add_trace(go.Bar(
            x=df_sce['mid_x'],
            width=df_sce['width'],
            y=df_sce['Value_y'],
            name=df_sce[graph_params['color_col']].unique()[0],
            showlegend=df_sce[graph_params['color_col']].unique()[0] not in legend_entries,
            marker=dict(
                color=color_map[df_sce[graph_params['color_col']].unique()[0]],
            ),
        ))
        legend_entries.append(df_sce[graph_params['color_col']].unique()[0])

    # text annotations
    for sce in sce_group_params['scenarios']:
        df_sce = df_graph[df_graph['Scenario'] == sce].copy()
        fig.add_annotation(
            x=df_sce['mid_x'].unique()[0],
            y=max(0, df_sce['Value_y'].unique()[0]) + 90,
            text=sce_group_params['name_map'][sce],
            textangle=90,
            showarrow=True,
            startarrowsize=0.3,
            yanchor='auto',
            yshift=0,
            font=dict(
                color='black',
                size=8.5,
            )
        )

    # add in tick marks
    fig.update_xaxes(
        showgrid=True,
        ticks="outside",
        tickson="boundaries",
        ticklen=10
    )
    fig.update_yaxes(
        showgrid=True,
        ticks="outside",
        tickson="boundaries",
        ticklen=10
    )

    fig.update_traces(textfont_size=10, textposition='inside')
    fig.update_layout(uniformtext_minsize=6, uniformtext_mode='hide')

    fig = update_fig_styling(fig, graph_params)
    fig.write_image(FIGURES_PATH / f"{graph_params['fname']}_{graph_params['group_id']}{IMAGE_FORMAT}",
                    scale=IMAGE_SCALE)


def update_fig_styling(fig, graph_params):

    # update title, xaxis title, and yaxis title
    fig = update_titles(fig, graph_params['title'], graph_params['xaxis_title'], graph_params['yaxis_title'])

    # update plot size
    fig = update_plot_size(fig, graph_params['plot_width'], graph_params['plot_height'])

    # update legend title and positioning
    fig.update_layout(legend_title='')
    if graph_params['legend_position'] == 'below':
        fig = place_legend_below(fig, graph_params['xaxis_title'])
    elif graph_params['legend_position'] == 'hide':
        fig.update_layout(showlegend=False)

    # axis settings
    if graph_params['yaxis_to_zero']:
        fig.update_yaxes(rangemode="tozero")

    if graph_params['xaxis_to_zero']:
        fig.update_xaxes(rangemode="tozero")

    if graph_params['xaxis_lim'] != '':
        fig.update_xaxes(range=[float(val) for val in graph_params['xaxis_lim'].split(',')])

    if graph_params['yaxis_lim'] != '':
        fig.update_yaxes(range=[float(val) for val in graph_params['yaxis_lim'].split(',')])

    # update text annotations
    if 'annotate' in graph_params:
        if graph_params['annotate']:
            fig.update_traces(
                textfont_size=10,
                textposition=graph_params['annotation_position'],
                textangle=graph_params['annotation_angle'],
            )
            fig.update_layout(
                uniformtext_minsize=8,
                uniformtext_mode='hide',
            )

    return fig


def form_df_graph_load(df_in, sce_group_params, graph_params, sum_across_branches=False):
    # filter for relevant scenarios
    df_graph = df_in[df_in['Scenario'].isin(sce_group_params['scenarios'])].copy()

    # filter for correct result
    df_graph = df_graph[df_graph['Result Variable'] == graph_params['result']].copy()

    # scale result
    df_graph['Value'] = df_graph['Value'] * graph_params['multiplier']

    # sum loads across branches
    if sum_across_branches:
        df_graph = sum_load_across_branches(df_graph)

    # add columns based on the relevant maps (name_map, color_map...)
    for k, v in sce_group_params.items():
        if k.endswith('_map'):
            df_graph[k[:-4]] = df_graph['Scenario'].map(v)

    # get rid of years not specified to be included
    for sce, yr in sce_group_params['load_shape_yr_map'].items():
        df_graph = df_graph.reset_index(drop=True)
        rows_to_drop = np.array(
            (df_graph['Scenario'] == sce) &
            (df_graph['Year'] != yr)
        )
        row_ids_to_drop = list(np.where(rows_to_drop)[0])
        df_graph = df_graph.drop(index=row_ids_to_drop)

    return df_graph


def load_shape_area(df_in, color_map, sce_group_params, graph_params):

    df_graph = form_df_graph_load(df_in, sce_group_params, graph_params, sum_across_branches=False)

    fig = px.area(
        df_graph,
        x=graph_params['xcol'],
        y=graph_params['ycol'],
        color=graph_params['color_col'],
        color_discrete_map=color_map,
        line_shape='spline',
    )
    fig = update_to_load_shape_layout(fig)
    fig = update_fig_styling(fig, graph_params)
    fig.write_image(FIGURES_PATH / f"{graph_params['fname']}_{graph_params['group_id']}{IMAGE_FORMAT}",
                    scale=IMAGE_SCALE)


def load_shape_disaggregated(df_in, color_map, sce_group_params, graph_params):

    df_graph = form_df_graph_load(df_in, sce_group_params, graph_params, sum_across_branches=False)

    fig = px.line(
        df_graph,
        x=graph_params['xcol'],
        y=graph_params['ycol'],
        color=graph_params['color_col'],
        color_discrete_map=color_map,
    )

    if graph_params['include_sum']:
        df_sum = pd.DataFrame(columns=[graph_params['xcol'], graph_params['ycol']])
        for time_pt in df_graph[graph_params['xcol']].unique():
            sum_in_t = df_graph[df_graph[graph_params['xcol']] == time_pt][graph_params['ycol']].sum()
            df_sum.loc[len(df_sum.index)] = [time_pt, sum_in_t]
        # add line to graph showing sum
        fig.add_trace(go.Scatter(
            mode='lines',
            x=df_sum[graph_params['xcol']],
            y=df_sum[graph_params['ycol']],
            name="Total",
            showlegend=True,
            line=dict(
                color='black',
                dash='solid',
            )
        ))

    fig = update_to_load_shape_layout(fig)
    fig = update_fig_styling(fig, graph_params)
    fig.write_image(FIGURES_PATH / f"{graph_params['fname']}_{graph_params['group_id']}{IMAGE_FORMAT}",
                    scale=IMAGE_SCALE)


def multiple_load_shapes(df_in, color_map, sce_group_params, graph_params):

    df_graph = form_df_graph_load(df_in, sce_group_params, graph_params, sum_across_branches=True)

    fig = px.line(
        df_graph,
        x=graph_params['xcol'],
        y=graph_params['ycol'],
        color=graph_params['color_col'],
        color_discrete_map=color_map,
    )
    fig = update_to_load_shape_layout(fig)
    fig = update_fig_styling(fig, graph_params)
    fig.write_image(FIGURES_PATH / f"{graph_params['fname']}_{graph_params['group_id']}{IMAGE_FORMAT}",
                    scale=IMAGE_SCALE)


def create_scenario_copies(df):
    """ Function to create copies of specified scenarios under a new name"""
    df_excel = pd.read_excel(CONTROLLER_PATH / 'controller.xlsm', sheet_name="scenario_copies")

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

    df = pd.read_excel(CONTROLLER_PATH / 'controller.xlsm', sheet_name="branch_maps")

    # check if there are any branches missing from the controller
    missing_branches = list(all_branches - set(df['Branch'].unique()))
    if len(missing_branches) > 0:
        print(f"Branches not included in controller: {missing_branches}")

    # form maps of branches
    branch_maps = dict()
    map_names = df.columns.tolist()
    map_names.remove('Branch')

    # iterate through columns in the controller
    for map_name in map_names:
        branch_maps[map_name] = dict()

        df_map = df[['Branch'] + [map_name]].copy()

        # map unique sector (or other variable) to relevant branches
        for key, dfg in df_map.groupby(map_name):
            if key == False:
                continue
            branch_maps[map_name][key] = dfg['Branch'].tolist()

    return branch_maps


def form_sce_group_params():
    df = pd.read_excel(CONTROLLER_PATH / 'controller.xlsm', sheet_name="scenario_group_params")

    relevant_scenarios = set(df['scenario'].unique())
    relevant_scenarios.update(set(df['relative_to'].unique()))

    map_val_cols = [col for col in df.columns.tolist() if col not in ['group_id', 'scenario']]
    map_val_cols_by_name = [col for col in df.columns.tolist() if col not in ['group_id', 'scenario', 'name']]

    sce_group_params = dict()
    for group_id, dfg in df.groupby(by=['group_id']):
        sce_group_params[group_id] = dict()

        sce_group_params[group_id]['scenarios'] = dfg['scenario'].tolist()
        sce_group_params[group_id]['relevant_scenarios'] = list(set(dfg['scenario'].tolist() +
                                                                    dfg['relative_to'].tolist()))

        for col in map_val_cols:
            sce_group_params[group_id][col + '_map'] = dict(zip(dfg['scenario'], dfg[col]))

        for col in map_val_cols_by_name:
            sce_group_params[group_id][col + '_map_by_name'] = dict(zip(dfg['name'], dfg[col]))

    return relevant_scenarios, sce_group_params


def load_map(sheet_name):
    """ Function to load color map from controller """
    df = pd.read_excel(CONTROLLER_PATH / 'controller.xlsm', sheet_name=sheet_name)

    return dict(zip(df['key'], df['value']))


def calculate_annual_result_by_subgroup(df_in, result, subgroup_dict):
    """
    Function to sum the result variable in each year for the branches in each key/value pairing of subgroup dict
    :param df_in: dataframe containing all relevant results
    :param result: either a string or list of the relevant results (eg: Output by Output Fuel)
    :param subgroup_dict: dictionary mapping groups to their relevant branches (Eg: 'buildings' --> [Demand\Residential...]
    :return: dataframe with cols Year, Scenario, Fuel, Subgroup, Value. Subgroups are the keys of the subgroup_dict
    """

    df_out = pd.DataFrame(columns=['Year', 'Scenario', 'Fuel', 'Subgroup', 'Value'])

    # convert result_str to list so that multiple result_strings can be passed into the function as a list
    # note this is useful for energy demand and inputs
    if type(result) == str:
        result = [result]

    for key, dfg in df_in[df_in['Result Variable'].isin(result)].groupby(by=['Year', 'Scenario', 'Fuel']):
        yr, sce, fuel = key
        mask = np.array(
            (dfg['Year'] == yr) &
            (dfg['Scenario'] == sce) &
            (dfg['Fuel'] == fuel)
        )
        row_ids = list(np.where(mask)[0])
        for subgroup, branches in subgroup_dict.items():
            # eliminate branches that do not appear in dfg
            branches = list(set(branches).intersection(set(dfg.columns)))
            value = dfg[branches].iloc[row_ids].sum(axis=1).sum()
            df_out.loc[len(df_out.index)] = [yr, sce, fuel, subgroup, value]

    return df_out


def marginalize_it(df_in, relative_to_dict):
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
    """
    Function to discount all costs
    :param df_in: dataframe containing results with cols Year, Scenario, Fuel, Subgroup, Value
    :return: dataframe with discounted costs
    """
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
    """
    Function to calculate cumulative sum of 'Value' column across years and separated by Scenario, subgroup, and fuel
    :param df_in: dataframe containing results with cols Year, Scenario, Fuel, Subgroup, Value
    :return: dataframe with 'Value' column now containing the cumulative sum beginning from the base year
    """
    df = df_in.copy()
    df = df.sort_values(by='Year', axis=0)
    for key, dfg in df.groupby(by=['Scenario', 'Subgroup', 'Fuel']):
        sce, subg, fuel = key
        mask = np.array(
            (df['Scenario'] == sce) &
            (df['Subgroup'] == subg) &
            (df['Fuel'] == fuel)
        )
        ids = list(np.where(mask)[0])
        df.iloc[ids, df.columns.get_loc('Value')] = dfg['Value'].cumsum(axis=0)

    return df


def evaluate_cumulative_marginal_emissions_cumulative_marginal_cost(df_in, subgroup_dict, relative_to_map):
    """
    Function to evaluate cumulative marginal emissions and cumulative marginal costs
    :param df_in: raw results from LEAP script
    :param subgroup_dict: dict mapping subgroup --> list of relevant branches
    :param relative_to_map: dict mapping scenario --> scenario to marginalize against
    :return: dataframe containing cols 'cumulative_marginal_cost' and 'cumulative_marginal_emissions'
    """
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
    """
    Function to evalute the cost of abatement
    :param df_in: dataframe containing results
    :param subgroup_dict: dict mapping subgroup --> list of branches
    :param relative_to_map: dict mapping scenario --> scenario it should be marginalized against
    :return: df containing col 'cost_of_abatement' for year == End year
    """
    df = evaluate_cumulative_marginal_emissions_cumulative_marginal_cost(df_in, subgroup_dict, relative_to_map)
    df = df[df['Year'] == END_YEAR].copy()
    df['annualized_cost'] = df['cumulative_marginal_cost'] * CAPITAL_RECOVERY_FACTOR
    df['annualized_emissions_reduction'] = -1 * df['cumulative_marginal_emissions'] / TOTAL_YEARS
    df['cost_of_abatement'] = df['annualized_cost'] / df['annualized_emissions_reduction']
    df['Value'] = df['cost_of_abatement']

    return df


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


def update_titles(fig, title, xaxis_title, yaxis_title):
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )
    return fig


def place_legend_below(fig, xaxis_title):
    if xaxis_title == '':
        y = -.08
    else:
        y = -0.2

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


def update_plot_size(fig, width=800, height=500):
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
