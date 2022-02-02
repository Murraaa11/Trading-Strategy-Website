# import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
from scipy import stats
import statsmodels.api as sm
import yfinance as yf
import dash
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
from datetime import date
import time
import dash_bootstrap_components as dbc
import resources.graph as my_graph
import resources.trading as td
from dash.dash_table import DataTable, FormatTemplate
from dash.dash_table.Format import Format, Scheme, Trim
from dash.exceptions import PreventUpdate

ticker_list = pd.read_csv("tickers.csv", usecols=["Symbol"])

strategy_options = [
                    {'label': 'Filter Rules', 'value': 'FR'},
                    {'label': 'Simple Moving Average', 'value': 'SMA'},
                    {'label': 'Exponential Moving Average', 'value': 'EMA'},
                    {'label': 'Channel Breakouts', 'value': 'CB'},
                ]

metrics_list = ['Ticker', 'Round-trip trades', 'Total return (%)', 'Buy & hold return (%)',
                'Winning trades (%)', 'Ratio of average winning amount to average losing amount (%)',
                'Maximum drawdown (%)']

def get_tickers():
    return ([{'label': c, 'value': c} for c in ticker_list['Symbol'].drop_duplicates()])


def get_dropdown(name, options, multi=False, disabled=False):
    content = [
        # html.Label(name + ':'),
        dbc.Label(name + ':'),
        dcc.Dropdown(id = name.lower().replace(' ', '_'), options = options, multi=multi, disabled=disabled),
    ]
    return(content)

def get_parameter(name, input_id, min = 1.0, max = 100.0, step = 1.0):
    content = [
        dbc.Label(name + ':'),
        dbc.Input(id = input_id, type="number", min = min, max = max, step = step, style={'width': '10em'}),
    ]
    return(content)

def get_daterange(name):
    content = [
        dbc.Label(name + ':'),
        dcc.DatePickerRange(id = name.lower().replace(' ', '_'),
                            start_date=date(2016, 9, 27),
                            end_date=date(2021, 9, 24),
        ),
    ]
    return(content)

# Dash app

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

app.layout = html.Div(
    [
        html.Center(html.H2('Quantitative Trading Platform')),
        html.Hr(),

        #input division
        html.Div([
            html.Div(get_dropdown('Tickers', get_tickers(), multi=True),
                     style={'margin-left': '2em', 'margin-right': '1em', 'width': '15em'}),
            html.Div(get_daterange('Time Period'), style={'margin-right': '10em', 'width': '20em'}),
            html.Div(id='strategy-list', children=get_dropdown('Strategy', strategy_options, disabled=True),
                     style={'margin-right': '1em', 'width': '15em'}),
            html.Div(id='window-size', children=get_parameter('Window Size', 'WS'),
                     style={'display': 'none'}),
            html.Div(id='short-window-size', children=get_parameter('Short Window Size', 'SWS'),
                     style={'display': 'none'}),
            html.Div(id='long-window-size', children=get_parameter('Long Window Size', 'LWS'),
                     style={'display': 'none'}),
            html.Div(id='k-value', children=get_parameter('k', 'K', min = 0, max = 5, step = 0.1),
                     style={'display': 'none'}),
            html.Div(id='L-value', children=get_parameter('L', 'L', max = 50),
                     style={'display': 'none'}),
            html.Div(id='threshold', children=get_parameter('Threshold', 'THRES', min = 0, max = 0.1, step = 0.001),
                     style={'display': 'none'}),
        ], style={'display': 'flex', 'flex-direction': 'row', 'padding': '1em'}),

        #plot division
        html.Div([
            dcc.Loading(
                html.Div(id='ticker-plot'),
                type="circle",
            ),
            dcc.Loading(
                html.Div(id='hist-result-plot'),
                type="circle",
            ),
        ], style={'display': 'flex', 'flex-direction': 'row', 'padding': '1em'}),


        dcc.Loading(
            html.Center(id='hist-result-table'),
            type="circle",
        ),
        html.Br(),
        dcc.Loading(
            children = [html.Div(id='boot-result-table', children=[
                html.Center(html.H4('Bootstrapping Data Performance')),
                html.Label("""* The bootstrapping method is applied to examine if there exists data snooping bias.
                           The platform uses the AR(1) model to fit the historical log returns and calculates the 
                           empirical residuals. After that it samples from the residuals to form new sequences 
                           of innovations and uses them to generate new time series of returns and prices. 
                           The whole bootstrap process was repeated 10 times for brevity.""",
                           style={'margin-left': '17em', 'margin-right': '17em', 'font-size': '13px'}),
                #html.Br(),
                html.Label("Click on the grids to see bootstrapping details!",
                           style={'margin-left': '17em', 'color': 'blue', 'font-size': '13px'}),
                html.Center(
                    DataTable(
                        id='tbl',
                        style_table={'width': '70em'},
                        #row_selectable='single',
                        sort_action='native',
                        #export_format="xlsx",
                    ),
                )
                ], style={'display': 'none'}),],
            type="circle",
        ),
        html.Br(),
        html.Center(id='boot-detail'),

        # store the intermediate data
        dcc.Store(id='prdata'),
        dcc.Store(id='bootdata'),
    ],
    style={
        'margin': '2em',
        # 'border-radius': '1em',
        # 'border-style': 'solid',
        'padding': '1em',
        # 'background': '#ededed',
    }
)


@app.callback(
    Output('prdata', 'data'),
    Input('tickers', 'value'),
    Input('time_period', 'start_date'),
    Input('time_period', 'end_date'), prevent_initial_call=True)
def download_data(ticker, start_d, end_d):
    if ticker and start_d and end_d:
        myTicker = ' '
        myTicker = myTicker.join(ticker)
        if len(ticker) == 1:
            hist = yf.download(myTicker, start=start_d, end=end_d)[['Adj Close']]
            hist = hist.rename(columns={'Adj Close': myTicker})
        else:
            hist = yf.download(myTicker, start=start_d, end=end_d)['Adj Close']
        if hist.empty:
            return None
        tic = hist.columns
        for i in range(len(tic)):
            hist['{}_r'.format(tic[i])] = np.log(hist[tic[i]]).diff()
        return hist.to_json(date_format='iso')
    else:
        return None


@app.callback(
    Output('ticker-plot', 'children'),
    Input('prdata', 'data'),
    State('tickers', 'value'), prevent_initial_call=True)
def update_graphs(data, ticker):
    if data:
        hist = pd.read_json(data)

        # %% Price Graph generation
        fig1 = go.Figure()
        for i in range(len(ticker)):
            price = hist[[ticker[i]]].dropna()
            fig1.add_trace(go.Scatter(y=price[ticker[i]].to_list(), x=price.index.to_list(), name=ticker[i]))
        fig1 = my_graph.add_layout(fig1, 'Historical Stock Closing Prices', 'Closing Date', 'Price')

        # %% Return Graph generation
        fig2 = go.Figure()
        for i in range(len(ticker)):
            r = hist[['{}_r'.format(ticker[i])]].dropna()
            fig2.add_trace(go.Scatter(y=r['{}_r'.format(ticker[i])].to_list(), x=r.index.to_list(), name=ticker[i]))
        fig2 = my_graph.add_layout(fig2, 'Historical Returns', 'Closing Date', 'Return')

        return html.Div([
            dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2),
        ])
    else:
        return None


div_ids = ['window-size', 'short-window-size', 'long-window-size', 'k-value', 'L-value', 'threshold']
input_ids = ['WS', 'SWS', 'LWS', 'K', 'L', 'THRES']
@app.callback(
    [Output(i, 'style') for i in div_ids] +
    [Output(i, 'value') for i in input_ids] +
    [Output('strategy', 'disabled'), Output('strategy', 'value')],
    [Input('prdata', 'data'), Input('strategy', 'value')] +
    [Input(i, 'value') for i in input_ids],)
def select_parameter(data, strategy, *args):
    show = {'margin-right': '1em'}
    not_show = {'display': 'none'}
    values = [None]*len(div_ids)
    if data == None:
        return not_show, not_show, not_show, not_show, not_show, not_show, *values, True, None
    if strategy == 'FR':
        return show, not_show, not_show, not_show, not_show, show, *args, False, strategy
    elif strategy == 'SMA':
        return not_show, show, show, not_show, not_show, show, *args, False, strategy
    elif strategy == 'EMA':
        return not_show, show, show, not_show, not_show, show, *args, False, strategy
    elif strategy == 'CB':
        return show, not_show, not_show, show, show, not_show, *args, False, strategy
    else:
        return not_show, not_show, not_show, not_show, not_show, not_show, *values, False, strategy


@app.callback(
    Output('hist-result-plot', 'children'),
    Output('hist-result-table', 'children'),
    [Input('prdata', 'data'), Input('strategy', 'value')] +
    [Input(i, 'value') for i in input_ids], )
def testing_result(data, strategy, WS, SWS, LWS, K, L, THRES):
    if data == None:
        return None, None
    hist = pd.read_json(data)
    if strategy == 'FR':
        if WS and THRES != None:
            s, r = td.filter_rules(hist, WS, THRES)
            x = hist.index[WS: (len(hist) - 1)]
        else:
            return None, None
    elif strategy == 'SMA':
        if SWS and LWS and THRES != None:
            s, r = td.SMA(hist, SWS, LWS, THRES)
            x = hist.index[(LWS - 1): (len(hist) - 1)]
        else:
            return None, None
    elif strategy == 'EMA':
        if SWS and LWS and THRES != None:
            s, r = td.EMA(hist, SWS, LWS, THRES)
            x = hist.index[(LWS - 1): (len(hist) - 1)]
        else:
            return None, None
    elif strategy == 'CB':
        if WS and K != None and L:
            s, r = td.channel_breakouts(hist, WS, K, L)
            x = hist.index[max(WS, L): (len(hist) - 1)]
        else:
            return None, None
    else:
        return None, None

    # $$ plot generation
    ticker = s.columns

    fig1 = go.Figure()
    for i in range(len(ticker)):
        fig1.add_trace(go.Scatter(y=s[ticker[i]].to_list(), x=x.to_list(), name=ticker[i], mode='markers'))
    fig1 = my_graph.add_layout(fig1, 'Daily Holding Position', 'Closing Date', 'Position')
    fig1.update_traces(marker={'symbol': 'circle-dot'})

    # %% Return Graph generation
    fig2 = go.Figure()
    for i in range(len(ticker)):
        rr = r[ticker[i]].to_list()
        v = np.cumprod(rr + np.ones(len(rr))) * 100
        fig2.add_trace(go.Scatter(y=v, x=x.to_list(), name=ticker[i]))
    fig2 = my_graph.add_layout(fig2, 'Daily Holding Value (start with $100)', 'Closing Date', 'Holding Value')


    # generate result table
    history_result = pd.DataFrame(columns=['id'] + metrics_list[1:], index=ticker)
    for n in range(len(ticker)):
        history_result[:].loc[ticker[n]] = [ticker[n]] + td.metrics(s[ticker[n]], r[ticker[n]])

    return html.Div([dcc.Graph(figure=fig1), dcc.Graph(figure=fig2)]), html.Div([
        html.Center(html.H4('Historical Data Performance')),
        DataTable(
            id='historical', data=history_result.to_dict('records'),
            columns=[dict(id=history_result.columns[0], name=history_result.columns[0])] +
                    [dict(id=i, name=i, type='numeric', format=Format(precision=2, scheme=Scheme.fixed, trim=Trim.yes)) \
                     for i in history_result.columns[1:]],
            style_table={'width': '70em'},
            #row_selectable='single',
            sort_action='native',
            #export_format='xlsx',
        ),
    ])



@app.callback(
    Output('boot-result-table', 'style'),
    Output('tbl', 'data'),
    Output('tbl', 'columns'),
    Output('tbl', 'active_cell'),
    Output('bootdata', 'data'),
    [Input('prdata', 'data'), Input('strategy', 'value')] +
    [Input(i, 'value') for i in input_ids], )
def bootstrapping_result(data, strategy, WS, SWS, LWS, K, L, THRES):
    not_show = {'display':'none'}
    if data is None:
        return not_show, None, None, None, None
    hist = pd.read_json(data)
    num = int(len(hist.columns) / 2)
    ticker = hist.columns[:num]
    if strategy == 'FR':
        if WS and THRES != None:
            boottable = td.bootstrap(hist, strategy, WS, THRES)
        else:
            return not_show, None, None, None, None
    elif strategy == 'SMA':
        if SWS and LWS and THRES != None:
            boottable = td.bootstrap(hist, strategy, SWS, LWS, THRES)
        else:
            return not_show, None, None, None, None
    elif strategy == 'EMA':
        if SWS and LWS and THRES != None:
            boottable = td.bootstrap(hist, strategy, SWS, LWS, THRES)
        else:
            return not_show, None, None, None, None
    elif strategy == 'CB':
        if WS and K != None and L:
            boottable = td.bootstrap(hist, strategy, WS, K, L)
        else:
            return not_show, None, None, None, None
    else:
        return not_show, None, None, None, None

    boot_result = pd.DataFrame(columns=['id'] + metrics_list[1:], index=ticker)
    for n in range(num):
        boot_result[:].loc[ticker[n]] = [ticker[n]] + boottable[boottable['Ticker'] == ticker[n]].\
            drop(columns=['Ticker']).mean().tolist()

    cols = [dict(id=boot_result.columns[0], name=boot_result.columns[0])] + \
           [dict(id=i, name=i, type='numeric', format=Format(precision=2, scheme=Scheme.fixed, trim=Trim.yes)) \
            for i in boot_result.columns[1:]]
    return {}, boot_result.to_dict('records'), cols, None, boottable.to_json(date_format='iso')


@app.callback(
    Output('boot-detail', 'children'),
    Input('tbl', 'active_cell'),
    State('bootdata', 'data'),
)
def bootstrapping_detail(active_cell, data):
    if active_cell and data:
        boottable = pd.read_json(data)
        df = boottable[boottable['Ticker'] == active_cell['row_id']]
        return html.Div([
        html.Center(html.H4('Bootstrapping Details')),
        DataTable(
            id='detail', data=df.to_dict('records'),
            columns=[dict(id=df.columns[0], name=df.columns[0])] +
                    [dict(id=i, name=i, type='numeric', format=Format(precision=2, scheme=Scheme.fixed, trim=Trim.yes)) \
                     for i in df.columns[1:]],
            style_table={'width': '70em'},
            #row_selectable='single',
            sort_action='native',
            #export_format="xlsx",
        ),
    ])
    else:
        return None



# This is always the last line where the app is actually instanciated
if __name__ == '__main__':
    app.run_server(debug=False)
