# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import re
import au_census_analysis_functions as cnss_func
import table_funcs as tbl_func

available_indicators = tbl_func.return_measures()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    
    html.H1(
        children='Test Dashboard',
        style={
            'textAlign': 'center',
            'margin-bottom':'1em',}
            ),
    
    html.Div(id='y-section',children=[

        html.Div([
            dcc.Dropdown(
                id='y-category-dropdown',
                options=[{'label': i, 'value': i} for i in available_indicators],
                placeholder='Select the category you are interested in predicting...'
            )
        ],
        style={'width': '32%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='y-table-dropdown',
                placeholder='Select the table to select the measure from...'
            )
        ], style={'width': '32%', 'padding-left': '2%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='y-measure-dropdown',
                options=[{'label': i, 'value': i} for i in ['Placeholder1','Placeholder2','Need to build function for this...']],
                placeholder='Select the measure you want to predict...'
            )
        ], style={'width': '32%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'
    }),

    html.Div(id='X-section', children=[

        html.Div([
            dcc.Dropdown(
                id='x-category-dropdown',
                options=[{'label': i, 'value': i} for i in available_indicators],
                placeholder='Select the categories you are interested in as predictors...',
                multi=True
            )
        ],
        style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='x-table-dropdown',
                placeholder='Select the tables to draw data from...',
                multi=True
            )
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'
    }),

    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
        )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),

])


@app.callback(
    dash.dependencies.Output('x-table-dropdown', 'options'),
    [dash.dependencies.Input('x-category-dropdown', 'value')])
def update_x_table_dropdown(x_measures):
    return update_table_dropdown(x_measures)


@app.callback(
    dash.dependencies.Output('y-table-dropdown', 'options'),
    [dash.dependencies.Input('y-category-dropdown', 'value')])
def update_y_table_dropdown(y_measures):
    if y_measures == None or len(y_measures) == 0: return update_table_dropdown(None)
    return update_table_dropdown([y_measures])


def update_table_dropdown(measures):
    # Return list of tables which include any of the selected measures in the dropdown
    if measures == None or len(measures) == 0:
        # Note: weird behaviour where the input intialises as None but reverts to an empty list if item/s selected then all selections are removed
        options_df = tbl_func.return_relevant_tables(available_indicators)
    else:    
        options_df = tbl_func.return_relevant_tables(measures)

    # Remove duplicates in the list where there are tables with multiple files (e.g. G09A, G09B, etc.)
    temp_df = pd.DataFrame(options_df['Table name'].value_counts()).reset_index()
    temp_df.columns = ['Table name','Count']
    options_df = options_df.reset_index().merge(temp_df, on='Table name')    
    
    # Only change the reference table name where there is >1 instance of the table, 
    # e.g. if filtering has resulted in returning "G09B", but not any other "G09" table, do not trim it
    options_df.loc[options_df.Count > 1, 'DataPack file'] = options_df['DataPack file'].str[:3]
    options_df = options_df.drop('Count', axis=1)
    options_df = options_df.drop_duplicates(subset='Table name')

    options_df = options_df.set_index('DataPack file')
    options_df.sort_index(inplace=True)

    return [{'label': '{} - {}'.format(val, index), 'value': index} for index, val in options_df['Table name'].iteritems()]


def main():
    app.run_server(debug=True, port=8000, host='127.0.0.1',)


if __name__ == '__main__':
    main()