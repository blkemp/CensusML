# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import re
import au_census_analysis_functions as cnss_func
import table_funcs as tbl_func

all_categories = tbl_func.return_categories()
available_tables = tbl_func.return_tables()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    
    html.H1(
        children='The Training Grounds',
        style={
            'padding-left': '2%',
            'padding-bottom': '0px'
            }
            ),
    
    html.H6(
        children='Choose features available from the census to build a prediction algorithm',
        style={
            'padding-left': '2%',
            'margin-bottom':'1em',}
            ),

    # First row of header section containing dropdown selectors to filter for the measure to predict
    html.Div(id='y-section',children=[
        html.P(children='Choose the feature you want to predict'),
        
        html.Div([
            dcc.Dropdown(
                id='y-category-dropdown',
                placeholder='Select category(ies) you are interested in predicting...',
                optionHeight=42,
                multi=True
            )
        ],
        style={'width': '32%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='y-table-dropdown',
                placeholder='Select the table to select the measure from...',
                optionHeight=42
            )
        ], style={'width': '32%', 'padding-left': '2%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='y-category-field-dropdown',
                placeholder='[Optional] Select categorical fields to limit scope where possible...',
                optionHeight=64,
                multi=True
            )
        ], 
        style={'width': '32%', 'float': 'right', 'padding-left': '2%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='y-measure-dropdown',
                optionHeight=64
            )
        ], style={'width': '66%', 'display': 'inline-block'})
    ], style={
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'
    }),

    # second row of header section containing dropdown selectors to filter for features to include as variables in the model
    html.Div(id='X-section', children=[
        html.P(children='Choose the variables you want to use as inputs for your prediction'),

        html.Div([
            dcc.Dropdown(
                id='x-category-dropdown',
                options=[{'label': i, 'value': i} for i in all_categories],
                placeholder='Select the categories you are interested in as predictors...',
                optionHeight=42,
                multi=True
            )
        ],
        style={'width': '32%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='x-table-dropdown',
                placeholder='Select the tables to draw data from...',
                optionHeight=42,
                multi=True
            )
        ], 
        style={'width': '32%', 'padding-left': '2%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='x-category-field-dropdown',
                placeholder='[Optional] Select categorical fields to limit scope where possible...',
                optionHeight=64,
                multi=True
            )
        ], 
        style={'width': '32%', 'padding-left': '2%', 'display': 'inline-block'}),
        
        
        html.Div([
            dcc.Dropdown(
                id='x-measure-dropdown',
                options=[{'label': i, 'value': i} for i in ['test 1', 'test 2']],
                placeholder='[Advanced] Select specific measures to include/exclude from the available tables...',
            ),
            dcc.RadioItems(
                id='x-measure-inclusion-radio',
                options=[{'label': i, 'value': i} for i in ['Exclude', 'Include']],
                value='Exclude',
                labelStyle={'display': 'inline-block'}
            )
        ], 
        style={'width': '66%', 'display': 'inline-block'}
        )

    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px',
        'padding-bottom': '2em'
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
    [dash.dependencies.Input('y-category-dropdown', 'value'),
    dash.dependencies.Input('y-category-field-dropdown', 'value')])
def update_y_table_dropdown(y_category, y_cat_field):
    if y_category is None and y_cat_field is None: return update_table_dropdown(None)
    if y_category is None: return update_table_dropdown(None, y_cat_field)
    if y_cat_field is None: return update_table_dropdown(y_category, cat_intersection = True)
    # if category selection has fields selected, use the catergory intersection filter
    if y_cat_field is None: return update_table_dropdown(y_category, y_cat_field, cat_intersection = True)

def update_table_dropdown(chosen_categories, cat_field = [], cat_intersection = False):
    # Return list of tables which include any of the selected measures in the dropdown
    if chosen_categories is None or len(chosen_categories) == 0:
        # Note: weird behaviour where the input intialises as None but reverts to an empty list if item/s selected then all selections are removed
        options_df = tbl_func.return_relevant_tables(categories_list = all_categories, category_field_list = cat_field, category_intersection = cat_intersection)
    else:    
        options_df = tbl_func.return_relevant_tables(categories_list = chosen_categories, category_field_list = cat_field, category_intersection = cat_intersection)

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

@app.callback(
    [dash.dependencies.Output('y-measure-dropdown', 'options'),
    dash.dependencies.Output('y-category-dropdown', 'options'),
    dash.dependencies.Output('y-measure-dropdown', 'placeholder'),
    dash.dependencies.Output('y-measure-dropdown', 'disabled')],
    [dash.dependencies.Input('y-category-dropdown', 'value'),
    dash.dependencies.Input('y-table-dropdown', 'value'),
    dash.dependencies.Input('y-category-field-dropdown', 'value')]
)
def update_y_measure_dropdown(y_category, y_table, y_cat_field):
    return update_measure_dropdown(y_category, [y_table], y_cat_field, True) # predicted feature you want to narrow down so list should only be intersection of specified categories


@app.callback(
    [dash.dependencies.Output('x-measure-dropdown', 'options'),
    dash.dependencies.Output('x-category-dropdown', 'options'),
    dash.dependencies.Output('x-measure-dropdown', 'placeholder'),
    dash.dependencies.Output('x-measure-dropdown', 'disabled')],
    [dash.dependencies.Input('x-category-dropdown', 'value'),
    dash.dependencies.Input('x-table-dropdown', 'value'),
    dash.dependencies.Input('x-category-field-dropdown', 'value')]
)
def update_x_measure_dropdown(x_category, x_table, x_cat_field):
    return update_measure_dropdown(x_category, x_table, x_cat_field, False) # want to have a wide search space for variable selection so should be a union of specified categories


def update_measure_dropdown(category_list, table_list, cat_field_list, category_intersection):
    # address annoying behaviour of passing empty values when selections haven't been made yet
    cat_empty = False
    table_list_empty = False
    disable_dropdown = False
    placeholder_text = 'Select the measure you want to predict...'
    #print('{}; {}; {}'.format(category_list, table_list, cat_field_list))

    if cat_field_list is None: cat_field_list = []
    
    if category_list is None or len(category_list) == 0: cat_empty = True

    if table_list is None or len(table_list) == 0: 
        table_list_empty = True
        categories_output = [{'label': i, 'value': i} for i in all_categories]
    elif table_list[0] is None:
        table_list_empty = True
        categories_output = [{'label': i, 'value': i} for i in all_categories]
    else:
        categories_output = [{'label': i, 'value': i} for i in tbl_func.return_relevant_categories(table_list)] 
    
    #print('{}; {}'.format(cat_empty, table_list_empty))

    if cat_empty & table_list_empty: 
        features_df = tbl_func.return_relevant_features(['Measure'], available_tables, cat_field_list, category_intersection)
    elif cat_empty & (not table_list_empty):
        features_df = tbl_func.return_relevant_features(all_categories, table_list, cat_field_list, category_intersection)
    elif (not cat_empty) & table_list_empty: 
        features_df = tbl_func.return_relevant_features(category_list, available_tables, cat_field_list, category_intersection)
    else:
        features_df = tbl_func.return_relevant_features(category_list, table_list, cat_field_list, category_intersection)

    features_df = features_df.drop_duplicates(subset='Measures')
    features_df = features_df.set_index('Measures')

    #print(len(features_df))

    if len(features_df['Measure Desc'])>1000:
        # disable measure selection to prevent ridiculously slow loading
        placeholder_text = 'Please filter further to select the measure you want to predict...'
        disable_dropdown = True

    measures_output = [{'label': val, 'value': index} for index, val in features_df['Measure Desc'].iteritems()]

    return measures_output, categories_output, placeholder_text, disable_dropdown


@app.callback(
    dash.dependencies.Output('y-category-field-dropdown', 'options'),
    [dash.dependencies.Input('y-category-dropdown', 'value'),
    dash.dependencies.Input('y-category-dropdown', 'options'),
    dash.dependencies.Input('y-table-dropdown', 'value')])
def update_y_cat_field_dropdown(y_cat_input, y_cat_options, y_tbl_input):
    if y_tbl_input is None: return update_cat_field_dropdown(y_cat_input, y_cat_options, y_tbl_input)
    # if not empty, then convert the table selection to list for use in the return_features_subsets function
    return update_cat_field_dropdown(y_cat_input, y_cat_options, [y_tbl_input])


@app.callback(
    dash.dependencies.Output('x-category-field-dropdown', 'options'),
    [dash.dependencies.Input('x-category-dropdown', 'value'),
    dash.dependencies.Input('x-category-dropdown', 'options'),
    dash.dependencies.Input('x-table-dropdown', 'value')])
def update_x_cat_field_dropdown(x_cat_input, x_cat_options, x_tbl_input):
    return update_cat_field_dropdown(x_cat_input, x_cat_options, x_tbl_input)


def update_cat_field_dropdown(cat_input, cat_options, tbl_input):
    if cat_input is None: 
        if cat_options is None:
            categories_list = all_categories
        else:
            # convert "options" output (list of dictionaries in format {'key':x, 'value':x}) to a simple list
            categories_list = [x['value'] for x in cat_options]
    else:
        categories_list = cat_input

    # if tbl_input is not selected [empty] then use the defaults embedded in the function
    if tbl_input is None:
        feature_subset_df = tbl_func.return_features_subsets(categories_list = categories_list, category_intersection = False)
    else:        
        feature_subset_df = tbl_func.return_features_subsets(categories_list = categories_list, tables_list = tbl_input, category_intersection = False)

    return [{'label': index, 'value': val} for index, val in feature_subset_df['Measure|Category'].iteritems()]

    
def main():
    app.run_server(debug=True, port=8000, host='127.0.0.1',)


if __name__ == '__main__':
    main()