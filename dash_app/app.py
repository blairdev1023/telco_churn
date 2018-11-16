import dash
import dash_table
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd

import dash

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

################################# Pandas Stuff #################################
df = pd.read_csv('../data/telco_churn_clean.csv')

################################ Common Labels #################################
# columns with less than 5 unique values, all others are non categorical
categorical_cols = df.nunique()[df.nunique() < 5].index.tolist()

################################### App Stuff ##################################
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# In the main website this will need to be imported
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1('Telco Study', style={'textAlign': 'center'}),
    # EDA
    html.H2('EDA', style={'textAlign': 'center'}),
    # Bar Plot
    dcc.Graph(
        id='feature-bar',
        style={
            'width': '50%',
            'float': 'left',
            'display': 'inline-block',
            'height': 500
        }),
    # Dropdown and Pie Chart
    html.Div([
        dcc.Dropdown(
            id='feature-dropdown',
            options=[{'label': i, 'value': i} for i in categorical_cols],
            value='TechSupport'
        ),
        dcc.Graph(id='feature-pie')
    ], style={
        'width': '50%',
        'float': 'left',
        'display': 'inline-block',
        'height': 500
    }),
], style={'width': '80%', 'margin-left': 'auto', 'margin-right': 'auto'})

@app.callback(Output('feature-bar', 'figure'),
              [Input('feature-dropdown', 'value')])
def display_bar(col):
    trace_series = df[col].value_counts().sort_index()
    data = [go.Trace(
        x=trace_series.index,
        y=trace_series.values,
        type='bar'
    )]

    layout = go.Layout(
        title=f'{col} Visualization'
    )
    return {'data': data, 'layout': layout}

@app.callback(Output('feature-pie', 'figure'),
              [Input('feature-dropdown', 'value')])
def display_pie(col):
    trace_series = df[col].value_counts().sort_index() / len(df)
    data = [go.Pie(
        labels=trace_series.index,
        values=trace_series.values,
        hole=0.5
    )]

    layout = go.Layout(
        title=f'{col} Visualization'
    )
    return {'data': data, 'layout': layout}

# EDA
#
# Currently includes:
#     * bar plot and pie chart examiner for a single feature
#
# working on:
#     * radial plot for churn vs non_churn percentages on binary data
#     * corr matrix
#     * groups of tenure over a single feature

if __name__ == '__main__':
    app.run_server(debug=True)
