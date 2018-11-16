import dash
import dash_table
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import colorlover as cl

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

################################# Pandas Stuff #################################
df = pd.read_csv('../data/telco_churn_clean.csv')
df_churn = df[df['Churn'] == 'Yes']
df_no_churn = df[df['Churn'] == 'No']

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
        },
        config={'displayModeBar': False}),
    # Dropdown and Pie Chart
    html.Div([
        dcc.Dropdown(
            id='feature-dropdown',
            options=[{'label': i, 'value': i} for i in categorical_cols],
            value='TechSupport'
        ),
        dcc.Graph(id='feature-pie', config={'displayModeBar': False})
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
    churn_series = df_churn[col].value_counts().sort_index()
    no_churn_series = df_no_churn[col].value_counts().sort_index()

    data = []
    data.append(go.Bar(
        x=churn_series.index,
        y=churn_series.values,
        name='Churn',
        marker={'color': 'red'}
    ))
    data.append(go.Bar(
        x=no_churn_series.index,
        y=no_churn_series.values,
        name='No Churn',
        marker={'color': 'blue'}
    ))

    layout = go.Layout(
        title=f'{col} Visualization'
    )
    return {'data': data, 'layout': layout}

@app.callback(Output('feature-pie', 'figure'),
              [Input('feature-dropdown', 'value')])
def display_pie(col):
    n_churn = len(df_churn)
    n_no_churn = len(df_no_churn)
    churn_series = df_churn[col].value_counts().sort_index() / n_churn
    no_churn_series = df_no_churn[col].value_counts().sort_index() / n_no_churn
    n_features = len(churn_series.index)

    # The whole plus one minus one thing is to skip the ultra white first color
    colors = cl.scales[str(n_features+1)]['seq']['Greens'][1:]

    # data, churn pie
    data = []
    data.append(go.Pie(
        labels=churn_series.index,
        values=churn_series.values,
        hole=0.65,
        domain={'x': [0, 1], 'y': [0, 1]},
        marker={
            'line': {'color': 'white', 'width': 2},
            'colors': colors
        },
        sort=False
    ))
    # data, no churn pie
    data.append(go.Pie(
        labels=no_churn_series.index,
        values=no_churn_series.values,
        domain={'x': [0.2, 0.8], 'y': [0.2, 0.8]},
        marker={
            'line': {'color': 'white', 'width': 2},
            'colors': colors
        },
        sort=False
    ))

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
