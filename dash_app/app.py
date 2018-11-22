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

df_numeric = pd.read_csv('../data/telco_churn_numeric.csv')

################################ Common Labels #################################
# columns with less than 5 unique values, all others are non categorical
categorical_cols = df.nunique()[df.nunique() < 5].index.tolist()
categorical_cols.remove('Churn')

bin_cols = df_numeric.nunique()[df_numeric.nunique() < 5].index.tolist()
bin_cols.remove('Churn')

col_label_dict = {'gender': 'Gender', 'SeniorCitizen': 'Senior Citizen',
    'Partner': 'Partner', 'Dependents': 'Dependents',
    'PhoneService': 'Phone Service', 'MultipleLines': 'Multiple Lines',
    'InternetService': 'Internet Service', 'OnlineSecurity': 'Online Security',
    'OnlineBackup': 'Online Backup', 'DeviceProtection': 'Device Protection',
    'TechSupport': 'Tech Support', 'StreamingTV': 'Streaming TV',
    'StreamingMovies': 'Streaming Movies', 'Contract': 'Contract',
    'PaperlessBilling': 'Paperless Billing', 'PaymentMethod': 'Payment Method'}

pie_cols = [{'label': col_label_dict[i], 'value': i} for i in categorical_cols]

################################## Markdowns ###################################

with open('assets/markdown/overview.md', 'r') as f:
    overview = f.read()

################################# Static Plots #################################
def churn_polar():
    '''
    Returns the plotly figure for the churn polar chart. Since this chart is
    static I don't need to wrap it with a callback
    '''

    churn_idxs = df_churn.index
    no_churn_idxs = df_no_churn.index
    churn_means = df_numeric[bin_cols].loc[churn_idxs].mean()
    no_churn_means = df_numeric[bin_cols].loc[no_churn_idxs].mean()
    churn_means = churn_means.round(3)
    no_churn_means = no_churn_means.round(3)

    data = [
        go.Scatterpolar(
            r=churn_means,
            theta=churn_means.index,
            marker=dict(size=5, color='red'),
            mode='markers+lines',
            fill='toself'
        ),
        go.Scatterpolar(
            r=no_churn_means,
            theta=no_churn_means.index,
            marker=dict(size=5, color='blue'),
            mode='markers+lines',
            fill='toself'
        )
    ]

    layout = go.Layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                showgrid=False,
            )
        ),
        showlegend=False
    )

    return {'data': data, 'layout': layout}

################################### App Stuff ##################################
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# In the main website this will need to be imported
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div(
        className='app-header',
        children=[html.Div('Telco Case Study', className='app-header--title')]
    ),
    # Overview
    dcc.Markdown(overview, className='markdown'),
    html.Div(
        className='section-header',
        children=[html.Div('Overview', className='section-header--title')]
    ),
    html.Div(html.P(className='text-block', children=[]
    , style={'padding-top': 50, 'padding-bottom': 50}
        ), style={
            'backgroundColor': 'white'
        }
    ),
    # EDA
    html.Div(
        className='section-header',
        children=[
            html.Div('Exploratory Data Analysis',
            className='section-header--title')
        ]
    ),
    # Bar Plot
    dcc.Graph(
        id='feature-bar',
        style=dict(
            width='50%',
            float='left',
            display='inline-block',
            height=500
        ),
        config={'displayModeBar': False}),
    # Dropdown and Pie Chart
    html.Div([
        html.Div(dcc.Dropdown(
            id='feature-dropdown',
            options=pie_cols,
            value='TechSupport',
        ), style={'width': '80%', 'margin': 'auto'}),
        dcc.Graph(id='feature-pie', config={'displayModeBar': False})
    ], style=dict(
        width='50%',
        float='left',
        display='inline-block',
        height=500,
        backgroundColor='white'
    )),
    # Polar
    dcc.Graph(
        id='churn-polar',
        figure=churn_polar(),
        config={'displayModeBar': False},
        style=dict(
            width='100%',
            float='left',
            display='inline-block',
            height=600
        )
    )
], style=dict(width='80%', margin='auto'))

@app.callback(Output('feature-bar', 'figure'),
              [Input('feature-dropdown', 'value')])
def display_bar(col):
    '''
    Makes the figure for the bar plot in the EDA
    '''
    churn_series = df_churn[col].value_counts().sort_index()
    no_churn_series = df_no_churn[col].value_counts().sort_index()

    data = []
    data.append(go.Bar(
        x=churn_series.index,
        y=churn_series.values,
        name='Churn',
        marker=dict(
            color='red',
            opacity=0.8,
            line=dict(color='white', width=1)
        )
    ))
    data.append(go.Bar(
        x=no_churn_series.index,
        y=no_churn_series.values,
        name='No Churn',
        marker=dict(
            color='blue',
            opacity=0.8,
            line=dict(color='white', width=1)
        )
    ))

    layout = go.Layout(
        title=f'{col_label_dict[col]} by the Numbers'
    )
    return {'data': data, 'layout': layout}

@app.callback(Output('feature-pie', 'figure'),
              [Input('feature-dropdown', 'value')])
def display_pie(col):
    '''
    Makes the figure for the pie plot in the EDA
    '''
    n_churn = len(df_churn)
    n_no_churn = len(df_no_churn)
    churn_series = df_churn[col].value_counts().sort_index() / n_churn
    no_churn_series = df_no_churn[col].value_counts().sort_index() / n_no_churn
    n_features = len(churn_series.index)

    # The whole plus one minus one thing is to skip the ultra-white first color
    # Also, there is no "2" version for the palette
    reds = cl.scales[str(n_features+2)]['seq']['Reds'][2:]
    blues = cl.scales[str(n_features+2)]['seq']['Blues'][2:]

    # data, no churn pie
    data = [go.Pie(
        labels=no_churn_series.index,
        values=no_churn_series.values,
        domain={'x': [0, 1], 'y': [0, 1]},
        hole=0.65,
        marker={
            'line': {'color': 'white', 'width': 2},
            'colors': blues
        },
        sort=False,
        showlegend=False
    ),
    # data, churn pie
    go.Pie(
        labels=churn_series.index,
        values=churn_series.values,
        domain={'x': [0.2, 0.8], 'y': [0.2, 0.8]},
        marker={
            'line': {'color': 'white', 'width': 2},
            'colors': reds
        },
        sort=False,
        showlegend=False
    )]


    layout = go.Layout(
        title=f'{col_label_dict[col]} by Percentage'
    )
    return {'data': data, 'layout': layout}

if __name__ == '__main__':
    app.run_server(debug=True)
