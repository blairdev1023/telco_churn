import dash
import dash_table
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import colorlover as cl
from sklearn.neighbors import KernelDensity

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

################################## Markdowns ###################################

with open('assets/markdown/overview.md', 'r') as f:
    overview = f.read()
with open('assets/markdown/eda.md', 'r') as f:
    eda = f.read()


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
            domain=dict(
                x=[0, 1],
                y=[0, 1]
            ),
            radialaxis=dict(
                visible=True,
                # showgrid=False,
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
    html.Div(
        className='section-header',
        children=[html.Div('Overview', className='section-header--title')]
    ),
    html.Div(
        dcc.Markdown(overview, className='markdown-text'),
        className='markdown-div',
    ),
    # EDA
    html.Div(
        className='section-header',
        children=[
            html.Div('Who Churns? Exploratory Data Analysis',
            className='section-header--title')
        ]
    ),
    html.Div(
        dcc.Markdown(eda, className='markdown-text'),
        className='markdown-div',
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
            options=[{'label': col_label_dict[i], 'value': i} for i in
                    categorical_cols],
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
            width='75%',
            float='left',
            display='inline-block',
            height=600
        )
    ),
    html.Div(
        dcc.Markdown('LOL', className='markdown-text'),
        className='markdown-div',
        style={'width': '25%', 'float': 'left', 'height': 600}
    ),
    # Continuous Variables
    # KDE
    html.Div([
        dcc.RadioItems(
            id='kde-dropdown',
            options=[{'label': i, 'value': i} for i in
                    ['Monthly', 'Total', 'Tenure']],
            value='Monthly',
            labelStyle=dict(display='inline'),
            style=dict(width='80%', margin='auto')
        ),
        dcc.Graph(
            id='kde-plot',
        )
    ], style=dict(
        width='50%',
        float='left',
        display='inline-block',
        height=500,
        backgroundColor='white'
    )),
    # Mean Binned
    html.Div([
        dcc.RadioItems(
            id='charge-dropdown-feature',
            options=[{'label': i, 'value': i} for i in ['Monthly', 'Total']],
            value='Monthly',
            labelStyle=dict(display='inline'),
            style=dict(
                width='50%',
                # margin='auto',
                textAlign='center',
                display='inline-block'
            )
        ),
        dcc.RadioItems(
            id='charge-dropdown-display',
            options=[{'label': i, 'value': i} for i in ['Raw', 'Difference']],
            value='Raw',
            labelStyle=dict(display='inline'),
            style=dict(
                width='50%',
                # margin='auto',
                textAlign='center',
                display='inline-block'
            )
        ),
        dcc.Graph(
            id='charge-plot',
        )
    ], style=dict(
        width='50%',
        float='left',
        display='inline-block',
        height=500,
        backgroundColor='white'
    )),
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
    data = [
    go.Pie(
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
        title=f'{col_label_dict[col]} by Percentage',
        legend=dict(
            y=0.5,
            traceorder='reversed',
            font=dict(
                size=16
            )
        )
    )
    return {'data': data, 'layout': layout}

@app.callback(
    Output('kde-plot', 'figure'),
    [Input('kde-dropdown', 'value')])
def kde_plotter(feature):
    '''
    Returns a plotly figure of a kde plot for the selected feature
    '''

    if feature.lower() == 'tenure':
        col = 'tenure'
    else:
        col = feature + 'Charges'
    x_churn = df_churn[col][:, np.newaxis]
    x_no_churn = df_no_churn[col][:, np.newaxis]

    # Set x limit of plot to 20% feature max
    x_lim = df[col].max()
    x_lim *= 1.2
    X_plot = np.linspace(0, x_lim, 1000)[:, np.newaxis]

    # Make Gaussian and Trace for both churn and non churn
    data = []
    for i in range(2):
        x = [x_churn, x_no_churn][i]
        color = ['red', 'blue'][i]
        name = ['Churn', 'No Churn'][i]

        kde = KernelDensity(kernel='gaussian', bandwidth=5).fit(x)
        log_dens = kde.score_samples(X_plot)
        data.append(go.Scatter(
            x=X_plot[:, 0],
            y=np.exp(log_dens),
            mode='lines',
            fill='tozeroy',
            name=name,
            line=dict(color=color, width=2)
        ))

    layout = go.Layout(
        title='This is a title'
    )

    return {'data': data, 'layout': layout}

if __name__ == '__main__':
    app.run_server(debug=True)
