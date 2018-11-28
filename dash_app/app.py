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
        height=800,
        polar=dict(
            domain=dict(
                x=[0, 1],
                y=[0, 1]
            ),
            radialaxis=dict(
                visible=True,
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
        id='categorical-bar',
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
            id='categorical-dropdown',
            options=[{'label': col_label_dict[i], 'value': i} for i in
                    categorical_cols],
            value='TechSupport',
        ), style={'width': '80%', 'margin': 'auto'}),
        dcc.Graph(id='categorical-pie', config={'displayModeBar': False})
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
            # height=700
        )
    ),
    # Getting rid of this for now, want to make the full thing a radar
    # Might bring it back later, dunno! Probably tossing it.
    # html.Div(
    #     dcc.Markdown('I might put something here', className='markdown-text'),
    #     className='markdown-div',
    #     style=dict(
    #         width='25%',
    #         float='left',
    #         height=540 # This adds to 600 with the padding in the className
    #     )
    # ),
    # Continuous Variables
    # Distribution
    html.Div([
        html.Div([
        dcc.RadioItems(
            id='continuous-var',
            options=[{'label': i, 'value': i} for i in
                    ['Monthly', 'Total', 'Tenure']],
            value='Monthly',
            labelStyle=dict(display='inline'),
            style=dict(
                width='50%',
                textAlign='center',
                display='inline-block'
            )
        ),
        dcc.RadioItems(
            id='continuous-chart',
            options=[{'label': i, 'value': i} for i in
                    ['KDE', 'Histogram']],
            value='KDE',
            labelStyle=dict(display='inline'),
            style=dict(
                width='25%',
                textAlign='center',
                display='inline-block'
            )
        ),
        dcc.Checklist(
            id='continuous-view',
            options=[{'label': '1/20th View', 'value': 1}],
            values=[],
            labelStyle=dict(display='inline'),
            style=dict(
                width='25%',
                textAlign='center',
                display='inline-block'
            )
        )], className='dashboard-div'),
        dcc.Graph(
            id='continuous-plot',
            style=dict(height=400),
            config={'displayModeBar': False}
        ),
    ], style=dict(
        width='50%',
        float='left',
        display='inline-block',
        height=500,
        backgroundColor='white'
    )),
    # Charges over Tenure
    html.Div([
        # Toggles and radios
        html.Div([
        dcc.RadioItems(
            id='charge-radio-feature',
            options=[{'label': i, 'value': i} for i in ['Monthly', 'Total']],
            value='Monthly',
            labelStyle=dict(display='inline'),
            style=dict(
                width='24%',
                textAlign='center',
                display='inline-block'
            )
        ),
        dcc.RadioItems(
            id='charge-radio-bars',
            options=[{'label': i, 'value': i} for i in range(3,8)],
            value=5,
            labelStyle=dict(display='inline'),
            style=dict(
                width='25%',
                textAlign='center',
                display='inline-block',
                borderLeft='thin rgb(42, 207, 255) solid'
            )
        ),
        dcc.Checklist(
            id='charge-check-difference',
            options=[{'label': 'Toggle Difference', 'value': 1}],
            values=[],
            labelStyle=dict(display='inline'),
            style=dict(
                width='25%',
                textAlign='center',
                display='inline-block',
                borderLeft='thin rgb(42, 207, 255) solid'
            )
        ),
        dcc.Checklist(
            id='charge-check-stdev',
            options=[{'label': 'Show 1 Std. Dev.', 'value': 1}],
            values=[],
            labelStyle=dict(display='inline'),
            style=dict(
                width='25%',
                textAlign='center',
                display='inline-block',
                borderLeft='thin rgb(42, 207, 255) solid'
            )
        )],
        className='dashboard-div',
        style=dict(borderLeft='rgb(42, 207, 255) solid')),
        dcc.Graph(
            id='charge-plot',
            config={'displayModeBar': False}
        )
    ], style=dict(
        width='50%',
        float='left',
        display='inline-block',
        height=500,
        backgroundColor='white'
    )),
], style=dict(width='80%', margin='auto'))

@app.callback(Output('categorical-bar', 'figure'),
              [Input('categorical-dropdown', 'value')])
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
            opacity=0.7,
            line=dict(color='white', width=1)
        )
    ))
    data.append(go.Bar(
        x=no_churn_series.index,
        y=no_churn_series.values,
        name='No Churn',
        marker=dict(
            color='blue',
            opacity=0.7,
            line=dict(color='white', width=1)
        )
    ))

    layout = go.Layout(
        title=f'{col_label_dict[col]} by the Numbers'
    )
    return {'data': data, 'layout': layout}

@app.callback(Output('categorical-pie', 'figure'),
              [Input('categorical-dropdown', 'value')])
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
        marker=dict(
            line={'color': 'white', 'width': 2},
            colors=blues,
        ),
        sort=False,
        showlegend=False
    ),
    # data, churn pie
    go.Pie(
        labels=churn_series.index,
        values=churn_series.values,
        domain={'x': [0.2, 0.8], 'y': [0.2, 0.8]},
        marker=dict(
            line={'color': 'white', 'width': 2},
            colors=reds,
        ),
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
    Output('continuous-plot', 'figure'),
    [Input('continuous-var', 'value'),
    Input('continuous-chart', 'value'),
    Input('continuous-view', 'values')])
def continuous_var_plotter(feature, chart, view):
    '''
    Returns a plotly figure of either a kde plot or bar plot of the selected
    continuous variable
    '''

    if feature.lower() == 'tenure':
        col = 'tenure'
        title = 'Tenure KDE Plot'
        xaxis = {'title': 'Months'}
    else:
        col = feature + 'Charges'
        title = feature + ' Charges KDE Plot'
        xaxis = {'title': 'Charges ($)'}

    x_churn = df_churn[col]
    x_no_churn = df_no_churn[col]

    # Set plot domains
    feature_max = df[col].max()
    if view:
        x_upper = feature_max * 1/20
    else:
        x_upper = feature_max

    # Make Trace for both churn and non churn
    data = []
    for i in range(2):
        x = [x_churn, x_no_churn][i]
        color = ['red', 'blue'][i]
        name = ['Churn', 'No Churn'][i]

        # KDE Trace
        if chart == 'KDE':
            X_plot = np.linspace(0, x_upper * 1.2, 1000)
            kde = KernelDensity(kernel='gaussian', bandwidth=5)
            kde = kde.fit(x[:, np.newaxis])
            log_dens = kde.score_samples(X_plot[:, np.newaxis])
            data.append(go.Scatter(
                x=X_plot,
                y=np.exp(log_dens),
                mode='lines',
                fill='tozeroy',
                name=name,
                line=dict(color=color, width=2)
            ))
        # Histogram Trace
        elif chart == 'Histogram':
            x = x.copy()
            x = x[x <= x_upper]
            data.append(go.Histogram(
                x=x,
                name=name,
                marker=dict(
                    color=color,
                    opacity=0.7,
                    line=dict(color='white', width=1)
                )
            ))

    # Layout
    layout = go.Layout(
        title=title,
        xaxis=xaxis,
        legend=dict(x=.8, y=1),
        barmode='overlay'
    )

    return {'data': data, 'layout': layout}

@app.callback(
    Output('charge-plot', 'figure'),
    [Input('charge-radio-feature', 'value'),
    Input('charge-radio-bars', 'value'),
    Input('charge-check-difference', 'values'),
    Input('charge-check-stdev', 'values')
    ])
def charge_over_tenure(feature, bars, difference, show_stdev):
    '''
    Returns either the Monthly Charges or Totals Charges over Tenure as a
    Plotly Bar figure
    '''
    bin_ranges = [range(0, 76, 25), range(0, 81, 20), range(0, 76, 15),
                 range(0, 73, 12), range(0, 78, 11)]
    bins = bin_ranges[bars-3]

    bin_size = [25, 20, 15, 12, 11][bars-3]

    x_labels = []
    for i in range(bars-1):
        x_labels.append(f'{i*bin_size}-{(i+1)*bin_size}')
    x_labels.append(f'> {(bars-1)*bin_size}')

    gb_churn = df_churn.groupby(pd.cut(df_churn['tenure'], bins))
    gb_no_churn = df_no_churn.groupby(pd.cut(df_no_churn['tenure'], bins))

    if show_stdev:
        show_stdev = True
    else:
        show_stdev = False

    # Make Trace for both churn and non churn
    data = []
    for i in range(2):
        df_agg = [df_churn, df_no_churn][i]
        color = ['red', 'blue'][i]
        name = ['Churn', 'No Churn'][i]

        gb = df_agg.groupby(pd.cut(df_agg['tenure'], bins))
        means = gb[feature+'Charges'].mean()
        st_devs = gb[feature+'Charges'].std()

        text = means.round(2).apply(lambda x: '$' + str(x))

        data.append(go.Bar(
            x=x_labels,
            y=means.values,
            text=text,
            hoverinfo='text',
            name=name,
            marker=dict(
                color=color,
                opacity=0.7,
                line=dict(color='white', width=1)
            ),
            error_y=dict(
                type='data',
                array=st_devs.values,
                visible=show_stdev
            )
        ))

    layout = go.Layout(
        title=f'Average {feature} Charges over Tenure',
        xaxis=dict(title='Months'),
        yaxis=dict(title='Charges ($)')
    )

    return {'data': data, 'layout': layout}

if __name__ == '__main__':
    app.run_server(debug=True)
