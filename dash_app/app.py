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

################################################################################
################################# Pandas Stuff #################################
################################################################################

df = pd.read_csv('../data/telco_churn_clean.csv')
df_churn = df[df['Churn'] == 'Yes']
df_retain = df[df['Churn'] == 'No']

df_numeric = pd.read_csv('../data/telco_churn_numeric.csv')

################################################################################
################################ Common Labels #################################
################################################################################

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

hover_col = 'TechSupport'

dt_cols = ['Lower Bound', 'Upper Bound', 'N Churn', 'N Retain', 'Churn %']

################################################################################
################################## Markdowns ###################################
################################################################################

with open('assets/markdown/overview.md', 'r') as f:
    overview = f.read()
with open('assets/markdown/descriptions.md', 'r') as f:
    descriptions = f.read()
with open('assets/markdown/eda.md', 'r') as f:
    eda = f.read()
with open('assets/markdown/categorical.md', 'r') as f:
    categorical = f.read()
with open('assets/markdown/continuous.md', 'r') as f:
    continuous = f.read()
with open('assets/markdown/correlation.md', 'r') as f:
    correlation = f.read()

################################################################################
################################# Static Plots #################################
################################################################################
# Note! These must be ABOVE the app, being regular functions and all
def categorical_polar():
    '''
    Returns the plotly figure for the categorical polar chart.
    '''
    # Seperate churn from retain and compute means for features
    churn_idxs = df_churn.index
    retain_idxs = df_retain.index
    churn_means = df_numeric[bin_cols].loc[churn_idxs].mean()
    retain_means = df_numeric[bin_cols].loc[retain_idxs].mean()

    # Trace stuff
    col_names = churn_means.index

    hovertext = []
    for i, col in enumerate(col_names):
        churn_mean = round(churn_means[i] * 100, 2)
        retain_mean = round(retain_means[i] * 100, 2)
        # Make text in hover box
        text = col + '<br>'
        text += 'Churn:  ' + str(churn_mean) + '%<br>'
        text += 'Retain: ' + str(retain_mean) + '%<br>'
        hovertext.append(text)

    data = [
        go.Scatterpolar(
            name='Churn',
            r=churn_means,
            theta=col_names,
            marker=dict(size=10, color='red'),
            mode='markers+lines',
            fill='toself',
            hoverinfo='text',
            hovertext=hovertext,
        ),
        go.Scatterpolar(
            name='Retain',
            r=retain_means,
            theta=col_names,
            marker=dict(size=10, color='blue'),
            mode='markers+lines',
            fill='toself',
            hoverinfo='text',
            hovertext=hovertext,
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
            )
        ),
        showlegend=False
    )

    return {'data': data, 'layout': layout}

def correlation_heatmap():
    '''
    Returns the plotly figure for the correlation heatmap
    '''
    correlation_matrix = df_numeric.corr()

    data = [go.Heatmap(
        z=np.array(correlation_matrix),
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='Jet',
        colorbar=dict(
            title='Pearson Correlation Coefficient',
            titleside='right'
        )
    )]

    layout = go.Layout(
        title='Correlation Matrix of Full Dataset',
        margin=dict(l=100, b=100),
        xaxis=dict(tickangle=30),
        yaxis=dict(tickangle=45)
    )

    return {'data': data, 'layout': layout}

################################################################################
################################### App Stuff ##################################
################################################################################

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
            html.Div('Exploratory Data Analysis',
            className='section-header--title')
        ]
    ),
    html.Div(
        dcc.Markdown(eda, className='markdown-text'),
        className='markdown-div',
    ),
    # Categorical Intro
    html.Div(
        dcc.Markdown(categorical, className='markdown-text-line'),
        className='markdown-div',
    ),

    ### Categoricals
    html.Div([
        # Polar
        dcc.Graph(
            id='categorical-polar',
            figure=categorical_polar(),
            hoverData=dict(points=[{'theta': 'TechSupport'}]),
            config={'displayModeBar': False},
            style=dict(
                width='60%',
                float='left',
                display='inline-block',
                height=700
            )
        ),
        html.Div([
        # Decsription
        html.Div(
            dcc.Markdown(
                id='categorical-description',
                className='markdown-text',
            ),
            className='markdown-div-padless',
            style=dict(
                width='100%',
                height=130,
                float='left',
                display='inline-block',
            ),
        ),
        # Pie
        dcc.Graph(
            id='categorical-pie',
            style=dict(
                width='100%',
                height=300,
                float='left',
                display='inline-block',
            ),
            config={'displayModeBar': False}
        ),
        # Bar
        dcc.Graph(
            id='categorical-bar',
            style=dict(
                width='100%',
                height=270,
                float='left',
                display='inline-block',
            ),
            config={'displayModeBar': False}
        )
        ], className='div',
        style=dict(
            width='39%',
            height=700,
            float='left',
            display='inline-block',
            border='thin black solid',
        )),
    ], style=dict(
        width='100%',
        display='inline-block',
        height=700,
        backgroundColor='white'
    )),

    ### Continuous variables
    # Intro
    html.Div(
        dcc.Markdown(continuous, className='markdown-text-line'),
        className='markdown-div',
        style=dict(
            width='100%',
            float='left',
            display='inline-block',
            paddingTop=100
        ),
    ),
    # KDE/Histogram
    html.Div([
        # Toggles and radios
        html.Div([
        dcc.RadioItems(
            id='continuous-var',
            options=[{'label': i, 'value': i} for i in
                    ['Monthly', 'Total', 'Tenure']],
            value='Monthly',
            labelStyle=dict(display='inline'),
            style=dict(
                width='100%',
                textAlign='center',
                display='inline-block'
            )
        )], className='dashboard-div'),
        # Plot
        dcc.Graph(
            id='continuous-plot',
            config={'displayModeBar': False}
        ),
        # Aggregator
        html.Div([
            # Label and Slider
            html.Div([
                html.P('Aggregator:',
                    style=dict(
                        fontWeight='bold',
                        width='20%',
                        marginLeft='10%',
                        display='inline-block',
                    )
                ),
                dcc.RangeSlider(
                    id='aggregator-rangeslider',
                    className='dcc-single',
                    allowCross=False
                ),
            ], style=dict(
                width='100%',
                float='left',
                display='inline-block',
                height=60,
                backgroundColor='white'
            )),
            # Info Table
            html.Div(dash_table.DataTable(
                id='aggregator-datatable',
                columns=[{'name': i, 'id': i} for i in dt_cols],
                # data=[{'Lower Bound': 0, 'Upper Bound': 100, 'N Churn': 50, 'N Retain': 150, 'Churn %': '33%'}]
            ), style=dict(
                width='80%',
                marginLeft='10%',
                marginRight='10%',
                height=90,
                display='inline-block')
            ),
        ], style=dict(
            width='100%',
            float='left',
            display='inline-block',
            height=150,
            backgroundColor='white'
        ))
    ], style=dict(
        width='50%',
        float='left',
        display='inline-block',
        height=650,
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
                width='29%',
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
                width='30%',
                textAlign='center',
                display='inline-block',
                borderLeft='thin rgb(42, 207, 255) solid'
            )
        ),
        dcc.Checklist(
            id='charge-check-difference',
            options=[{'label': 'Difference', 'value': 1}],
            values=[],
            labelStyle=dict(display='inline'),
            style=dict(
                width='20%',
                textAlign='center',
                display='inline-block',
                borderLeft='thin rgb(42, 207, 255) solid'
            )
        ),
        dcc.Checklist(
            id='charge-check-stdev',
            options=[{'label': '1 Std. Dev.', 'value': 1}],
            values=[],
            labelStyle=dict(display='inline'),
            style=dict(
                width='20%',
                textAlign='center',
                display='inline-block',
                borderLeft='thin rgb(42, 207, 255) solid'
            )
        )],
        className='dashboard-div',
        style=dict(borderLeft='thick rgb(42, 207, 255) solid')
        ),
        # Plot
        dcc.Graph(
            id='charge-plot',
            config={'displayModeBar': False}
        ),
    ], style=dict(
        width='50%',
        float='left',
        display='inline-block',
        height=650,
        backgroundColor='white'
    )),

    ### Correlation
    # Horizontal line (not the best way to do but it's late and this works)
    html.Div(
        dcc.Markdown('', className='markdown-text-line'),
        className='markdown-div',
        style=dict(
            width='100%',
            float='left',
            display='inline-block',
            paddingBottom=100
        ),
    ),
    html.Div([
    # Intro
    html.Div(
        dcc.Markdown(correlation, className='markdown-text'),
        className='markdown-div-padless',
        style=dict(
            width='33%',
            height=600,
            float='left',
            display='inline-block',
        )
    ),
    # Plot
    dcc.Graph(
        id='correlation-heatmap',
        figure=correlation_heatmap(),
        config={'displayModeBar': False},
        style=dict(
            width='66%',
            height=600,
            float='left',
            display='inline-block',
            config={'displayModeBar': False},
        )
    )], style=dict(
        width='100%',
        backgroundColor='white',
        display='inline-block'
    )),

    ### Insights
    html.Div(dcc.Tabs(id='eda-tabs', value='tab-1', children=[
        dcc.Tab(label='Tab One', value='tab-1'),
        dcc.Tab(label='Tab Two', value='tab-2'),
        dcc.Tab(label='Tab Three', value='tab-3'),
        dcc.Tab(label='Tab Four', value='tab-4'),
        dcc.Tab(label='Tab Five', value='tab-5'),
    ]), style=dict(
        width='100%',
        float='left',
        display='inline-block',
    )),
    html.Div(
        dcc.Markdown(id='eda-insights', className='markdown-text'),
        className='markdown-div',
        style=dict(
            width='100%',
            float='left',
            display='inline-block',
        )
    )
], style=dict(width='80%', margin='auto', backgroundColor='white'))

def check_polar_hoverData(hoverData):
    '''
    Parses the hoverData dictionary and returns the pandas column related to
    the user's hovered column
    '''
    points_dict = hoverData['points'][0]
    if 'theta' in points_dict:
        hover_col = hoverData['points'][0]['theta']
        is_cat = hover_col in categorical_cols
        if hover_col.split('_')[0] == 'Payment':
            hover_col = 'PaymentMethod'
        elif hover_col.split('_')[0] == 'Contract':
            hover_col = 'Contract'
        return hover_col
    else:
        return hover_col

@app.callback(
    Output('categorical-description', 'children'),
    [Input('categorical-polar', 'hoverData')])
def feature_description(hoverData):
    '''
    Returns the feature description of whatever column is hovered on in the
    polar plot
    '''
    col_idx = {col: i for i, col in enumerate(categorical_cols)}
    hover_col = check_polar_hoverData(hoverData)
    return descriptions.split('.\n')[col_idx[hover_col]]

# Bars
@app.callback(
    Output('categorical-bar', 'figure'),
    [Input('categorical-polar', 'hoverData')])
def display_bar(hoverData):
    '''
    Makes the figure for the bar plot in the EDA
    '''
    col = check_polar_hoverData(hoverData)
    churn_series = df_churn[col].value_counts().sort_index()
    retain_series = df_retain[col].value_counts().sort_index()

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
        x=retain_series.index,
        y=retain_series.values,
        name='Retain',
        marker=dict(
            color='blue',
            opacity=0.7,
            line=dict(color='white', width=1)
        )
    ))

    layout = go.Layout(margin=dict(t=5))
    return {'data': data, 'layout': layout}

# Pie
@app.callback(
    Output('categorical-pie', 'figure'),
    [Input('categorical-polar', 'hoverData')])
def display_pie(hoverData):
    '''
    Makes the figure for the pie plot in the EDA
    '''
    col = check_polar_hoverData(hoverData)

    n_churn = len(df_churn)
    n_retain = len(df_retain)
    churn_series = df_churn[col].value_counts().sort_index() / n_churn
    retain_series = df_retain[col].value_counts().sort_index() / n_retain
    n_features = len(churn_series.index)

    # The whole plus one minus one thing is to skip the ultra-white first color
    # Also, there is no "2" version for the palette
    reds = cl.scales[str(n_features+2)]['seq']['Reds'][2:]
    blues = cl.scales[str(n_features+2)]['seq']['Blues'][2:]

    # data, retain pie
    data = [
    go.Pie(
        labels=retain_series.index,
        values=retain_series.values,
        domain={'x': [0, 1], 'y': [0, 1]},
        hole=0.65,
        opacity=0.9,
        textposition='inside',
        insidetextfont = dict(color='black'),
        hoverinfo='label+percent',
        name='Retain',
        marker=dict(
            line={'color': 'white', 'width': 2},
            colors=blues,
        ),
        sort=False,
        legendgroup='group'
    ),
    # data, churn pie
    go.Pie(
        labels=churn_series.index,
        values=churn_series.values,
        domain={'x': [0.2, 0.8], 'y': [0.2, 0.8]},
        opacity=0.9,
        textposition='inside',
        insidetextfont = dict(color='black'),
        hoverinfo='label+percent',
        name='Churn',
        marker=dict(
            line={'color': 'white', 'width': 2},
            colors=reds,
        ),
        sort=False,
        showlegend=False
    )]

    layout = go.Layout(
        margin=dict(t=30, b=0, r=0, l=0),
        legend=dict(
            x=0, y=1.1,
            bgcolor='rgba(0,0,0,0)',
            orientation='h',
            traceorder='reversed',
        ),
    )
    return {'data': data, 'layout': layout}

# Univariate Plot
@app.callback(
    Output('continuous-plot', 'figure'),
    [Input('continuous-var', 'value'),
    Input('aggregator-rangeslider', 'value')])
def continuous_var_plotter(feature, agg_range):
    '''
    Returns a plotly figure of either a kde plot or bar plot of the selected
    continuous variable
    '''
    # Set plot details
    title = feature + ' Histogram (Stacked)'
    if feature.lower() == 'tenure':
        col = 'tenure'
        title = 'Tenure'
        xaxis = {'title': 'Months'}
    else:
        col = feature + 'Charges'
        feature += ' Charges'
        xaxis = {'title': 'Charges ($)'}

    # Get feature series
    x_churn = df_churn[col]
    x_retain = df_retain[col]

    # Make Trace for both churn and retain
    data = []
    for i in range(2):
        x = [x_churn, x_retain][i].copy()
        color = ['red', 'blue'][i]
        name = ['Churn', 'Retain'][i]

        # Set y-limit
        if feature == 'Tenure':
            y_upper = 700
        else:
            y_upper = 850

        # Histogram Trace
        data.append(go.Histogram(
            x=x,
            name=name,
            marker=dict(
                color=color,
                opacity=0.7,
                line=dict(color='white', width=1)
            )
        ))
    # Aggregator Trace
    for agg_x in agg_range:
        data.append(go.Scatter(
            x=(agg_x, agg_x),
            y=(0, y_upper),
            hoverinfo='none',
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            marker=dict(opacity=0),
            showlegend=False,
        ))

    # Layout
    layout = go.Layout(
        title=title,
        xaxis=xaxis,
        yaxis=dict(range=[0,y_upper]),
        legend=dict(x=.7, y=1, bgcolor='rgba(0,0,0,0)'),
        barmode='stack'
    )

    return {'data': data, 'layout': layout}

# Aggregator Table
@app.callback(
    Output('aggregator-datatable', 'data'),
    [Input('continuous-var', 'value'),
    Input('aggregator-rangeslider', 'value')])
def aggregator_table(feature, agg_range):
    '''
    Returns the dictionary of aggregated data for the DataTable
    '''
    if feature.lower() == 'tenure':
        col = 'tenure'
    else:
        col = feature + 'Charges'

    agg_lower = agg_range[0]
    agg_upper = agg_range[1]

    # Get feature
    x_churn = df_churn[col]
    x_retain = df_retain[col]

    # Get aggregated domain
    x_churn = x_churn[x_churn >= agg_lower]
    x_churn = x_churn[x_churn <= agg_upper]
    x_retain = x_retain[x_retain >= agg_lower]
    x_retain = x_retain[x_retain <= agg_upper]

    n_churn = len(x_churn)
    n_retain = len(x_retain)
    percent = (n_churn / (n_churn + n_retain)) * 100

    data = {}
    data['Lower Bound'] = round(agg_lower, 2)
    data['Upper Bound'] = round(agg_upper, 2)
    data['N Churn'] = n_churn
    data['N Retain'] = n_retain
    data['Churn %'] = round(percent, 2)

    return [data]

# Agg Rangeslider Marks
@app.callback(
    Output('aggregator-rangeslider', 'marks'),
    [Input('continuous-var', 'value')])
def agg_rangeslider_marks(feature):
    if feature == 'Tenure':
        print(feature, 'Tenure')
        return {i: str(i) for i in range(0, 73, 12)}
    elif feature == 'Monthly':
        print(feature, 'Monthly')
        feature_max = int(df[feature+'Charges'].max())
        return {i: '$' + str(i) for i in range(20, feature_max, 20)}
    elif feature == 'Total':
        print(feature, 'Total')
        feature_max = int(df[feature+'Charges'].max())
        return {i: '$' + str(i) for i in range(0, feature_max, 1000)}

# Agg Rangeslider Min
@app.callback(
    Output('aggregator-rangeslider', 'min'),
    [Input('continuous-var', 'value')])
def agg_rangeslider_marks(feature):
    if feature == 'Tenure':
        return 0
    elif feature == 'Monthly':
        return 18
    elif feature == 'Total':
        return 0

# Agg Rangeslider Max
@app.callback(
    Output('aggregator-rangeslider', 'max'),
    [Input('continuous-var', 'value')])
def agg_rangeslider_marks(feature):
    if feature == 'Tenure':
        return 72
    elif feature == 'Monthly':
        return round(df[feature+'Charges'].max())
    elif feature == 'Total':
        return round(df[feature+'Charges'].max())

# Agg Rangeslider Step
@app.callback(
    Output('aggregator-rangeslider', 'step'),
    [Input('continuous-var', 'value')])
def agg_rangeslider_marks(feature):
    if feature == 'Tenure':
        return 1
    elif feature == 'Monthly':
        return 1
    elif feature == 'Total':
        return 100

# Agg Rangeslider Value
@app.callback(
    Output('aggregator-rangeslider', 'value'),
    [Input('continuous-var', 'value')])
def agg_rangeslider_marks(feature):
    if feature == 'Tenure':
        return [0, 72]
    elif feature == 'Monthly':
        return [18, round(df[feature+'Charges'].max())]
    elif feature == 'Total':
        return [0, round(df[feature+'Charges'].max())]

### Charge over Tenure
@app.callback(
    Output('charge-plot', 'figure'),
    [Input('charge-radio-feature', 'value'),
    Input('charge-radio-bars', 'value'),
    Input('charge-check-difference', 'values'),
    Input('charge-check-stdev', 'values')])
def charge_over_tenure(feature, bars, difference, show_stdev):
    '''
    Returns either the Monthly Charges or Totals Charges over Tenure as a
    Plotly Bar figure
    '''
    bin_ranges = [range(0, 76, 25), range(0, 81, 20), range(0, 76, 15),
                 range(0, 73, 12), range(0, 78, 11)]
    bins = bin_ranges[bars-3]

    bin_size = [25, 20, 15, 12, 11][bars-3]

    labels = [f'{i*bin_size}-{(i+1)*bin_size}' for i in range(bars-1)]
    labels.append(f'> {(bars-1)*bin_size}')

    if show_stdev:
        show_stdev = True
    else:
        show_stdev = False

    # Means
    means_churn = charge_binner(df_churn, feature, bins, 'mean')
    means_retain = charge_binner(df_retain, feature, bins, 'mean')
    # Standard Devs.
    stdevs_churn = charge_binner(df_churn, feature, bins, 'stdev')
    stdevs_retain = charge_binner(df_retain, feature, bins, 'stdev')

    if difference:
        means = means_churn - means_retain
        # Error Prop. -  squareroot of the sum of the squares
        sqr_churn = np.square(stdevs_churn) / bars
        sqr_retain = np.square(stdevs_retain) / bars
        stdevs = np.sqrt(sqr_churn + sqr_retain)

        trace = charge_bar_tracer(means, stdevs, labels,
                'Churn - Retain', 'khaki', show_stdev)
        data = [trace]
    else:
        trace_churn = trace = charge_bar_tracer(means_churn,
                stdevs_churn, labels, 'Churn', 'red', show_stdev)
        trace_retain = trace = charge_bar_tracer(means_retain,
                stdevs_retain, labels, 'Retain', 'blue', show_stdev)
        data = [trace_churn, trace_retain]

    if feature == 'Monthly':
        y_lim = 125
    elif feature == 'Total':
        y_lim = 9000

    layout = go.Layout(
        title=f'Average {feature} Charges over Tenure',
        xaxis=dict(title='Months'),
        yaxis=dict(title='Charges ($)', range=[0, y_lim]),
        showlegend=True,
        legend=dict(x=.8, y=1.1, bgcolor='rgba(0,0,0,0)')
    )

    return {'data': data, 'layout': layout}

def charge_binner(df, feature, bins, mode):
    '''
    Helper function for charge_over_feature and aggregator_table

    Aggregates the pandas "df" across the "feature" into n "bins" (using
    'tenure' as the feature to bin into). Mode can be 'mean' or 'stdev'
    '''
    # Groubpy using cut
    gb = df.groupby(pd.cut(df['tenure'], bins))
    if mode == 'mean':
        return gb[feature+'Charges'].mean()
    elif mode == 'stdev':
        return gb[feature+'Charges'].std()

def charge_bar_tracer(means, stdevs, labels, name, color, show_stdev):
    '''
    Helper function for charge_over_tenure

    Returns the plotly bar trace
    '''
    return go.Bar(
        x=labels,
        y=means.round(2).values,
        name=name,
        width=.4,
        marker=dict(
            color=color,
            opacity=0.7,
            line=dict(color='dark'+color, width=1)
        ),
        error_y=dict(
            type='data',
            array=stdevs.round(2).values,
            visible=show_stdev
        ),
    )

@app.callback(
    Output('eda-insights', 'children'),
    [Input('eda-tabs', 'value')])
def display_tab(tab):
    return "Catz are theeee bessstt"

if __name__ == '__main__':
    app.run_server(debug=True)
