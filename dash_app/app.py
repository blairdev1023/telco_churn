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

hover_col = 'TechSupport'

################################## Markdowns ###################################

with open('assets/markdown/overview.md', 'r') as f:
    overview = f.read()
with open('assets/markdown/descriptions.md', 'r') as f:
    descriptions = f.read()
with open('assets/markdown/eda.md', 'r') as f:
    eda = f.read()
with open('assets/markdown/continuous.md', 'r') as f:
    continuous = f.read()

################################# Static Plots #################################
def churn_polar():
    '''
    Returns the plotly figure for the churn polar chart. Since this chart is
    static I don't need to wrap it with a callback
    '''
    # Seperate churn from non-churn and compute means for features
    churn_idxs = df_churn.index
    no_churn_idxs = df_no_churn.index
    churn_means = df_numeric[bin_cols].loc[churn_idxs].mean()
    no_churn_means = df_numeric[bin_cols].loc[no_churn_idxs].mean()

    # Trace stuff
    col_names = churn_means.index

    hovertext = []
    for i, col in enumerate(col_names):
        churn_mean = round(churn_means[i] * 100, 2)
        no_churn_mean = round(no_churn_means[i] * 100, 2)
        # Make text in hover box
        text = col + '<br>'
        text += 'Churn:     ' + str(churn_mean) + '%<br>'
        text += 'No Churn: ' + str(no_churn_mean) + '%<br>'
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
            name='No Churn',
            r=no_churn_means,
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
            html.Div('Exploratory Data Analysis',
            className='section-header--title')
        ]
    ),
    html.Div(
        dcc.Markdown(eda, className='markdown-text'),
        className='markdown-div',
    ),

    # Polar, Description, Pie, and Bars
    html.Div([
        # Polar
        dcc.Graph(
            id='categorical-polar',
            figure=churn_polar(),
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
        ], className='div-bordered',
        style=dict(
            width='39%',
            height=700,
            float='left',
            display='inline-block',
        )),
    ], style=dict(
        width='100%',
        display='inline-block',
        height=700,
        backgroundColor='white'
    )),

    ### Continuous variables
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
                width='44%',
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
                width='30%',
                textAlign='center',
                display='inline-block',
                borderLeft='thin rgb(42, 207, 255) solid'
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
                display='inline-block',
                borderLeft='thin rgb(42, 207, 255) solid'
            )
        )], className='dashboard-div'),
        # Plot
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
        height=500,
        backgroundColor='white'
    )),
    # Continuous Markdown
    html.Div([
        html.Div(
            dcc.Markdown(continuous, className='markdown-text'),
            className='markdown-div-bordered',
            style=dict(
                width='33%',
                float='left',
                display='inline-block',
            ),
        ),
        html.Div(dcc.Tabs(id='eda-tabs', value='tab-1', children=[
            dcc.Tab(label='Tab One', value='tab-1'),
            dcc.Tab(label='Tab Two', value='tab-2'),
            dcc.Tab(label='Tab Three', value='tab-3'),
            dcc.Tab(label='Tab Four', value='tab-4'),
            dcc.Tab(label='Tab Five', value='tab-5'),
        ]),
            style=dict(
                width='66%',
                float='left',
                display='inline-block',
            )
        ),
        html.Div(
            dcc.Markdown(id='eda-insights', className='markdown-text'),
            className='markdown-div',
            style=dict(
                width='66%',
                float='left',
                display='inline-block',
            )
        ),
    ], style=dict(
        width='100%',
        height=400,
        backgroundColor='white',
        display='inline-block',
    ))
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

@app.callback(
    Output('categorical-bar', 'figure'),
    [Input('categorical-polar', 'hoverData')])
def display_bar(hoverData):
    '''
    Makes the figure for the bar plot in the EDA
    '''
    col = check_polar_hoverData(hoverData)
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

    layout = go.Layout(margin=dict(t=5))
    return {'data': data, 'layout': layout}

@app.callback(
    Output('categorical-pie', 'figure'),
    [Input('categorical-polar', 'hoverData')])
def display_pie(hoverData):
    '''
    Makes the figure for the pie plot in the EDA
    '''
    col = check_polar_hoverData(hoverData)

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
        opacity=0.9,
        textposition='inside',
        insidetextfont = dict(color='black'),
        hoverinfo='label+percent',
        name='No Churn',
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
        # title=f'{col_label_dict[col]} by Percentage',
        margin=dict(t=30, b=0, r=0, l=0),
        legend=dict(
            x=0, y=1.1,
            bgcolor='rgba(0,0,0,0)',
            orientation='h',
            traceorder='reversed',
        ),
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
        title = 'Tenure'
        xaxis = {'title': 'Months'}
    else:
        col = feature + 'Charges'
        feature += ' Charges'
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
        kde_frac = len(x)/len(df) # for scaling each normalied kde

        # KDE Trace
        if chart == 'KDE':
            title = feature + ' KDE Plot (Overlaid)'
            X_plot = np.linspace(0, x_upper * 1.2, 1000)
            kde = KernelDensity(kernel='gaussian', bandwidth=5)
            kde = kde.fit(x[:, np.newaxis])
            log_dens = kde.score_samples(X_plot[:, np.newaxis])
            kde_pts = np.exp(log_dens)
            y = kde_pts * kde_frac
            data.append(go.Scatter(
                x=X_plot,
                y=y,
                mode='lines',
                fill='tozeroy',
                name=name,
                line=dict(color=color, width=2)
            ))
        # Histogram Trace
        elif chart == 'Histogram':
            title = feature + ' Histogram (Stacked)'
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
        legend=dict(x=.8, y=1, bgcolor='rgba(0,0,0,0)'),
        barmode='stack'
    )

    return {'data': data, 'layout': layout}

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

    # Groubpy using cut
    gb_churn = df_churn.groupby(pd.cut(df_churn['tenure'], bins))
    gb_no_churn = df_no_churn.groupby(pd.cut(df_no_churn['tenure'], bins))
    # Means
    means_churn = gb_churn[feature+'Charges'].mean()
    means_no_churn = gb_no_churn[feature+'Charges'].mean()
    # Standard Devs.
    st_devs_churn = gb_churn[feature+'Charges'].std()
    st_devs_no_churn = gb_no_churn[feature+'Charges'].std()

    if difference:
        means = means_churn - means_no_churn
        # Error Prop. -  squareroot of the sum of the squares
        sqr_churn = np.square(st_devs_churn) / bars
        sqr_no_churn = np.square(st_devs_no_churn) / bars
        st_devs = np.sqrt(sqr_churn + sqr_no_churn)

        trace = charge_bar_tracer(means, st_devs, labels,
                'Churn - No Churn', 'khaki', show_stdev)
        data = [trace]
    else:
        trace_churn = trace = charge_bar_tracer(means_churn,
                st_devs_churn, labels, 'Churn', 'red', show_stdev)
        trace_no_churn = trace = charge_bar_tracer(means_no_churn,
                st_devs_no_churn, labels, 'No Churn', 'blue', show_stdev)
        data = [trace_churn, trace_no_churn]

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

def charge_bar_tracer(means, st_devs, labels, name, color, show_stdev):
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
            array=st_devs.round(2).values,
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
