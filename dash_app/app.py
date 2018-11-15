import dash
import dash_table
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import pandas as pd

df = pd.read_csv('../data/telco_churn_clean.csv')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1('Telco Churn Study'),
    html.Div([
        # I seriously doubt I'll keep this but this was released earlier this
        # month so I figured I'd play with it
        dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('rows'),
            style_table={
                'overflowX': 'scroll',
                'overflowY': 'scroll',
                'maxHeight': '300'
            },
        )
    ], style={'width': '80%', 'margin-left': 'auto', 'margin-right': 'auto'})
])

if __name__ == '__main__':
    app.run_server(debug=True)
