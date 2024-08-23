import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.graph_objs as go
import base64
import io
import json
from astropy.io import fits
import os
import subprocess
from dash import callback

# Create the Dash app instance with multi-page support
app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define tabs for navigation
tabs = dcc.Tabs(
    id="tabs",
    value='tab-sfgui',
    children=[
        dcc.Tab(label='Sfgui', value='tab-sfgui'),
        dcc.Tab(label='Sggui', value='tab-sggui'),
    ],
    persistence_type='session',
    persistence=True
)

# Define layout
app.layout = html.Div(
    [
        tabs,
        dcc.Location(id="url", refresh=False),
        dash.page_container
    ]
)

# Define callback for tab routing
@app.callback(
    Output('url', 'pathname'),
    Input('tabs', 'value'),
    prevent_initial_call=True
)
def route(tab_value):
    if tab_value == 'tab-sfgui':
        return '/sfgui'
    if tab_value == 'tab-sggui':
        return '/sggui'
    raise PreventUpdate

# Run the server
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
   	 
