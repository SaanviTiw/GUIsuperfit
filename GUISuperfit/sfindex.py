#import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from app import app
import sftab as sftab
import sgtab as sgtab
#from sftab import sfapp as sfapp
#from sftab import epoch_slider_update



#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#external_stylesheets2 =  [dbc.themes.BOOTSTRAP]

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets2)
# Not sure if I need this
#app.scripts.config.serve_locally = True


app.layout = html.Div([
    sftab.navbar,
    dcc.Tabs(id='tabs-example', value='setup-tab', children=[
        dcc.Tab(label='Setup Fit', value='setup-tab'),
        dcc.Tab(label='View Results', value='results-tab'),
    ]),
    html.Div(id='tabs-example-content')
])

@app.callback(Output('tabs-example-content', 'children'),
              [Input('tabs-example', 'value')])
def render_content(tab):
    if tab == 'setup-tab':
        return html.Div([
            #html.H3('Tab content 1')
            sftab.sfbody    
        ])
    elif tab == 'results-tab':
        return html.Div([
            sgtab.sgbody
        ])


if __name__ == '__main__':
    app.run_server(debug=True)