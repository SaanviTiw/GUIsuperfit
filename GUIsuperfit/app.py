import dash
import dash_bootstrap_components as dbc

external_stylesheets1 = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets2 =  [dbc.themes.BOOTSTRAP]
external_stylesheets3 =  [dbc.themes.FLATLY]
external_stylesheets4 = ['https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/flatly/bootstrap.min.css']

#app = dash.Dash(__name__, external_stylesheets=[external_stylesheets2,external_stylesheets1], suppress_callback_exceptions=True)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets2, suppress_callback_exceptions=True)
#app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server