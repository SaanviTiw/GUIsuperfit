import dash
import json
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_table
import numpy as np
from extinction import ccm89, apply, remove
import pandas as pd
import plotly 
import plotly.plotly as py
import plotly.graph_objs as go
import base64
import io

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.scripts.config.serve_locally = True

# Navigation bar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Link", href="#")),
        dbc.DropdownMenu(
            nav=True,
            in_navbar=True,
            label="Menu",
            children=[
                dbc.DropdownMenuItem("Entry 1"),
                dbc.DropdownMenuItem("Entry 2"),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem("Entry 3"),
            ],
        ),
    ],
    brand="Superfit",
    brand_href="#",
    sticky="top",
)


known_redshift_tab_content = dbc.Card(
    dbc.CardBody(
        dbc.Form(
            [
                dbc.FormGroup(
                    [
                        dbc.Label("z", className="mr-1"),
                        dbc.Input(
                        	id='z-known',
                        	type="number", 
                        	className="mr-1",
                        	bs_size="sm"),
                    ],
                    className="mr-1", 
                ),
            ],
            inline=True,
        ),
    ),
    className="mt-1",
)

redshift_range_tab_content = dbc.Card(
    dbc.CardBody(
        dbc.Form(
            [
             #   dbc.FormGroup(
             #       [
                        dbc.Label("z1", className="mr-1"),
                        dbc.Input(
                        	id='z1-input',
                        	type="number", 
                        	className="col-sm-3 mr-1",
                        	bs_size="sm"),
                        dbc.Label("z2", className="mr-1"),
                        dbc.Input(
                        	id='z2-input',
                        	type="number", 
                        	className="col-sm-3 mr-1",
                        	bs_size="sm"),
                        dbc.Label("dz", className="mr-1"),
                        dbc.Input(
                        	id='dz-input',
                        	type="number", 
                        	className="col-sm-3 mr-1",
                        	bs_size="sm"),
              #      ],
              #      className="mr-1", 
             #   ),
            ],
            inline=True,
        )
    ),
    className="mt-1",
)


zform = dbc.Tabs(
    [
        dbc.Tab(known_redshift_tab_content, label="Known redshift",className="mr-1"),
        dbc.Tab(redshift_range_tab_content, label="Redshift range",className="mr-1")
    ]
)


# Redshift component of form
old_zform = dbc.Row([
    dbc.Form(
        [
            dbc.FormGroup(
                [
                    dbc.Label("z", className="mr-1"),
                    dbc.Input(type="number", className="mr-1",bs_size="sm"),
                ],
                className="mr-3", 
            ),
        ],
        inline=True,
        ),
    html.Details([
        html.Summary('Redshift range'),
        dbc.Form(
            [
                dbc.FormGroup(
                    [
                        dbc.Label("z end", className="mr-1"),
                        dbc.Input(type="number", className="col-sm-1 mr-1",bs_size="sm"),
                        dbc.Label("z interval", className="mr-1"),
                        dbc.Input(type="number", className="col-sm-1 mr-1",bs_size="sm"),
                    ],
                    className="mr-3", 
                ),
            ],
            inline=True,
        )
    ])
])

uploader = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '2px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    #dcc.Graph(id = 'output-data-upload'),
    #html.Div(id='output-data-upload'),
])


#Store info about choices user made to send to sggui
param_storage = dcc.Store(
    id='parameter-storage', 
    storage_type='local'
    )
df_storage = dcc.Store(
    id='df-storage', 
    storage_type='local'
    )



button = dbc.Button(
    "Run fit", 
    color="secondary",
    id="run-button"
    )

report=html.Div(
    id="report-box",
    )

sn_list = ['Ia', 'Iax', 'Ia-pec','Ib', 'Ic', 'Ic-BL', 'II', 'IIb', 'IIn', 'II-pec', 'Ca-rich','SLSN-I', 'SLSN-II']
#sn_dict = dict(zip(sn_list, sn_list))

sn_checklist=dbc.FormGroup([
    html.H6('Supernova types'),
    dbc.Checklist(
        id='SN-types',
        options=[
            {'label': i, 'value': i} for i in sn_list
        ],
        values=sn_list,
        inline=True,
        className = 'sm-2'
    )
],className='card p-1')

epochs=dbc.FormGroup([
    html.H6('Epochs (relative to maximum)'),
    #dbc.Form(
    #    [
    #        dbc.Label("Begin", className="mr-1"),
    #        dbc.Input(type="number", className="col-sm-3 mr-1",bs_size="sm",id="begin-epoch-form"),
    #        dbc.Label("End", className="mr-1"),
    #        dbc.Input(type="number", className="col-sm-3 mr-1",bs_size="sm",id="end-epoch-form"), 
    #    ],
    #    inline=True,
    #    id="epoch-form"
    #),
    dcc.RangeSlider(
        id='epoch-slider',
        count=1,
        min=-100,
        max=700,
        step=1,
        updatemode='drag',
        value=[-30,30]
    ),
    html.Div(id='output-container-range-slider')
],className='card p-1')

wavelimiter=dbc.FormGroup([
    #html.H6('Wavelengths to trim'),
    dcc.RangeSlider(
        id='wave-slider',
        count=1,
        min=3000,
        max=10000,
        step=1,
        updatemode='drag',
        value=[3400,9500]
    ),
    html.Div(id='output-container-waveslider')
],className='card p-1')

# Galaxies
galaxy_list=['E','S0','Sa','Sb','Sc','SB1','SB2','SB3','SB4','SB5','SB6']
galaxy_checklist=html.Details([
    html.Summary('Galaxies'),
    dbc.FormGroup([
        dbc.Checklist(
        	id="galaxy-types",
            options=[
                {'label': i, 'value': i} for i in galaxy_list
            ],
            values=[],
            inline=True,
            className = 'sm-2'
        )
    ],className='card p-1')
])

figure = dcc.Graph(
    id='spectrum-figure',
    figure={'data' : [
                go.Scatter(
                x = [4000, 5000, 6000], 
                y = [1, 0.4, 0.9],
                mode = 'lines',
                line=dict(
                   color="black",
                   width=2
                   )
                )               
            ],
            'layout' : go.Layout(
                xaxis=dict(
                    title= "Wavelength",
                    tickformat=".0f"
                    ),
                yaxis = dict(
                    title= 'Normalized Flux'
                    ),   
                showlegend=True,
                legend={'x': 0.7, 'y': 0.95},      
            )
        }
    )

#def read_spectrum(spectrum_file):
#    return pd.read_csv(spectrum_file, delim_whitespace=True, names=['wav','flux'],header=None)

def parse_contents(contents, filename):
    #Here's where I have a bug.  This works in upscatter.py
    content_type, content_string = contents.split(',')
    #content_string = contents[0]
    decoded = base64.b64decode(content_string)
    spectrum_file = io.StringIO(decoded.decode('utf-8'))
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                spectrum_file,
                names=['wav','flux'],header=None
                )
        elif 'txt' in filename:
            df = pd.read_csv(spectrum_file, 
                names=['wav','flux'],header=None
                )
            #df = pd.read_csv(
            #    io.StringIO(decoded.decode('utf-8')), delimiter = r'\s+')
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    #figure = dcc.Graph(
    #    figure={"data": [df]}
    #    )
    #return figure 
    #json_df=df.to_json(date_format='iso', orient='split')
    #hidden_div = html.Div([
    #    json_df
        #html.H5(filename),
        #dash_table.DataTable(
        #    data=df.to_dict('records'),
        #    columns=[{'name': i, 'id': i} for i in df.columns],
        #    ),
     #   ],style={'display': 'none'}
     #   )
    #figure = dcc.Graph(
    #    figure={"data": [{"x": df["wav"], "y": df["flux"]}]}
    #    )
    #return hidden_div
    df.name=filename
    return df

# Set up layout on web page
sfbody = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        uploader,
                        zform,
                        sn_checklist,
                        epochs,
                        galaxy_checklist,
                        button,
                        param_storage,
                        df_storage,
                        report,
                        #html.Div(id='intermediate-value', style={'display': 'none'})
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        figure,
                        wavelimiter,
                    ]
                ),
            ]
        )
    ],
    className="mt-4",
)

app.layout = html.Div([navbar, sfbody])

# Display slider values
@app.callback(
    Output(component_id='output-container-range-slider', component_property='children'),
    [Input(component_id='epoch-slider', component_property='value')])
def epoch_slider_update(value):
    return 'Epoch range "{}"'.format(value)

# Show wavelenths below wavelength slider
@app.callback(
    Output(component_id='output-container-waveslider', component_property='children'),
    [Input(component_id='wave-slider', component_property='value')])
def wave_slider_update(value):
    return 'Wavelength range "{}"'.format(value)

# Remember SN Types selected
#Take ouput values from SN type checklist and pass them to local-storage
@app.callback(Output(component_id='parameter-storage', component_property='data'),
    [Input(component_id='z-known', component_property='value'),
     Input(component_id='z1-input', component_property='value'),
     Input(component_id='z2-input', component_property='value'),
     Input(component_id='dz-input', component_property='value'),
     Input(component_id='SN-types', component_property='values'),
     Input(component_id='epoch-slider', component_property='value'),
     Input(component_id='galaxy-types', component_property='values'),
     Input(component_id='wave-slider', component_property='value'),
     #Input(component_id='upload-data', component_property='filename')
     ],
    #[State(component_id='SN-types', component_property='modified_timestamp')]
    )

#Every time a SN type is selected
def update_storage(z,z1,z2,dz,SNe_selected, epoch_ranges,galaxies_selected,wavelength_ranges):
    #print(SN_type_values)
    #if SNe_selected is None:
        # prevent the None callbacks is important with the store component.
        # you don't want to update the store for nothing.
        #raise fdate

    # Give a default data dict with 0 clicks if there's no data.
    #data = data or {'clicks': 0}
    #data['clicks'] = data['clicks'] + 1
    params={'z':z,'z1':z1,'z2':z2,'dz':dz,'SN types':SNe_selected,'Epochs':epoch_ranges, 'Galaxy types':galaxies_selected, 'Wavelengths': wavelength_ranges}
    return params

# output stuff to be stored into report-box
@app.callback(Output(component_id='report-box', component_property='children'),
                  # Since we use the data prop in an output,
                  # we cannot get the initial data on load with the data prop.
                  # To counter this, you can use the modified_timestamp
                  # as Input and the data as State.
                  # This limitation is due to the initial None callbacks
                  # https://github.com/plotly/dash-renderer/pull/81
    [Input(component_id='parameter-storage',component_property='data'),
    Input(component_id='df-storage',component_property='data')],
    #[State(omponent_id='parameter-storage', component_property='data')]
    )
def report_stored(stored_params,stored_df):
    #if stored_stuff is None:
    #    raise PreventUpdate
    #print(stored_stuff)
    #stored_stuff = stored_stuff or []
    #output_stored = 'Parameters: "{}", Spectrum: "{}"'.format(stored_params,stored_df)
    output_stored = 'Parameters: "{}"'.format(stored_params)
    return output_stored
######### https://kyso.io/KyleOS/creating-an-interactive-application-using-plotlys-dash

#Take the input form upload-data and put it into spectrum-figure
@app.callback(Output(component_id='spectrum-figure', component_property='figure'),
	           #Output(component_id='parameter-storage', component_property='data')],
            [Input(component_id='upload-data', component_property='contents'),
             Input(component_id='wave-slider', component_property='value')],
            [State(component_id='upload-data', component_property='filename')]
            )

def update_figure(contents, wave_limits, filename):

    #df = parse_contents(contents, filename)

    if contents is not None:
        dff = parse_contents(contents, filename)
        #do this in one line?
        #filtered_dff = dff[is_between=wave_limits[0] <= dff["wav"] <=wave_limits[1]]
        filtered_dff = dff[dff["wav"] > wave_limits[0]]
        filtered_dff = filtered_dff[filtered_dff["wav"] < wave_limits[1]]
        return{
            'data' : [
                go.Scatter(
                x = dff['wav'],
                y = dff['flux'],
                mode = 'lines',
                line=dict(
                   color="gray",
                   width=1
                   ),
                showlegend=False,
                ),
                go.Scatter(
                x = filtered_dff['wav'],
                y = filtered_dff['flux'],
                name=dff.name,
                mode = 'lines',
                line=dict(
                   color="black",
                   width=2
                   )
                )               
            ],
            'layout' : go.Layout(
                xaxis=dict(
                    title= "Wavelength",
                    tickformat=".0f"
                    ),
                yaxis = dict(
                    title= 'Normalized Flux'
                    ),   
                showlegend=True,
                legend={'x': 0.7, 'y': 0.95},      
            )
        }
    else:
        return{}

@app.callback(Output(component_id='df-storage', component_property='data'),
	           #Output(component_id='parameter-storage', component_property='data')],
            [Input(component_id='upload-data', component_property='contents'),
             Input(component_id='wave-slider', component_property='value')],
            [State(component_id='upload-data', component_property='filename')]
            )

def update_df_storage(contents, wave_limits, filename):
    if contents is not None:
        dff = parse_contents(contents, filename)
        filtered_dff = dff[dff["wav"] > wave_limits[0]]
        filtered_dff = filtered_dff[filtered_dff["wav"] < wave_limits[1]]
        # You can't return a data frame
        output = {'wav': filtered_dff["wav"],'flux': filtered_dff["flux"]}
        return output


if __name__ == "__main__":
    app.run_server(debug=True)