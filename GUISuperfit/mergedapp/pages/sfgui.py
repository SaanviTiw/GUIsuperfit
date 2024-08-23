import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
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


dash.register_page(__name__)


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
    	dbc.Row(
        	[
            	dbc.Col(html.Label("z", className="mr-1"), width="auto"),
            	dbc.Col(dbc.Input(id='z-known', type="number", size="sm"), width=5),
        	],
        	className="align-items-center"
    	),
	),
	className="mt-1",
)



redshift_range_tab_content = dbc.Card(
    dbc.CardBody(
        dbc.Row(
            [
            	dbc.Col(html.Label("z1", className="mr-1"), width="auto"),
				dbc.Col(dbc.Input(id='z1-input', type="number", size="sm", className='no-arrows'), width=2),
				dbc.Col(html.Label("z2", className="mr-1"), width="auto"),
				dbc.Col(dbc.Input(id='z2-input', type="number", size="sm", className='no-arrows'), width=2),
				dbc.Col(html.Label("dz", className="mr-1"), width="auto"),
				dbc.Col(dbc.Input(id='dz-input', type="number", size="sm", className='no-arrows'), width=2),

            ],
            className="align-items-center"
        ),
    ),
    className="mt-1",
)



# Layout integration
layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(redshift_range_tab_content, width=6),
            className="mb-4",
        ),
    ],
    fluid=True,
)





zform = dbc.Tabs(
	[
    	dbc.Tab(known_redshift_tab_content, label="Known redshift",className="mr-1"),
    	dbc.Tab(redshift_range_tab_content, label="Redshift range",className="mr-1")
	]
)


# Redshift component of form
old_zform = dbc.Row([
	dbc.Col(
    	dbc.Card(
        	dbc.CardBody(
            	dbc.Row(
                	[
                    	dbc.Col(dbc.Label("z", className="mr-1"), width="auto"),
                    	dbc.Col(
                        	dbc.Input(type="number", className="mr-1", size="sm"),
                        	width="auto"
                    	),
                	],
                	className="mr-3",
            	)
        	),
    	),
    	width="auto"
	),
	dbc.Col(
    	html.Details([
        	html.Summary('Redshift range'),
        	dbc.Card(
            	dbc.CardBody(
                	dbc.Row(
                    	[
                        	dbc.Col(dbc.Label("z end", className="mr-1"), width="auto"),
                        	dbc.Col(
                            	dbc.Input(type="number", className="col-sm-1 mr-1", size="sm"),
                            	width="auto"
                        	),
                        	dbc.Col(dbc.Label("z interval", className="mr-1"), width="auto"),
                        	dbc.Col(
                            	dbc.Input(type="number", className="col-sm-1 mr-1", size="sm"),
                            	width="auto"
                        	),
                    	],
                    	className="mr-3",
                	)
            	),
        	)
    	]),
    	width="auto"
	)
])

UPLOAD_DIRECTORY = os.path.abspath("/home/stiwary/Superfit/NGSF/")
JSON_OUTPUT_DIRECTORY = os.path.abspath("/home/stiwary/Superfit/NGSF")

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
	html.Div(id='output-data-upload'),

	#dcc.Graph(id = 'output-data-upload'),
	#html.Div(id='output-data-upload'),
])

# Define the upload directory (use absolute path for clarity)




#Store info about choices user made to send to sggui
param_storage = dcc.Store(
	id='parameter-storage',
	storage_type='session'
	)
df_storage = dcc.Store(
	id='df-storage',
	storage_type='session'
	)


generate_json_button = dbc.Button(
	"Generate JSON",
	color="secondary",
	id="generate-json-button",
	className="mr-1"
)

json_output = html.Div(id='json-output')

download_json = dcc.Download(id="download-json")


# Run Fit button
run_fit_button = dbc.Button(
	"Run Fit",
	color="primary",
	id="run-fit-button",
	className="mr-1"
)

clear_button = dbc.Button( "Clear", color="danger", id="clear-button", className="mr-1")

# Define app layout
layout = html.Div([
	navbar,
	dbc.Container([
    	dbc.Row([
        	dbc.Col(zform, md=6),
        	dbc.Col(html.Div([uploader, generate_json_button, run_fit_button, clear_button, download_json]), md=6),
    	], className="mt-4"),
    	json_output
	]),
])



report=html.Div(
	id="report-box",
	)



sn_categories = {
	'IA': ["Ia 02es-like", "Ia-02cx like", "Ia-CSM-(ambigious)", "Ia 91T-like", "Ia-CSM", "Ia-norm", "Ia 91bg-like", "Ia-rapid"],
	'IB': ["Ib", "Ca-Ib"],
	'II': ["IIb-flash", "II", "IIb", "II-flash", "ILRT"],
	'SLSN': ["SLSN-II", "SLSN-IIn", "SLSN-I", "SLSN-Ib", "SLSN-IIb"],
	'Other': ["computed", "TDE He", "Ca-Ia", "super_chandra", "IIn", "FBOT", "Ibn", "TDE H", "SN - Imposter", "TDE H+He", "Ic", "Ia-pec", "Ic-BL", "Ic-pec"]
}

# Generate supernova options
sn_options = [{'label': category, 'value': category} for category in sn_categories.keys()]
for sn_types in sn_categories.values():
	sn_options.extend([{'label': f"  {sn}", 'value': sn} for sn in sn_types])

sn_checklist = dbc.CardGroup([
	html.H6('Supernova types'),
	dbc.Checklist(
    	id='SN-types',
    	options=sn_options,
    	value=[],  # Initially none selected
    	inline=True,  # Display options in a single column
    	className='sm-2'
	)
], className='card p-1')
