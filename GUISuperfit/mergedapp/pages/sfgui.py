'''import dash
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

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],)
app.scripts.config.serve_locally = True
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
app.layout = dbc.Container(
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

# Run Fit button
run_fit_button = dbc.Button(
	"Run Fit",
	color="primary",
	id="run-fit-button",
	className="mr-1"
)

# Define app layout
app.layout = html.Div([
	navbar,
	dbc.Container([
    	dbc.Row([
        	dbc.Col(zform, md=6),
        	dbc.Col(html.Div([uploader, generate_json_button, run_fit_button]), md=6),
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






app.layout = html.Div([
    html.H1("Supernovae"),
    sn_checklist,
    html.Div(id='output-selected-supernovae')
])



epochs = dbc.CardGroup([
	dbc.CardBody([
    	html.H6('Epochs Range Slider'),
    	dcc.RangeSlider(
        	id='epoch-slider',
        	min=-100,
        	max=700,
        	step=1,
        	value=[-30, 30],
        	marks={i: {'label': str(i), 'style': {'color': 'black'}} for i in range(-100, 701, 100)},
        	allowCross=False,
        	updatemode='drag'
    	),
    	html.Div(id='output-container-range-slider', style={'marginTop': 20})
	]),
])


app.layout = html.Div([
	sn_checklist
])



# Other components in your layout
app.layout = html.Div([
	epochs,
	dcc.Upload(id='upload-data'),
	html.Div(id='output-container-waveslider'),
	# Add other components as per your existing layout
])

app.layout = html.Div([
	navbar,
	dbc.Container([
    	dbc.Row([
        	dbc.Col(redshift_range_tab_content, md=6),
        	dbc.Col(html.Div("Other content"), md=6),
    	], className="mt-4"),
	]),
])


app.layout = html.Div([
	html.H6('Wavelength Range Slider'),
	dcc.RangeSlider(
    	id='wave-slider',
    	min=3000,
    	max=10000,
    	step=1,
    	value=[3000, 10000],
    	marks={3000: {'label': '3000', 'style': {'color': 'black'}},
           	10000: {'label': '10000', 'style': {'color': 'black'}}},
    	allowCross=False,  # Prevents range from crossing
    	updatemode='drag'   # Update only on mouse release
	),
	html.Div(id='output-container-waveslider', style={'marginTop': 20})
])

# Galaxies
galaxy_list=['E','S0','Sa','Sb','Sc','SB1','SB2','SB3','SB4','SB5','SB6']
galaxy_checklist=html.Details([
	html.Summary('Galaxies'),
	dbc.CardGroup([
    	dbc.Checklist(
        	id="galaxy-types",
        	options=[
            	{'label': i, 'value': i} for i in galaxy_list
        	],
        	value=[],
        	inline=True,
        	className = 'sm-2'
    	)
	],className='card p-1')
])


# Reddening options with fillable input boxes
reddening_section = dbc.Card(
    dbc.CardBody(
        dbc.Row(
            [
                dbc.Col(html.Label("A_hi", className="mr-1"), width="auto"),
                dbc.Col(dbc.Input(id='A_hi-input', type="number", size="sm", className='no-arrows'), width=2),
                dbc.Col(html.Label("A_lo", className="mr-1 ml-3"), width="2"),
                dbc.Col(dbc.Input(id='A_lo-input', type="number", size="sm", className='no-arrows'), width=2),
                dbc.Col(html.Label("A_i", className="mr-1 ml-3"), width="auto"),
                dbc.Col(dbc.Input(id='A_i-input', type="number", size="sm", className='no-arrows'), width=2),
            ],
            className="align-items-center"
        ),
    ),
    className="mt-1",
)


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



def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = None
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), names=['wav', 'flux'], header=None)
        elif 'fits' in filename:
            with fits.open(io.BytesIO(decoded)) as hdul:
                data = hdul[1].data
                df = pd.DataFrame(data)
                if 'wav' not in df.columns or 'flux' not in df.columns:
                    raise ValueError("FITS file does not contain 'wav' or 'flux' columns")
        elif 'dat' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), delim_whitespace=True, names=['wav', 'flux'], header=None)
        else:
            raise ValueError("Unsupported file type")
        df['wav'] = pd.to_numeric(df['wav'], errors='coerce')
        df['flux'] = pd.to_numeric(df['flux'], errors='coerce')
        df = df.dropna()
    except Exception as e:
        print(f"There was an error processing the file {filename}: {e}")
        df = pd.DataFrame()
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
                    	reddening_section,
                    	run_fit_button,
                    	generate_json_button,
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
                    	app.layout,
                	]
            	),
        	]
    	)
	],
	className="mt-4",
)

app.layout = html.Div([navbar, sfbody, json_output])


@app.callback(
	Output('output-container-range-slider', 'children'),
	Input('epoch-slider', 'value'))
def epoch_slider_update(value):
	return 'Epoch Range: "{}"'.format(value)

#wavelength slider callback
@app.callback(
	Output('output-container-waveslider', 'children'),
	Input('wave-slider', 'value'))
def wave_slider_update(value):
	return 'Wavelength range "{}"'.format(value)


#Take ouput values from SN type checklist and pass them to local-storage
@app.callback(Output(component_id='parameter-storage', component_property='data'),
	[Input(component_id='z-known', component_property='value'),
 	Input(component_id='z1-input', component_property='value'),
 	Input(component_id='z2-input', component_property='value'),
 	Input(component_id='dz-input', component_property='value'),
 	Input(component_id='SN-types', component_property='value'),
 	Input(component_id='epoch-slider', component_property='value'),
 	Input(component_id='galaxy-types', component_property='value'),
 	Input(component_id='wave-slider', component_property='value'),
 	Input(component_id='A_hi-input', component_property='value'),
 	Input(component_id='A_lo-input', component_property='value'),
 	Input(component_id='A_i-input', component_property='value'),
 	#Input(component_id='upload-data', component_property='filename')
 	],
	#[State(component_id='SN-types', component_property='modified_timestamp')]
	)


#Every time a SN type is selected
def update_storage(z,z1,z2,dz,SNe_selected, epoch_ranges,galaxies_selected,wavelength_ranges, A_hi, A_lo, A_i):
	#print(SN_type_values)
	#if SNe_selected is None:
    	# prevent the None callbacks is important with the store component.
    	# you don't want to update the store for nothing.
    	#raise fdate

	# Give a default data dict with 0 clicks if there's no data.
	#data = data or {'clicks': 0}
	#data['clicks'] = data['clicks'] + 1
	params={'z':z,'z1':z1,'z2':z2,'dz':dz,'SN types':SNe_selected,'Epochs':epoch_ranges, 'Galaxy types':galaxies_selected, 'Wavelengths': wavelength_ranges, 'A_hi':A_hi, 'A_lo':A_lo, 'A_i':A_i}
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
	#	raise PreventUpdate
	#print(stored_stuff)
	#stored_stuff = stored_stuff or []
	#output_stored = 'Parameters: "{}", Spectrum: "{}"'.format(stored_params,stored_df)
	output_stored = 'Parameters: "{}"'.format(stored_params)
	return output_stored

@app.callback(
	Output(component_id='spectrum-figure', component_property='figure'),
	[Input(component_id='upload-data', component_property='contents'),
 	Input(component_id='wave-slider', component_property='value')],
	[State(component_id='upload-data', component_property='filename')]
)
def update_figure(contents, wave_limits, filename):
    if contents is not None:
        dff = parse_contents(contents, filename)
        if dff is None:
            return {'data': []}  # Return empty data if parsing failed
        filtered_dff = dff[(dff["wav"] > wave_limits[0]) & (dff["wav"] < wave_limits[1])]
        # Construct the figure data with Plotly traces
        figure_data = [
            {
                'x': dff['wav'],
                'y': dff['flux'],
                'mode': 'lines',
                'line': {'color': 'gray', 'width': 1},
                'showlegend': False,
            },
            {
                'x': filtered_dff['wav'],
                'y': filtered_dff['flux'],
                'name': filename,
                'mode': 'lines',
                'line': {'color': 'black', 'width': 2},
            }
        ]
        # Return the figure dictionary
        return {
            'data': figure_data,
            'layout': {
                'xaxis': {'title': 'Wavelength', 'tickformat': '.0f'},
                'yaxis': {'title': 'Normalized Flux'},
                'showlegend': True,
                'legend': {'x': 0.7, 'y': 0.95},
            }
        }
    else:
        return {'data': []}  # Return empty data if no file is uploaded




@app.callback(
	Output('SN-types', 'value'),
	Input('SN-types', 'value'),
	State('SN-types', 'options')
)
def update_sn_selection(selected_values, options):
    if not selected_values:
        return []
    updated_values = set(selected_values)
    for category, sn_types in sn_categories.items():
        # If the category is selected, add its supernova types
        if category in selected_values:
            updated_values.update(sn_types)
        # If the category is not selected, remove its supernova types
        else:
            updated_values.difference_update(sn_types)
    # Only return selected values that are either categories or individual supernova types
    return [value for value in updated_values if value in selected_values or value in sn_categories or any(value in sn_types for sn_types in sn_categories.values())]







# Function to write JSON to a file
def write_json(target_path, target_file, data):
    if not os.path.exists(target_path):
        try:
            os.makedirs(target_path)
        except Exception as e:
            print(e)
            raise
    with open(os.path.join(target_path, target_file), 'w') as f:
        json.dump(data, f)

# Callback to generate the JSON
@app.callback(
    Output('json-output', 'children'),
    [Input('generate-json-button', 'n_clicks')],
    [
        State('z-known', 'value'),
        State('z1-input', 'value'),
        State('z2-input', 'value'),
        State('dz-input', 'value'),
        State('upload-data', 'contents'),
        State('upload-data', 'filename'),
    	State('SN-types', 'value'),
        State('epoch-slider', 'value'),
        State('galaxy-types', 'value'),
        State('wave-slider', 'value'),
        State('A_hi-input', 'value'),
        State('A_lo-input', 'value'),
        State('A_i-input', 'value'),
    ]
)
def generate_json(n_clicks, z_known, z1_input, z2_input, dz_input, contents, filename, SNe_selected, epoch_ranges, galaxies_selected, wavelength_ranges, A_hi, A_lo, A_i):
    if n_clicks is None:
        raise PreventUpdate
    parameters = {}
    if filename:
        parameters['object_to_fit'] = filename
    if z_known is not None:
        parameters['use_exact_z'] = 1
        parameters['z_exact'] = z_known
    if z1_input is not None and z2_input is not None and dz_input is not None:
        parameters['z_range_begin'] = z1_input
        parameters['z_range_end'] = z2_input
        parameters['z_int'] = dz_input
    if SNe_selected:
        parameters['temp_sn_tr'] = SNe_selected
    if galaxies_selected:
        parameters['temp_gal_tr'] = galaxies_selected
    if wavelength_ranges:
        parameters['lower_lam'] = wavelength_ranges[0]
        parameters['upper_lam'] = wavelength_ranges[1]
    if epoch_ranges:
        parameters['epoch_high'] = epoch_ranges[1]
        parameters['epoch_low'] = epoch_ranges[0]
    if A_hi is not None:
        parameters['A_hi'] = A_hi
    if A_lo is not None:
        parameters['A_lo'] = A_lo
    if A_i is not None:
        parameters['A_i'] = A_i
   
    parameters['resolution'] = 10
    parameters['error_spectrum'] = "sg"
    parameters['saving_results_path'] = ""
    parameters['show_plot'] = 1
    parameters['how_many_plots'] = 5
    parameters['mask_galaxy_lines'] = 1
    parameters['mask_telluric'] = 1
    parameters['minimum_overlap'] = 0.7
    if contents:
        df = parse_contents(contents, filename)
        if not df.empty:
            df_dict = df.to_dict('records')
            parameters['data'] = df_dict

    save_dir = os.path.expanduser('~/Superfit/NGSF')
    save_file = 'fit_parameters.json'
    write_json(save_dir, save_file, parameters)

    json_str = json.dumps(parameters, indent=4)
    b64_json = base64.b64encode(json_str.encode()).decode('utf-8')
    href = f"data:application/json;charset=utf-8;base64,{b64_json}"
    return html.A("Download JSON", href=href, download="fit_parameters.json")







# Callback to store the uploaded file
@app.callback(
    Output('df-storage', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def store_uploaded_file(contents, filename):
    if contents:
        df = parse_contents(contents, filename)
        if not df.empty:
            return df.to_dict('records')
    return None

def store_uploaded_file(contents, filename):
    if contents:
        df = parse_contents(contents, filename)
        if not df.empty:
            return df.to_dict('records')
    return None
    
def update_df_storage(contents, wave_limits, filename):
    if contents is not None:
        dff = parse_contents(contents, filename)
        filtered_dff = dff[dff["wav"] > wave_limits[0]]
        filtered_dff = filtered_dff[filtered_dff["wav"] < wave_limits[1]]
        # You can't return a data frame
        output = {'wav': filtered_dff["wav"], 'flux': filtered_dff["flux"]}
        return output



@app.callback(
    Output('run-fit-button', 'children'),
    [Input('run-fit-button', 'n_clicks')],
    [State('parameter-storage', 'data')]
)
def run_fit(n_clicks, parameters):
    if n_clicks is None:
        raise PreventUpdate

    # Activate the conda environment and run the command
    fit_command = "source ~/miniconda3/etc/profile.d/conda.sh && conda activate NGSF && cd ~/Superfit/NGSF && python run.py fit_parameters.json"

    # Open a new terminal window and execute the command
    subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', f'{fit_command}; exec bash'])

    return "Running Fit..."  # Update the button text or return any message



if __name__ == '__main__':
    app.run_server(debug=True)
'''
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, callback
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import base64
import io
from astropy.io import fits

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

# Layout components
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

zform = dbc.Tabs(
    [
        dbc.Tab(known_redshift_tab_content, label="Known redshift", className="mr-1"),
        dbc.Tab(redshift_range_tab_content, label="Redshift range", className="mr-1")
    ]
)

uploader = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
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
        multiple=False
    ),
])

param_storage = dcc.Store(id='parameter-storage', storage_type='session')
df_storage = dcc.Store(id='df-storage', storage_type='session')

generate_json_button = dbc.Button("Generate JSON", color="secondary", id="generate-json-button", className="mr-1")
run_fit_button = dbc.Button("Run Fit", color="primary", id="run-fit-button", className="mr-1")
json_output = html.Div(id='json-output')

# Supernova types and options
sn_categories = {
    'IA': ["Ia 02es-like", "Ia-02cx like", "Ia-CSM-(ambiguous)", "Ia 91T-like", "Ia-CSM", "Ia-norm", "Ia 91bg-like", "Ia-rapid"],
    'IB': ["Ib", "Ca-Ib"],
    'II': ["IIb-flash", "II", "IIb", "II-flash", "ILRT"],
    'SLSN': ["SLSN-II", "SLSN-IIn", "SLSN-I", "SLSN-Ib", "SLSN-IIb"],
    'Other': ["computed", "TDE He", "Ca-Ia", "super_chandra", "IIn", "FBOT", "Ibn", "TDE H", "SN - Imposter", "TDE H+He", "Ic", "Ia-pec", "Ic-BL", "Ic-pec"]
}

sn_options = [{'label': category, 'value': category} for category in sn_categories.keys()]
for sn_types in sn_categories.values():
    sn_options.extend([{'label': f"  {sn}", 'value': sn} for sn in sn_types])

sn_checklist = dbc.CardGroup([
    html.H6('Supernova types'),
    dbc.Checklist(
        id='SN-types',
        options=sn_options,
        value=[],
        inline=True,
        className='sm-2'
    )
], className='card p-1')

# Epochs and wavelength sliders
epochs = dbc.CardGroup([
    dbc.CardBody([
        html.H6('Epochs Range Slider'),
        dcc.RangeSlider(
            id='epoch-slider',
            min=-100,
            max=700,
            step=1,
            value=[-30, 30],
            marks={i: {'label': str(i), 'style': {'color': 'black'}} for i in range(-100, 701, 100)},
            allowCross=False,
            updatemode='drag'
        ),
        html.Div(id='output-container-range-slider', style={'marginTop': 20})
    ]),
])

wavelength_slider = html.Div([
    html.H6('Wavelength Range Slider'),
    dcc.RangeSlider(
        id='wave-slider',
        min=3000,
        max=10000,
        step=1,
        value=[3000, 10000],
        marks={3000: {'label': '3000', 'style': {'color': 'black'}},
               10000: {'label': '10000', 'style': {'color': 'black'}}},
        allowCross=False,
        updatemode='drag'
    ),
    html.Div(id='output-container-waveslider', style={'marginTop': 20})
])

# Galaxies checklist
galaxy_list = ['E', 'S0', 'Sa', 'Sb', 'Sc', 'SB1', 'SB2', 'SB3', 'SB4', 'SB5', 'SB6']
galaxy_checklist = html.Details([
    html.Summary('Galaxies'),
    dbc.CardGroup([
        dbc.Checklist(
            id="galaxy-types",
            options=[{'label': i, 'value': i} for i in galaxy_list],
            value=[],
            inline=True,
            className='sm-2'
        )
    ], className='card p-1')
])

# Reddening section
reddening_section = dbc.Card(
    dbc.CardBody(
        dbc.Row(
            [
                dbc.Col(html.Label("A_hi", className="mr-1"), width="auto"),
                dbc.Col(dbc.Input(id='A_hi-input', type="number", size="sm", className='no-arrows'), width=2),
                dbc.Col(html.Label("A_lo", className="mr-1 ml-3"), width="2"),
                dbc.Col(dbc.Input(id='A_lo-input', type="number", size="sm", className='no-arrows'), width=2),
                dbc.Col(html.Label("A_i", className="mr-1 ml-3"), width="auto"),
                dbc.Col(dbc.Input(id='A_i-input', type="number", size="sm", className='no-arrows'), width=2),
            ],
            className="align-items-center"
        ),
    ),
    className="mt-1",
)

# Spectrum figure
figure = dcc.Graph(
    id='spectrum-figure',
    figure={
        'data': [
            go.Scatter(
                x=[4000, 5000, 6000],
                y=[1, 0.4, 0.9],
                mode='lines',
                line=dict(color="black", width=2)
            )
        ],
        'layout': go.Layout(
            xaxis=dict(title="Wavelength", tickformat=".0f"),
            yaxis=dict(title='Normalized Flux'),
            showlegend=True,
            legend={'x': 0.7, 'y': 0.95},
        )
    }
)

# Define app layout
layout = html.Div([
    navbar,
    dbc.Container([
        dbc.Row([
            dbc.Col(zform, md=6),
            dbc.Col(html.Div([uploader, generate_json_button, run_fit_button]), md=6),
        ], className="mt-4"),
        dbc.Row([
            dbc.Col([
                sn_checklist,
                epochs,
                galaxy_checklist,
                reddening_section,
                wavelength_slider,
                param_storage,
                df_storage,
                #report,
            ], md=4),
            dbc.Col([
                figure,
                json_output,
            ]),
        ]),
    ]),
])


@callback(
    Output('output-container-waveslider', 'children'),
    Input('wave-slider', 'value')
)
def wave_slider_update(value):
    return f'Wavelength range: {value}'


@callback(
    Output('parameter-storage', 'data'),
    [
        Input('z-known', 'value'),
        Input('z1-input', 'value'),
        Input('z2-input', 'value'),
        Input('dz-input', 'value'),
        Input('SN-types', 'value'),
        Input('epoch-slider', 'value'),
        Input('galaxy-types', 'value'),
        Input('wave-slider', 'value'),
        Input('A_hi-input', 'value'),
        Input('A_lo-input', 'value'),
        Input('A_i-input', 'value'),
    ]
)
def update_storage(z, z1, z2, dz, SNe_selected, epoch_ranges, galaxies_selected, wavelength_ranges, A_hi, A_lo, A_i):
    params = {
        'z': z,
        'z1': z1,
        'z2': z2,
        'dz': dz,
        'SN types': SNe_selected,
        'Epochs': epoch_ranges,
        'Galaxy types': galaxies_selected,
        'Wavelengths': wavelength_ranges,
        'A_hi': A_hi,
        'A_lo': A_lo,
        'A_i': A_i
    }
    return params


@callback(
    Output('report-box', 'children'),
    [
        Input('parameter-storage', 'data'),
        Input('df-storage', 'data')
    ]
)
def report_stored(stored_params, stored_df):
    output_stored = f'Parameters: {stored_params}'
    return output_stored


@callback(
    Output('spectrum-figure', 'figure'),
    [
        Input('upload-data', 'contents'),
        Input('wave-slider', 'value')
    ],
    [Input('upload-data', 'filename')]
)
def update_figure(contents, wave_slider_value, filename):
    if contents is None:
        raise PreventUpdate

    df = parse_contents(contents, filename)
    if df.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['wav'],
        y=df['flux'],
        mode='lines',
        line=dict(color='black', width=2)
    ))

    fig.update_layout(
        xaxis_title='Wavelength',
        yaxis_title='Normalized Flux',
        showlegend=True,
        legend=dict(x=0.7, y=0.95)
    )

    return fig

