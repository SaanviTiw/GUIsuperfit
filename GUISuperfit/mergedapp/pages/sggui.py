'''import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import chart_studio
import plotly
import plotly.graph_objs as go
import base64
import io
import json
from astropy.io import fits
import numpy as np



chart_studio.tools.set_credentials_file(username='ahowell', api_key='IV3kTKFQTTugQOjv5Yga')
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
##### Functions #########
dash.register_page(__name__)
def read_sfo(sfo_filename):
    # initialize
    sfo_dictionary = {}
    newlist = []
    df = pd.DataFrame(columns=['file', 'SN', 'Epoch', 'Type', 'S', 'z', 'Galaxy', 'Av', 'C', 'F', 'gfrac', 'sfrac'])
    opened_file = open(sfo_filename, "r")
    rows = opened_file.readlines()
    for row in rows:
        # read footer
        if row[0] == ';':
            pair = row.split('=', 1)
            # Get rid of comment lines and whitespace
            pair[0] = pair[0].strip(';')
            pair[0] = pair[0].strip()
            pair[1] = pair[1].strip()
            # convert numbers to floats but leave strings
            try:
                pair[1] = float(pair[1])
            except ValueError:
                pass
            sfo_dictionary.update({pair[0]: pair[1]})
        else:
            # read main part of file
            [file, S, z, Galaxy, Av, C, F, gfrac, sfrac] = row.split()
            [category, Type, filename] = file.split('/')
            S = float(S)
            z = float(z)
            Av = float(Av)
            C = float(C)
            F = float(F)
            gfrac = float(gfrac)
            sfrac = float(sfrac)
            [SN, Epoch, template_filetype] = filename.split('.')
            if S < 999:
                list = [file, SN, Epoch, Type, S, z, Galaxy, Av, C, F, gfrac, sfrac]
                newlist.append(list)
    df = pd.DataFrame(newlist, columns=['file', 'SN', 'Epoch', 'Type', 'S', 'z', 'Galaxy', 'Av', 'C', 'F', 'gfrac', 'sfrac'])
    return [sfo_dictionary, df]
    
def read_spectrum(spectrum_path):
    return pd.read_csv(spectrum_path, sep=r'\s+', names=['wav', 'flux'], header=None)
    
    
def normalize_spectrum(spectrum):
    # Normalize flux
    median = np.median(spectrum['flux'])
    spectrum['flux'] = spectrum['flux'] / median
    return spectrum
def scale_spectrum(spectrum, scale_value):
    # Scale spectrum to scale_value
    spectrum['flux'] = spectrum['flux'] * scale_value
    return spectrum
def binspec(spectrum, start_wavelength, end_wavelength, wavelength_bin):
    # rebin spectrum given starting wavelength, ending wavelength, and bin
    binned_wavelength = np.arange(start_wavelength, end_wavelength, wavelength_bin)
    binned_flux = np.interp(binned_wavelength, spectrum["wav"], spectrum["flux"], left=np.nan, right=np.nan)
    zipped = zip(binned_wavelength, binned_flux)
    binned_df = pd.DataFrame(zipped, columns=["wav", "flux"])
    return binned_df
    
def ccm89(wavelength, Av, param):
	result = wavelength * Av * param
	return result

def redden_scale_template(template, c, Av, z):
    unredshifted_wav = template["wav"].to_numpy() / (1.0 + z)
    np_template_flux = template["flux"].to_numpy()
    
    # Assuming ccm89 is defined and returns a function that can be applied
    output_flux = np.apply_along_axis(ccm89, axis=0, arr=unredshifted_wav, Av=Av, param=3.1)
    
    output_df = template.copy()
    output_df["wav"] = template["wav"]
    output_df["flux"] = c * pd.Series(output_flux)
    
    return output_df

###### Main Program ##########
sfo_filename = 'SN2019ein.p02.sfo'
[sfo_dictionary, df] = read_sfo(sfo_filename)
observation_filename = sfo_dictionary["o"].split('/')[-1]
observed_spectrum = read_spectrum(observation_filename)
normalized_observed_spectrum = normalize_spectrum(observed_spectrum)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    html.Div(id='datatable-interactivity-container'),
    html.Label('Show graphs:'),
    dcc.Checklist(
        id='plotting_checklist',
        options=[
            {'label': 'Observation - Galaxy', 'value': 'omg'},
            {'label': 'Template', 'value': 'tem'},
            {'label': 'Galaxy', 'value': 'gal'},
            {'label': 'Observation', 'value': 'obs'},
            {'label': 'Normalized Template', 'value': 'ute'}
        ],
        value=['omg', 'tem'],
        style={'columnCount': 6}
    ),
    html.Label('Binning (A):'),
    dcc.Input(
        id='bin_input',
        size='30px',
        type='number',
        value=sfo_dictionary["disp"],
    ),
    dash_table.DataTable(
        id='datatable-interactivity',
        columns=[
            {"name": i, "id": i, "deletable": False} for i in df.columns[1:]
        ],
        style_header={
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        style_cell_conditional=[
            {'if': {'column_id': 'SN'},
             'textAlign': 'left',
             'width': '20px'},
            {'if': {'column_id': 'Epoch'},
             'textAlign': 'left',
             'width': '20px'},
            {'if': {'column_id': 'Type'},
             'textAlign': 'left',
             'width': '20px'},
            {'if': {'column_id': 'S'},
             'width': '20px'},
            {'if': {'column_id': 'z'},
             'width': '20px'},
            {'if': {'column_id': 'Galaxy'},
             'textAlign': 'left',
             'width': '20px'},
            {'if': {'column_id': 'Av'},
             'width': '20px'},
            {'if': {'column_id': 'C'},
             'width': '20px'},
            {'if': {'column_id': 'F'},
             'width': '20px'},
            {'if': {'column_id': 'gfrac'},
             'width': '20px'},
            {'if': {'column_id': 'sfrac'},
             'width': '20px'}
        ],
        data=df.to_dict('records'),
        editable=True,
        row_selectable="single",
        row_deletable=True,
        style_table={
            'maxHeight': '500px',
            'overflowY': 'scroll',
            'border': 'thin lightgrey solid'
        },
    )
])
@app.callback(
    Output('datatable-interactivity-container', "children"),
    [Input('datatable-interactivity', "derived_virtual_data"),
     Input('datatable-interactivity', "derived_virtual_selected_rows"),
     Input('bin_input', 'value'),
     Input('plotting_checklist', 'value')])
def update_graphs(rows, derived_virtual_selected_rows, bin_input, plotting_checklist):
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []
    if bin_input == 0:
        bin_input = 1
    if plotting_checklist is None:
        plotting_checklist = []
    dff = df if rows is None else pd.DataFrame(rows)
    selection = 0
    for i in derived_virtual_selected_rows:
        selection = int(i)
    selected_row = dff.iloc[selection].to_dict()
    template_path = selected_row["file"]
    template_spectrum = read_spectrum(template_path)
    normalized_template_spectrum = normalize_spectrum(template_spectrum)
    reddened_scaled_template = redden_scale_template(normalized_template_spectrum, selected_row["C"], selected_row["Av"], selected_row["z"])
    scaled_template_spectrum = normalized_template_spectrum.copy()
    scaled_template_spectrum["flux"] = selected_row["C"] * scaled_template_spectrum["flux"]
    galaxy_path = "gal/" + selected_row["Galaxy"]
    galaxy_spectrum = read_spectrum(galaxy_path)
    normalized_galaxy_spectrum = normalize_spectrum(galaxy_spectrum)
    scaled_galaxy_spectrum = scale_spectrum(normalized_galaxy_spectrum, selected_row["F"])
    beginw = 4000
    endw = 8000
    bin_wav = bin_input
    binned_galaxy = binspec(scaled_galaxy_spectrum, beginw, endw, bin_wav)
    binned_template = binspec(normalized_template_spectrum, beginw, endw, bin_wav)
    binned_observed = binspec(normalized_observed_spectrum, beginw, endw, bin_wav)
    ObsMinusGal = binned_observed.copy()
    ObsMinusGal["flux"] = ObsMinusGal["flux"] - binned_galaxy["flux"]
    plot_label = {
        'obs': observation_filename,
        'gal': selected_row["Galaxy"],
        'omg': observation_filename + " - " + selected_row["Galaxy"],
        'tem': selected_row["SN"] + " " + selected_row["Epoch"],
        'ute': 'Scaled template'
    }
    traces = []
    for i in plotting_checklist:
      if i == 'obs':
        spectrum_to_plot = normalized_observed_spectrum
      elif i == 'omg':
        spectrum_to_plot = ObsMinusGal
      elif i == 'gal':
        spectrum_to_plot = scaled_galaxy_spectrum
      elif i == 'tem':
        spectrum_to_plot = reddened_scaled_template
      elif i == 'ute':
        spectrum_to_plot = scaled_template_spectrum
      traces.append(go.Scatter(
        x=spectrum_to_plot["wav"],
        y=spectrum_to_plot["flux"],
        mode='lines',
        name=plot_label[i]
    ))


    return [
        dcc.Graph(
            id="SNgraph",
            figure={
                'data': traces,
                'layout': go.Layout
                    xaxis=dict(
                    title= "Wavelength",
                    tickformat=".0f"
                    ),
                    yaxis=dict(
                    title= 'Normalized Flux'
                    ),
                    #width=1000,
                    showlegend=True,
                    legend={'x': 0.9, 'y': 0.95},
                )
            }
        )
        # check if column exists - user may have deleted it
        # If `column.deletable=False`, then you don't
        # need to do this check.
        #for column in ["pop"] if column in dff
    ]

    
if __name__ == "__main__":
    app.run_server(debug=True) 
 
'''

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, callback
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import chart_studio
import plotly
import plotly.graph_objs as go
import base64
import io
import json
from astropy.io import fits
import numpy as np


#chart_studio.tools.set_credentials_file(username='ahowell', api_key='IV3kTKFQTTugQOjv5Yga')
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
dash.register_page(__name__, path='/sggui')


layout = html.Div([
	html.H1("Sggui"),
])
	
def read_sfo(sfo_filename):
    sfo_dictionary = {}
    newlist = []
    df = pd.DataFrame(columns=['file', 'SN', 'Epoch', 'Type', 'S', 'z', 'Galaxy', 'Av', 'C', 'F', 'gfrac', 'sfrac'])
    with open(sfo_filename, "r") as opened_file:
        rows = opened_file.readlines()
        for row in rows:
            if row[0] == ';':
                pair = row.split('=', 1)
                pair[0] = pair[0].strip(';').strip()
                pair[1] = pair[1].strip()
                try:
                    pair[1] = float(pair[1])
                except ValueError:
                    pass
                sfo_dictionary.update({pair[0]: pair[1]})
            else:
                [file, S, z, Galaxy, Av, C, F, gfrac, sfrac] = row.split()
                [category, Type, filename] = file.split('/')
                S = float(S)
                z = float(z)
                Av = float(Av)
                C = float(C)
                F = float(F)
                gfrac = float(gfrac)
                sfrac = float(sfrac)
                [SN, Epoch, template_filetype] = filename.split('.')
                if S < 999:
                    list = [file, SN, Epoch, Type, S, z, Galaxy, Av, C, F, gfrac, sfrac]
                    newlist.append(list)
    df = pd.DataFrame(newlist, columns=['file', 'SN', 'Epoch', 'Type', 'S', 'z', 'Galaxy', 'Av', 'C', 'F', 'gfrac', 'sfrac'])
    return [sfo_dictionary, df]

def read_spectrum(spectrum_path):
    return pd.read_csv(spectrum_path, sep=r'\s+', names=['wav', 'flux'], header=None)
    
    


def normalize_spectrum(spectrum):
    median = np.median(spectrum['flux'])
    spectrum['flux'] = spectrum['flux'] / median
    return spectrum

def scale_spectrum(spectrum, scale_value):
    spectrum['flux'] = spectrum['flux'] * scale_value
    return spectrum

def binspec(spectrum, start_wavelength, end_wavelength, wavelength_bin):
    binned_wavelength = np.arange(start_wavelength, end_wavelength, wavelength_bin)
    binned_flux = np.interp(binned_wavelength, spectrum["wav"], spectrum["flux"], left=np.nan, right=np.nan)
    zipped = zip(binned_wavelength, binned_flux)
    binned_df = pd.DataFrame(zipped, columns=["wav", "flux"])
    return binned_df

def ccm89(wavelength, Av, param):
    result = wavelength * Av * param
    return result

def redden_scale_template(template, c, Av, z):
    unredshifted_wav = template["wav"].to_numpy() / (1.0 + z)
    np_template_flux = template["flux"].to_numpy()
    output_flux = np.apply_along_axis(ccm89, axis=0, arr=unredshifted_wav, Av=Av, param=3.1)
    output_df = template.copy()
    output_df["wav"] = template["wav"]
    output_df["flux"] = c * pd.Series(output_flux)
    return output_df

sfo_filename = 'SN2019ein.p02.sfo'
[sfo_dictionary, df] = read_sfo(sfo_filename)
observation_filename = sfo_dictionary["o"].split('/')[-1]
observed_spectrum = read_spectrum(observation_filename)
normalized_observed_spectrum = normalize_spectrum(observed_spectrum)

def layout():
    return html.Div([
        html.H1("Sggui Page"),
        html.Div(id='datatable-interactivity-container'),
        html.Label('Show graphs:'),
        dcc.Checklist(
            id='plotting_checklist',
            options=[
                {'label': 'Observation - Galaxy', 'value': 'omg'},
                {'label': 'Template', 'value': 'tem'},
                {'label': 'Galaxy', 'value': 'gal'},
                {'label': 'Observation', 'value': 'obs'},
                {'label': 'Normalized Template', 'value': 'ute'}
            ],
            value=['omg', 'tem'],
            style={'columnCount': 6}

    ),
    html.Label('Binning (A):'),
    dcc.Input(
        id='bin_input',
        size='30px',
        type='number',
        value=sfo_dictionary["disp"],
    ),
    dash_table.DataTable(
        id='datatable-interactivity',
        columns=[{"name": i, "id": i, "deletable": False} for i in df.columns[1:]],
        style_header={
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        style_cell_conditional=[
            {'if': {'column_id': 'SN'}, 'textAlign': 'left', 'width': '20px'},
            {'if': {'column_id': 'Epoch'}, 'textAlign': 'left', 'width': '20px'},
            {'if': {'column_id': 'Type'}, 'textAlign': 'left', 'width': '20px'},
            {'if': {'column_id': 'S'}, 'width': '20px'},
            {'if': {'column_id': 'z'}, 'width': '20px'},
            {'if': {'column_id': 'Galaxy'}, 'textAlign': 'left', 'width': '20px'},
            {'if': {'column_id': 'Av'}, 'width': '20px'},
            {'if': {'column_id': 'C'}, 'width': '20px'},
            {'if': {'column_id': 'F'}, 'width': '20px'},
            {'if': {'column_id': 'gfrac'}, 'width': '20px'},
            {'if': {'column_id': 'sfrac'}, 'width': '20px'}
        ],
        data=df.to_dict('records'),
        editable=True,
        row_selectable="single",
        row_deletable=True,
        style_table={
            'maxHeight': '500px',
            'overflowY': 'scroll',
            'border': 'thin lightgrey solid'
        },
    )
])

@callback(
    Output('datatable-interactivity-container', "children"),
    [Input('datatable-interactivity', "derived_virtual_data"),
     Input('datatable-interactivity', "derived_virtual_selected_rows"),
     Input('bin_input', 'value'),
     Input('plotting_checklist', 'value')]
)
def update_graphs(rows, derived_virtual_selected_rows, bin_input, plotting_checklist):
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []
    if bin_input == 0:
        bin_input = 1
    if plotting_checklist is None:
        plotting_checklist = []
    dff = df if rows is None else pd.DataFrame(rows)
    selection = 0
    for i in derived_virtual_selected_rows:
        selection = int(i)
    selected_row = dff.iloc[selection].to_dict()
    template_path = selected_row["file"]
    template_spectrum = read_spectrum(template_path)
    normalized_template_spectrum = normalize_spectrum(template_spectrum)
    reddened_scaled_template = redden_scale_template(normalized_template_spectrum, selected_row["C"], selected_row["Av"], selected_row["z"])
    scaled_template_spectrum = normalized_template_spectrum.copy()
    scaled_template_spectrum["flux"] = selected_row["C"] * scaled_template_spectrum["flux"]
    galaxy_path = "gal/" + selected_row["Galaxy"]
    galaxy_spectrum = read_spectrum(galaxy_path)
    normalized_galaxy_spectrum = normalize_spectrum(galaxy_spectrum)
    scaled_galaxy_spectrum = scale_spectrum(normalized_galaxy_spectrum, selected_row["F"])
    beginw = 4000
    endw = 8000
    bin_wav = bin_input
    binned_galaxy = binspec(scaled_galaxy_spectrum, beginw, endw, bin_wav)
    binned_template = binspec(normalized_template_spectrum, beginw, endw, bin_wav)
    binned_observed = binspec(normalized_observed_spectrum, beginw, endw, bin_wav)
   # Calculate the difference between observed and galaxy spectra
    ObsMinusGal = binned_observed.copy()
    ObsMinusGal["flux"] = ObsMinusGal["flux"] - binned_galaxy["flux"]

    # Create plot labels
    plot_label = {
        'obs': observation_filename,
        'gal': selected_row["Galaxy"],
        'omg': observation_filename + " - " + selected_row["Galaxy"],
        'tem': selected_row["SN"] + " " + selected_row["Epoch"],
        'ute': 'Scaled template'
    }

    # Create plot traces based on the checklist
    traces = []
    for i in plotting_checklist:
        if i == 'obs':
            spectrum_to_plot = normalized_observed_spectrum
        elif i == 'omg':
            spectrum_to_plot = ObsMinusGal
        elif i == 'gal':
            spectrum_to_plot = scaled_galaxy_spectrum
        elif i == 'tem':
            spectrum_to_plot = reddened_scaled_template
        elif i == 'ute':
            spectrum_to_plot = scaled_template_spectrum
        else:
            continue

        traces.append(go.Scatter(
            x=spectrum_to_plot["wav"],
            y=spectrum_to_plot["flux"],
            mode='lines',
            name=plot_label[i]
        ))

    # Return the updated graph
    return [
        dcc.Graph(
            id="SNgraph",
            figure={
                'data': traces,
                'layout': go.Layout(
                    xaxis=dict(
                        title="Wavelength",
                        tickformat=".0f"
                    ),
                    yaxis=dict(
                        title='Normalized Flux'
                    ),
                    showlegend=True,
                    legend={'x': 0.9, 'y': 0.95},
                )
            }
        )
    ]

if __name__ == '__main__':
    app.run_server(debug=True)


    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
