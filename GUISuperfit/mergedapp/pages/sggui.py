import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.graph_objs as go
from extinction import ccm89, apply, remove
import base64
import io
import json
from astropy.io import fits
import os
import subprocess
import plotly
import numpy as np
from dash import callback
import time
from watchdog.events import FileSystemEventHandler

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
dash.register_page(__name__)


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
                sfo_dictionary[pair[0]] = pair[1]
            else:
                file, S, z, Galaxy, Av, C, F, gfrac, sfrac = row.split()
                category, Type, filename = file.split('/')
                S = float(S)
                z = float(z)
                Av = float(Av)
                C = float(C)
                F = float(F)
                gfrac = float(gfrac)
                sfrac = float(sfrac)
                SN, Epoch, template_filetype = filename.split('.')
                if S < 999:
                    newlist.append([file, SN, Epoch, Type, S, z, Galaxy, Av, C, F, gfrac, sfrac])
    df = pd.DataFrame(newlist, columns=['file', 'SN', 'Epoch', 'Type', 'S', 'z', 'Galaxy', 'Av', 'C', 'F', 'gfrac', 'sfrac'])
    return sfo_dictionary, df
def read_spectrum(spectrum_path):
    return pd.read_csv(spectrum_path, delim_whitespace=True, names=['wav', 'flux'], header=None)
def normalize_spectrum(spectrum):
    median = spectrum['flux'].median()
    spectrum['flux'] = spectrum['flux'] / median
    return spectrum
def scale_spectrum(spectrum, scale_value):
    spectrum['flux'] = spectrum['flux'] * scale_value
    return spectrum
def binspec(spectrum, start_wavelength, end_wavelength, wavelength_bin):
    binned_wavelength = np.arange(start_wavelength, end_wavelength, wavelength_bin)
    binned_flux = np.interp(binned_wavelength, spectrum["wav"], spectrum["flux"], left=np.nan, right=np.nan)
    return pd.DataFrame(list(zip(binned_wavelength, binned_flux)), columns=["wav", "flux"])
def redden_scale_template(template, c, Av, z):
    unredshifted_wav = template["wav"].to_numpy() / (1.0 + z)
    np_template_flux = template["flux"].to_numpy()
    output_flux = apply(ccm89(unredshifted_wav, Av, 3.1), np_template_flux)
    output_df = template.copy()
    output_df["flux"] = c * output_flux
    return output_df
sfo_filename = 'SN2019ein.p02.sfo'
sfo_dictionary, df = read_sfo(sfo_filename)
observation_filename = sfo_dictionary["o"].split('/')[-1]
observed_spectrum = read_spectrum(observation_filename)
normalized_observed_spectrum = normalize_spectrum(observed_spectrum)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
layout = html.Div([
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
            {"name": "file", "id": "file", "deletable": False},
            {"name": "SN", "id": "SN", "deletable": False},
            {"name": "Epoch", "id": "Epoch", "deletable": False},
            {"name": "Type", "id": "Type", "deletable": False},
            {"name": "S", "id": "S", "deletable": False},
            {"name": "z", "id": "z", "deletable": False},
            {"name": "Galaxy", "id": "Galaxy", "deletable": False},
            {"name": "Av", "id": "Av", "deletable": False},
            {"name": "C", "id": "C", "deletable": False},
            {"name": "F", "id": "F", "deletable": False},
            {"name": "gfrac", "id": "gfrac", "deletable": False},
            {"name": "sfrac", "id": "sfrac", "deletable": False},
        ],
        style_header={
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        style_cell_conditional=[
            {'if': {'column_id': 'file'}, 'textAlign': 'left', 'width': '120px'},
            {'if': {'column_id': 'SN'}, 'textAlign': 'left', 'width': '60px'},
            {'if': {'column_id': 'Epoch'}, 'textAlign': 'left', 'width': '60px'},
            {'if': {'column_id': 'Type'}, 'textAlign': 'left', 'width': '60px'},
            {'if': {'column_id': 'S'}, 'width': '40px'},
            {'if': {'column_id': 'z'}, 'width': '40px'},
            {'if': {'column_id': 'Galaxy'}, 'textAlign': 'left', 'width': '120px'},
            {'if': {'column_id': 'Av'}, 'width': '40px'},
            {'if': {'column_id': 'C'}, 'width': '40px'},
            {'if': {'column_id': 'F'}, 'width': '40px'},
            {'if': {'column_id': 'gfrac'}, 'width': '60px'},
            {'if': {'column_id': 'sfrac'}, 'width': '60px'}
        ],
        data=df.to_dict('records'),
        editable=True,
        row_selectable="single",
        row_deletable=True,
        selected_rows=[],
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
                'layout': go.Layout(
                    xaxis=dict(title="Wavelength", tickformat=".0f"),
                    yaxis=dict(title='Normalized Flux'),
                    showlegend=True,
                    legend={'x': 0.9, 'y': 0.95},
                )
            }
        )
    ]
