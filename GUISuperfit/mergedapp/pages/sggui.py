import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, Input, Output, State, callback
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.graph_objs as go
import base64
import io
import numpy as np
import os

# Dash app initialization
dash.register_page(__name__)

# Define the directory containing binned spectra files
BINSPEC_DIR = "/home/stiwary/Superfit/GUIsuperfit/mergedapp/NGSF/NGSF"

# Global DataFrame to store CSV data
df = pd.DataFrame(columns=[
    'SPECTRUM', 'GALAXY', 'SN', 'CONST_SN', 'CONST_GAL', 'Z', 'A_v', 'Phase',
    'Band', 'Frac(SN)', 'Frac(gal)', 'CHI2/dof', 'CHI2/dof2'
])

# Function to read and process spectral data
def read_spectrum(spectrum_path):
    return pd.read_csv(spectrum_path, delim_whitespace=True, names=['wav', 'flux'], header=None)

def normalize_spectrum(spectrum):
    median = spectrum['flux'].median()
    spectrum['flux'] = spectrum['flux'] / median
    return spectrum

def binspec(spectrum, start_wavelength, end_wavelength, wavelength_bin):
    binned_wavelength = np.arange(start_wavelength, end_wavelength, wavelength_bin)
    binned_flux = np.interp(binned_wavelength, spectrum["wav"], spectrum["flux"], left=np.nan, right=np.nan)
    return pd.DataFrame(list(zip(binned_wavelength, binned_flux)), columns=["wav", "flux"])

def find_spectrum_file(base_dir, spectrum_name):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith(spectrum_name) and file.endswith('_binned.txt'):
                return os.path.join(root, file)
    return None
    
def adjust_galaxy_spectrum(galaxy_df, const_gal):
    if pd.notna(const_gal) and galaxy_df.shape[1] > 1:
        galaxy_df.iloc[:, 1] = galaxy_df.iloc[:, 1] * const_gal
    return galaxy_df



# Layout of the Dash app
layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Button('Upload CSV'),
        multiple=False
    ),
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
        value=10,
    ),
    dash_table.DataTable(
        id='datatable-interactivity',
        columns=[
            {"name": "SPECTRUM", "id": "SPECTRUM"},
            {"name": "GALAXY", "id": "GALAXY"},
            {"name": "SN", "id": "SN"},
            {"name": "CONST_SN", "id": "CONST_SN"},
            {"name": "CONST_GAL", "id": "CONST_GAL"},
            {"name": "Z", "id": "Z"},
            {"name": "A_v", "id": "A_v"},
            {"name": "Phase", "id": "Phase"},
            {"name": "Band", "id": "Band"},
            {"name": "Frac(SN)", "id": "Frac(SN)"},
            {"name": "Frac(gal)", "id": "Frac(gal)"},
            {"name": "CHI2/dof", "id": "CHI2/dof"},
            {"name": "CHI2/dof2", "id": "CHI2/dof2"}
        ],
        style_header={
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        style_cell_conditional=[
            {'if': {'column_id': 'SPECTRUM'}, 'textAlign': 'left', 'width': '150px'},
            {'if': {'column_id': 'GALAXY'}, 'textAlign': 'left', 'width': '100px'},
            {'if': {'column_id': 'SN'}, 'textAlign': 'left', 'width': '100px'},
            {'if': {'column_id': 'CONST_SN'}, 'textAlign': 'left', 'width': '100px'},
            {'if': {'column_id': 'CONST_GAL'}, 'textAlign': 'left', 'width': '100px'},
            {'if': {'column_id': 'Z'}, 'textAlign': 'left', 'width': '60px'},
            {'if': {'column_id': 'A_v'}, 'textAlign': 'left', 'width': '60px'},
            {'if': {'column_id': 'Phase'}, 'textAlign': 'left', 'width': '80px'},
            {'if': {'column_id': 'Band'}, 'textAlign': 'left', 'width': '60px'},
            {'if': {'column_id': 'Frac(SN)'}, 'textAlign': 'left', 'width': '80px'},
            {'if': {'column_id': 'Frac(gal)'}, 'textAlign': 'left', 'width': '80px'},
            {'if': {'column_id': 'CHI2/dof'}, 'textAlign': 'left', 'width': '80px'},
            {'if': {'column_id': 'CHI2/dof2'}, 'textAlign': 'left', 'width': '80px'},
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


'''
@callback(
    [Output('datatable-interactivity', 'data'),
     Output('datatable-interactivity', 'columns'),
     Output('datatable-interactivity-container', 'children')],
    [Input('upload-data', 'contents'),
     Input('datatable-interactivity', 'derived_virtual_selected_rows'),
     Input('bin_input', 'value'),
     Input('plotting_checklist', 'value')]
)
def update_graphs(uploaded_contents, derived_virtual_selected_rows, bin_input, plotting_checklist):
    global df

    BASE_DIR = "/home/stiwary/Superfit/NGSF/"
    GAL_DIR_SUFFIX = 'bank/binnings/10A/gal/'  # Directory suffix for galaxy spectra
    expected_columns = [
        'SPECTRUM', 'GALAXY', 'SN', 'CONST_SN', 'CONST_GAL', 'Z', 'A_v', 'Phase', 'Band',
        'Frac(SN)', 'Frac(gal)', 'CHI/dof', 'sn_name'
    ]

    # Read and process uploaded CSV data if available
    if uploaded_contents is not None:
        content_type, content_string = uploaded_contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            csv_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            for col in expected_columns:
                if col not in csv_df.columns:
                    csv_df[col] = pd.NA

            csv_df = csv_df[expected_columns]
        except Exception as e:
            print(f"Error reading file: {e}")
            raise PreventUpdate
    else:
        csv_df = pd.DataFrame(columns=expected_columns)

    # Update data table
    columns = [{"name": col, "id": col} for col in csv_df.columns]
    data = csv_df.to_dict('records')

    # Initialize traces list for plotting
    traces = []

    # Handle selected row for graph generation
    if derived_virtual_selected_rows:
        selection = derived_virtual_selected_rows[0]
        selected_row = csv_df.iloc[selection]

        # Plotting the spectrum from sn_name column if 'Template' is checked
        if 'tem' in plotting_checklist:
            sn_name = selected_row.get('sn_name')
            if pd.notna(sn_name):
                spectrum_path = os.path.join(BASE_DIR, sn_name)
                if os.path.isfile(spectrum_path):
                    try:
                        print(f"Processing SN spectrum file: {spectrum_path}")
                        spectrum_df = pd.read_csv(spectrum_path, delim_whitespace=True, header=None, names=['wav', 'flux'])
                        normalized_spectrum = normalize_spectrum(spectrum_df)
                        binned_spectrum = binspec(normalized_spectrum, 4000, 8000, bin_input)

                        traces.append(go.Scatter(
                            x=binned_spectrum["wav"],
                            y=binned_spectrum["flux"],
                            mode='lines',
                            name='sn_name'
                        ))
                    except Exception as e:
                        print(f"Error processing sn_name file: {e}")

        # Plotting the galaxy spectrum if 'Galaxy' is checked
        if 'gal' in plotting_checklist:
            galaxy_file = selected_row.get('GALAXY')
            const_gal = selected_row.get('CONST_GAL', 1)  # Default to 1 if CONST_GAL is missing
            if pd.notna(galaxy_file):
                galaxy_spectrum_path = os.path.join(BASE_DIR, GAL_DIR_SUFFIX, galaxy_file)
                print(f"Galaxy spectrum path: {galaxy_spectrum_path}")
                print(f"Galaxy constant (CONST_GAL): {const_gal}")

                if os.path.isfile(galaxy_spectrum_path):
                    try:
                        # Read the galaxy spectrum file with no header
                        galaxy_spectrum_df = pd.read_csv(galaxy_spectrum_path, delim_whitespace=True, header=None)
                        # Check the content of the galaxy spectrum file
                        print(f"Galaxy spectrum file content:\n{galaxy_spectrum_df.head()}")
                        # Multiply the second column by CONST_GAL
                        galaxy_spectrum_df.iloc[:, 1] *= const_gal
                        # Use the same column names for consistency
                        galaxy_spectrum_df.columns = ['wav', 'flux']
                        normalized_galaxy_spectrum = normalize_spectrum(galaxy_spectrum_df)
                        binned_galaxy_spectrum = binspec(normalized_galaxy_spectrum, 4000, 8000, bin_input)

                        traces.append(go.Scatter(
                            x=binned_galaxy_spectrum["wav"],
                            y=binned_galaxy_spectrum["flux"],
                            mode='lines',
                            name='Adjusted Galaxy Spectrum: ' + os.path.basename(galaxy_spectrum_path)
                        ))
                    except Exception as e:
                        print(f"Error processing Galaxy file: {e}")

        # Plotting the spectrum from SPECTRUM column if present
        spectrum_file = selected_row.get('SPECTRUM')
        if pd.notna(spectrum_file):
            spectrum_path = os.path.join(BASE_DIR, spectrum_file)
            if os.path.isfile(spectrum_path):
                try:
                    spectrum_df = pd.read_csv(spectrum_path, delim_whitespace=True, header=None, names=['wav', 'flux'])
                    normalized_spectrum = normalize_spectrum(spectrum_df)
                    binned_spectrum = binspec(normalized_spectrum, 4000, 8000, bin_input)

                    traces.append(go.Scatter(
                        x=binned_spectrum["wav"],
                        y=binned_spectrum["flux"],
                        mode='lines',
                        name='SPECTRUM Column Data'
                    ))
                except Exception as e:
                    print(f"Error processing SPECTRUM file: {e}")

    return data, columns, [
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
'''
@callback(
    [Output('datatable-interactivity', 'data'),
     Output('datatable-interactivity', 'columns'),
     Output('datatable-interactivity-container', 'children')],
    [Input('upload-data', 'contents'),
     Input('datatable-interactivity', 'derived_virtual_selected_rows'),
     Input('bin_input', 'value'),
     Input('plotting_checklist', 'value')]
)
def update_graphs(uploaded_contents, derived_virtual_selected_rows, bin_input, plotting_checklist):
    global df

    BASE_DIR = "/home/stiwary/Superfit/NGSF/"
    GAL_DIR_SUFFIX = 'bank/binnings/10A/gal/'  # Directory suffix for galaxy spectra
    expected_columns = [
        'GALAXY', 'SN', 'CONST_SN', 'CONST_GAL', 'Z', 'A_v', 'Phase', 'Band',
        'Frac(SN)', 'Frac(gal)', 'CHI2/dof', 'sn_name'
    ]

    # Read and process uploaded CSV data if available
    if uploaded_contents is not None:
        content_type, content_string = uploaded_contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            csv_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            for col in expected_columns + ['SPECTRUM', 'CHI2/dof2']:
                if col not in csv_df.columns:
                    csv_df[col] = pd.NA

            # Populate the data excluding the 'SPECTRUM' and 'CHI2/dof2' columns for display
            display_df = csv_df[expected_columns]
        except Exception as e:
            print(f"Error reading file: {e}")
            raise PreventUpdate
    else:
        csv_df = pd.DataFrame(columns=expected_columns + ['SPECTRUM', 'CHI2/dof2'])
        display_df = csv_df[expected_columns]

    # Update data table
    columns = [{"name": col, "id": col} for col in display_df.columns]
    data = display_df.to_dict('records')

    # Initialize traces list for plotting
    traces = []

    # Handle selected row for graph generation
    if derived_virtual_selected_rows:
        selection = derived_virtual_selected_rows[0]
        selected_row = csv_df.iloc[selection]

        # Plotting the spectrum from sn_name column if 'Template' is checked
        if 'tem' in plotting_checklist:
            sn_name = selected_row.get('sn_name')
            if pd.notna(sn_name):
                spectrum_path = os.path.join(BASE_DIR, sn_name)
                if os.path.isfile(spectrum_path):
                    try:
                        print(f"Processing SN spectrum file: {spectrum_path}")
                        spectrum_df = pd.read_csv(spectrum_path, delim_whitespace=True, header=None, names=['wav', 'flux'])
                        normalized_spectrum = normalize_spectrum(spectrum_df)
                        binned_spectrum = binspec(normalized_spectrum, 4000, 8000, bin_input)

                        traces.append(go.Scatter(
                            x=binned_spectrum["wav"],
                            y=binned_spectrum["flux"],
                            mode='lines',
                            name='sn_name'
                        ))
                    except Exception as e:
                        print(f"Error processing sn_name file: {e}")

        # Plotting the galaxy spectrum if 'Galaxy' is checked
        if 'gal' in plotting_checklist:
            galaxy_file = selected_row.get('GALAXY')
            const_gal = selected_row.get('CONST_GAL', 1)  # Default to 1 if CONST_GAL is missing
            if pd.notna(galaxy_file):
                galaxy_spectrum_path = os.path.join(BASE_DIR, GAL_DIR_SUFFIX, galaxy_file)
                print(f"Galaxy spectrum path: {galaxy_spectrum_path}")
                print(f"Galaxy constant (CONST_GAL): {const_gal}")

                if os.path.isfile(galaxy_spectrum_path):
                    try:
                        galaxy_spectrum_df = pd.read_csv(galaxy_spectrum_path, delim_whitespace=True, header=None)
                        galaxy_spectrum_df.iloc[:, 1] *= const_gal
                        galaxy_spectrum_df.columns = ['wav', 'flux']
                        normalized_galaxy_spectrum = normalize_spectrum(galaxy_spectrum_df)
                        binned_galaxy_spectrum = binspec(normalized_galaxy_spectrum, 4000, 8000, bin_input)

                        traces.append(go.Scatter(
                            x=binned_galaxy_spectrum["wav"],
                            y=binned_galaxy_spectrum["flux"],
                            mode='lines',
                            name='Adjusted Galaxy Spectrum: ' + os.path.basename(galaxy_spectrum_path)
                        ))
                    except Exception as e:
                        print(f"Error processing Galaxy file: {e}")

        # Plotting the spectrum from SPECTRUM column if present (but not displaying the column)
        spectrum_file = selected_row.get('SPECTRUM')
        if pd.notna(spectrum_file):
            spectrum_path = os.path.join(BASE_DIR, spectrum_file)
            if os.path.isfile(spectrum_path):
                try:
                    spectrum_df = pd.read_csv(spectrum_path, delim_whitespace=True, header=None, names=['wav', 'flux'])
                    normalized_spectrum = normalize_spectrum(spectrum_df)
                    binned_spectrum = binspec(normalized_spectrum, 4000, 8000, bin_input)

                    traces.append(go.Scatter(
                        x=binned_spectrum["wav"],
                        y=binned_spectrum["flux"],
                        mode='lines',
                        name='SPECTRUM Data'
                    ))
                except Exception as e:
                    print(f"Error processing SPECTRUM file: {e}")

        # You can process CHI2/dof2 here if necessary for calculations or other logic

    return data, columns, [
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
