#!/usr/bin/env python3
"""
Simple dashboard for viewing test results.
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import json
import os
from pathlib import Path
import base64
import socket

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Instrument Test Results Dashboard"

# Define the layout
app.layout = html.Div([
    html.H1("🔬 Instrument Test Results Dashboard", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
    
    # Test selection dropdown
    html.Div([
        html.Label("Select Test Type:", style={'fontSize': 18, 'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='test-selector',
            options=[
                {'label': 'Burn-In Test', 'value': 'burnin'},
                {'label': 'HiPot Test', 'value': 'hipot'},
                {'label': 'Isolation Test', 'value': 'isolation'},
                {'label': 'Laser Test', 'value': 'laser'},
                {'label': 'Parametric Test', 'value': 'parametric'},
                {'label': 'ICT Test', 'value': 'ict'}
            ],
            value='burnin',
            style={'width': '50%', 'marginBottom': 20}
        )
    ], style={'textAlign': 'center', 'marginBottom': 30}),
    
    # Summary statistics
    html.Div(id='summary-stats', style={'marginBottom': 30}),
    
    # Tabs for different views
    dcc.Tabs([
        dcc.Tab(label='📊 Summary Statistics', children=[
            html.Div(id='stats-content')
        ]),
        dcc.Tab(label='📈 Time Series Plots', children=[
            html.Div(id='timeseries-content')
        ]),
        dcc.Tab(label='📉 SPC Charts', children=[
            html.Div(id='spc-content')
        ]),
        dcc.Tab(label='📋 Raw Data', children=[
            html.Div(id='raw-data-content')
        ]),
        dcc.Tab(label='🖼️ Generated Images', children=[
            html.Div(id='images-content')
        ])
    ]),
    
    # Footer
    html.Div([
        html.Hr(),
        html.P("Instrument Data Pipeline Dashboard", 
               style={'textAlign': 'center', 'color': '#7f8c8d'})
    ], style={'marginTop': 50})
])

def get_unit_for_column(col_name, test_name):
    """Get the appropriate unit for a given column name and test type."""
    column_unit_mapping = {
        'burnin': {
            'temperature': '°C',
            'voltage': 'V',
            'current': 'A'
        },
        'hipot': {
            'voltage': 'kV',
            'current': 'mA'
        },
        'isolation': {
            'resistance': 'MΩ'
        },
        'laser': {
            'power': 'mW',
            'wavelength': 'nm'
        },
        'parametric': {
            'voltage': 'mV',
            'current': 'mA',
            'power': 'mW'
        },
        'ict': {
            'resistance': 'Ω',
            'value': 'V',  # Default for ICT values
            'continuity_stats': 'Ω',
            'resistor_stats': 'Ω',
            'capacitor_stats': 'μF',
            'power_stats': 'V'
        }
    }
    
    return column_unit_mapping.get(test_name, {}).get(col_name, '')

def get_unit_for_statistic(stat_key, test_name):
    """Get the appropriate unit for a given statistic and test type."""
    unit_mapping = {
        'burnin': {
            'temperature_stats': '°C',
            'voltage_stats': 'V',
            'current_stats': 'A'
        },
        'hipot': {
            'voltage_stats': 'kV',
            'current_stats': 'mA'
        },
        'isolation': {
            'resistance_stats': 'MΩ'
        },
        'laser': {
            'power_stats': 'mW',
            'wavelength_stats': 'nm'
        },
        'parametric': {
            'voltage_stats': 'mV',
            'current_stats': 'mA',
            'power_stats': 'mW'
        },
        'ict': {
            'continuity_stats': 'Ω',
            'resistor_stats': 'Ω',
            'capacitor_stats': 'μF',
            'power_stats': 'V'
        }
    }
    
    return unit_mapping.get(test_name, {}).get(stat_key, '')

def get_latest_files(test_name):
    """Get the latest files for a given test."""
    test_dir = Path(f"reports/{test_name}")
    if not test_dir.exists():
        return None, None, None
    
    # Find latest files
    json_files = list(test_dir.glob("*.json"))
    csv_files = list(test_dir.glob("*.csv"))
    png_files = list(test_dir.glob("*.png"))
    
    latest_json = max(json_files, key=lambda x: x.stat().st_mtime) if json_files else None
    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime) if csv_files else None
    latest_png = max(png_files, key=lambda x: x.stat().st_mtime) if png_files else None
    
    return latest_json, latest_csv, latest_png

@app.callback(
    Output('summary-stats', 'children'),
    Input('test-selector', 'value')
)
def update_summary_stats(selected_test):
    """Update summary statistics."""
    if not selected_test:
        return html.Div("No test selected")
    
    json_file, _, _ = get_latest_files(selected_test)
    if not json_file:
        return html.Div([
            html.H3(f"❌ No results found for {selected_test.upper()} test"),
            html.P("Run the test first using: python -m etl.simulations.{selected_test}_simulation")
        ])
    
    try:
        with open(json_file, 'r') as f:
            stats = json.load(f)
        
        # Create summary cards
        cards = []
        
        # Pass/Fail rate card
        if 'failure_rate' in stats:
            rate = stats['failure_rate'] * 100
            color = 'red' if rate > 5 else 'green'
            cards.append(html.Div([
                html.H4("Failure Rate"),
                html.H2(f"{rate:.2f}%", style={'color': color})
            ], style={'background': 'white', 'border': '1px solid #ddd', 'borderRadius': '8px', 
                      'padding': '20px', 'margin': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 
                      'minWidth': '200px', 'textAlign': 'center'}))
        elif 'pass_rate' in stats:
            rate = stats['pass_rate'] * 100
            color = 'green' if rate > 95 else 'orange'
            cards.append(html.Div([
                html.H4("Pass Rate"),
                html.H2(f"{rate:.2f}%", style={'color': color})
            ], style={'background': 'white', 'border': '1px solid #ddd', 'borderRadius': '8px', 
                      'padding': '20px', 'margin': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 
                      'minWidth': '200px', 'textAlign': 'center'}))
        elif 'overall_pass_rate' in stats:
            rate = stats['overall_pass_rate'] * 100
            color = 'green' if rate > 95 else 'orange'
            cards.append(html.Div([
                html.H4("Overall Pass Rate"),
                html.H2(f"{rate:.2f}%", style={'color': color})
            ], style={'background': 'white', 'border': '1px solid #ddd', 'borderRadius': '8px', 
                      'padding': '20px', 'margin': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 
                      'minWidth': '200px', 'textAlign': 'center'}))
        
        # Add statistics cards
        for key, value in stats.items():
            if isinstance(value, dict) and 'mean' in value:
                # Add appropriate units based on the key and test type
                unit = get_unit_for_statistic(key, selected_test)
                
                cards.append(html.Div([
                    html.H4(key.replace('_', ' ').title()),
                    html.P(f"Mean: {value['mean']:.2f} {unit}"),
                    html.P(f"Std: {value['std']:.2f} {unit}"),
                    html.P(f"Range: {value['min']:.2f} - {value['max']:.2f} {unit}")
                ], style={'background': 'white', 'border': '1px solid #ddd', 'borderRadius': '8px', 
                          'padding': '20px', 'margin': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 
                          'minWidth': '200px', 'textAlign': 'center'}))
        
        return html.Div([
            html.H3(f"📊 {selected_test.upper()} Test Summary"),
            html.Div(cards, style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px'})
        ])
        
    except Exception as e:
        return html.Div(f"Error loading statistics: {str(e)}")

@app.callback(
    Output('stats-content', 'children'),
    Input('test-selector', 'value')
)
def update_stats_content(selected_test):
    """Update detailed statistics content."""
    if not selected_test:
        return html.Div("No test selected")
    
    json_file, _, _ = get_latest_files(selected_test)
    if not json_file:
        return html.Div("No results found")
    
    try:
        with open(json_file, 'r') as f:
            stats = json.load(f)
        
        # Create detailed statistics table
        rows = []
        for key, value in stats.items():
            if isinstance(value, dict) and 'mean' in value:
                unit = get_unit_for_statistic(key, selected_test)
                rows.append(html.Tr([
                    html.Td(key.replace('_', ' ').title()),
                    html.Td(f"{value['mean']:.2f} {unit}"),
                    html.Td(f"{value['std']:.2f} {unit}"),
                    html.Td(f"{value['min']:.2f} {unit}"),
                    html.Td(f"{value['max']:.2f} {unit}")
                ]))
        
        return html.Div([
            html.H4("Detailed Statistics"),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Parameter"),
                    html.Th("Mean"),
                    html.Th("Std Dev"),
                    html.Th("Min"),
                    html.Th("Max")
                ])),
                html.Tbody(rows)
            ], style={'width': '100%', 'borderCollapse': 'collapse'})
        ])
        
    except Exception as e:
        return html.Div(f"Error: {str(e)}")

@app.callback(
    Output('timeseries-content', 'children'),
    Input('test-selector', 'value')
)
def update_timeseries_content(selected_test):
    """Update time series plots."""
    if not selected_test:
        return html.Div("No test selected")
    
    _, csv_file, _ = get_latest_files(selected_test)
    if not csv_file:
        return html.Div("No data found")
    
    try:
        df = pd.read_csv(csv_file)
        
        # Create time series plots based on available columns
        plots = []
        
        # Find numeric columns (excluding timestamp/sequence_num)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if 'timestamp' in numeric_cols:
            numeric_cols.remove('timestamp')
        if 'sequence_num' in numeric_cols:
            numeric_cols.remove('sequence_num')
        
        # Determine x-axis column based on test type
        if 'timestamp' in df.columns:
            x_col = 'timestamp'
        elif 'sequence_num' in df.columns:
            x_col = 'sequence_num'
        else:
            x_col = df.index
        
        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            # Get unit for the column
            unit = get_unit_for_column(col, selected_test)
            fig = px.line(df, x=x_col, y=col, title=f'{col.title()} Over Time ({unit})')
            fig.update_layout(height=400)
            plots.append(dcc.Graph(figure=fig))
        
        return html.Div(plots)
        
    except Exception as e:
        return html.Div(f"Error creating plots: {str(e)}")

@app.callback(
    Output('raw-data-content', 'children'),
    Input('test-selector', 'value')
)
def update_raw_data_content(selected_test):
    """Update raw data table."""
    if not selected_test:
        return html.Div("No test selected")
    
    _, csv_file, _ = get_latest_files(selected_test)
    if not csv_file:
        return html.Div("No data found")
    
    try:
        df = pd.read_csv(csv_file)
        
        return html.Div([
            html.H4(f"Raw Data ({len(df)} rows)"),
            html.Div([
                html.P(f"Shape: {df.shape}"),
                html.P(f"Columns: {', '.join(df.columns)}")
            ]),
            dcc.Graph(
                figure=go.Figure(data=[go.Table(
                    header=dict(values=list(df.columns)),
                    cells=dict(values=[df[col] for col in df.columns])
                )])
            )
        ])
        
    except Exception as e:
        return html.Div(f"Error loading data: {str(e)}")

@app.callback(
    Output('images-content', 'children'),
    Input('test-selector', 'value')
)
def update_images_content(selected_test):
    """Update generated images."""
    if not selected_test:
        return html.Div("No test selected")
    
    _, _, png_file = get_latest_files(selected_test)
    if not png_file:
        return html.Div("No images found")
    
    try:
        # Encode image to base64
        with open(png_file, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode()
        
        return html.Div([
            html.H4("Generated Plot"),
            html.Img(src=f'data:image/png;base64,{encoded_image}', 
                    style={'width': '100%', 'maxWidth': '800px'})
        ])
        
    except Exception as e:
        return html.Div(f"Error loading image: {str(e)}")

@app.callback(
    Output('spc-content', 'children'),
    Input('test-selector', 'value')
)
def update_spc_content(selected_test):
    """Update SPC charts."""
    if not selected_test:
        return html.Div("No test selected")
    
    _, csv_file, _ = get_latest_files(selected_test)
    if not csv_file:
        return html.Div("No data found")
    
    try:
        df = pd.read_csv(csv_file)
        
        # Create SPC charts for numeric columns
        plots = []
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if 'timestamp' in numeric_cols:
            numeric_cols.remove('timestamp')
        if 'sequence_num' in numeric_cols:
            numeric_cols.remove('sequence_num')
        
        # Determine x-axis column based on test type
        if 'timestamp' in df.columns:
            x_col = 'timestamp'
        elif 'sequence_num' in df.columns:
            x_col = 'sequence_num'
        else:
            x_col = df.index
        
        for col in numeric_cols[:2]:  # Limit to first 2 columns
            mean_val = df[col].mean()
            std_val = df[col].std()
            ucl = mean_val + 3 * std_val
            lcl = mean_val - 3 * std_val
            
            # Get unit for the column
            unit = get_unit_for_column(col, selected_test)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[x_col], y=df[col], mode='lines', name=col))
            fig.add_hline(y=mean_val, line_dash="dash", line_color="green", name="Mean")
            fig.add_hline(y=ucl, line_dash="dash", line_color="red", name="UCL")
            fig.add_hline(y=lcl, line_dash="dash", line_color="red", name="LCL")
            fig.update_layout(title=f'{col.title()} SPC Chart ({unit})', height=400)
            plots.append(dcc.Graph(figure=fig))
        
        return html.Div(plots)
        
    except Exception as e:
        return html.Div(f"Error creating SPC charts: {str(e)}")

def find_available_port(start_port=8050, max_attempts=10):
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return None

if __name__ == '__main__':
    port = find_available_port()
    if port is None:
        print("❌ No available ports found in range 8050-8059")
        exit(1)
    
    print(f"🚀 Starting dashboard on port {port}")
    print(f"📊 Open your browser to: http://localhost:{port}")
    
    app.run(debug=True, host='0.0.0.0', port=port) 