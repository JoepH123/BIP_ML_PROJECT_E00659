import base64
import io
import dash
from dash import dcc, html, dash_table, Input, Output, State
import pandas as pd
import dash_bootstrap_components as dbc
from ui_backend import show_correct_data, run_prediction_process


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])
app.title = "Exemption code predictor"

def create_table(id='output-data-table'):
    table = dbc.Card(
        dash_table.DataTable(
            id=id,
            page_size=10,  # Display 10 rows per page, looks better in the dashboard
            style_table={'overflowY': 'hidden', 'borderRadius': '15px', 'boxShadow': '0 2px 2px 0 rgba(0,0,0,0.2)'},
            style_cell={'padding': '10px', 'textAlign': 'left', 'border': 'none', 'color': "#444444"},
            style_header={
                'backgroundColor': 'rgb(0, 43, 54, 1)',
                'fontWeight': 'bold',
                'borderTopLeftRadius': '15px',
                'borderTopRightRadius': '15px',
                'color': '#e3e3e3',
                'padding': '10px',
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                },             
                {
                    "if": {"state": "selected"},
                    "backgroundColor": None, 
                    "border": "3px solid #b58900",
                },
            ],
        ),
        body=True,
        style={'borderRadius': '15px', 'boxShadow': '0 2px 2px 0 rgba(0,0,0,0.2)', 'marginTop': '20px'}
    )
    return table
  
app.layout = dbc.Container([
    html.Br(),
    html.H1('VAT Exemption Code Predictor', style={'textAlign': 'center', 'marginBottom': '20px', 'color': '#e3e3e3'}),
    html.Br(),
    dbc.Row(
        dbc.Col(
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files (.csv or .xlsx)')
                ]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '10px',
                    'textAlign': 'center', 'margin': '10px', 'color': '#e3e3e3', 'fontWeight': 'bold'
                },
                multiple=False  # Allow only one file to be uploaded
            ),
            width=4
        ),
        justify="center"
    ),
    dbc.Row(
        dbc.Col(
            create_table(id='input-data-table'),
            width=10
        ),
        justify="center"
    ),
    dbc.Row(
        dbc.Col(
            create_table(id='feature-data-table'),
            width=10
        ),
        justify="center"
    ),
    html.Br(),
    dbc.Row(
        dbc.Col(
            html.H3(id='clicked-row', style={"textAlign": "center", "color": "#e3e3e3"}),
            width=12
        ),
        justify="center"
    )
], fluid=True)

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # when user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xlsx' in filename:
            # when user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return html.Div([
                'Wrong file type inserted.'
            ])
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df

@app.callback(Output('input-data-table', 'data'),
              Output('input-data-table', 'columns'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))
def update_input(contents, filename):
    if contents is not None:
        df = parse_contents(contents, filename)
        if not isinstance(df, pd.DataFrame):  # make sure that we obtained a dataframe
            return [], []  # otherwise return empty table
        columns = [{'name': i, 'id': i} for i in df.columns if i != "IvaM"]
        data = df.to_dict('records')
        return data, columns
    return [], []

@app.callback(Output('clicked-row', 'children'),
              Output('feature-data-table', 'data'),    
              Input('input-data-table', 'active_cell'),
              State('input-data-table', 'data'))
def display_click_data(active_cell, rows):
    if active_cell and rows:
        row = rows[active_cell['row']]
        actual_ivam_code = row["IvaM"]
        feature_data = show_correct_data(row)
        predicted_ivam_code = run_prediction_process(feature_data)
        correct = actual_ivam_code == predicted_ivam_code
        correct_text = "correct" if correct else "not correct"
        return f'The predicted code is: {predicted_ivam_code}, and the actual code is: {actual_ivam_code}. The prediction is therefore {correct_text}!', feature_data
    return 'Click a row after uploading a file to see prediction here', [{}]

if __name__ == '__main__':
    app.run_server(debug=True)
