import pandas as pd
import numpy as np
import joblib

from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

# Load trained model and scaler
model = joblib.load("best_random_forest_model.pkl")
scaler = joblib.load("scaler_17cols.save")

# Define expected input fields based on training features
expected_features = model.feature_names_in_  # Requires scikit-learn >= 1.0

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = "Student Grade Prediction"

# Layout
app.layout = dbc.Container([
    html.H1("Predict Student Grade Class"),
    html.Hr(),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.Label(feature.replace('_', ' ').capitalize()),
                dcc.Input(id=feature, type='number', required=True, step=0.01, className="form-control")
            ], className='mb-2') for feature in expected_features
        ], md=6),
    ]),

    html.Br(),

    dbc.Button("Predict Grade Class", id="predict-btn", color="primary"),
    html.Br(), html.Br(),

    dbc.Alert(id='prediction-output', color='info')
])

# Callback
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    [State(feature, 'value') for feature in expected_features]
)
def predict_grade(n_clicks, *values):
    if n_clicks is None:
        return "Enter student data and click Predict."

    if any(v is None for v in values):
        return "Please fill in all input fields."

    try:
        input_df = pd.DataFrame([values], columns=expected_features)
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        return f"Predicted Grade Class: {int(pred)}"
    except Exception as e:
        return f"Prediction failed: {str(e)}"


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
