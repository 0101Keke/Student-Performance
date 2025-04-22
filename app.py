import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import plotly.graph_objs as go

from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

# Load trained model and scaler
nn_model = tf.keras.models.load_model("student_ffnn_model.keras")
rf_model = joblib.load("best_random_forest_model.pkl")
scaler = joblib.load("scaler_17cols.save")

# Define expected input fields based on training features
expected_features = rf_model.feature_names_in_  # Requires scikit-learn >= 1.0

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

    html.Div([
        html.Label("Choose model:"),
        dcc.Dropdown(
            id="model-selector",
            options=[
                {"label": "Random Forest", "value": "rf"},
                {"label": "Neural Network (TensorFlow)", "value": "nn"}
            ],
            value="rf",
            style={'width': '50%'}
        )
    ]),

    html.Br(),

    dbc.Button("Predict Grade Class", id="predict-btn", color="primary"),
    html.Br(), html.Br(),

    dbc.Alert(id='prediction-output', color='info'),
    dcc.Graph(id='probability-graph')
])

# Callback
@app.callback(
    [Output('prediction-output', 'children'),
    Output('probability-graph', 'figure')],
    Input('predict-btn', 'n_clicks'),
    [State(feature, 'value') for feature in expected_features] + [State("model-selector", "value")]
)
def predict_grade(n_clicks, *args):
    if n_clicks is None:
        return "Enter student data and click Predict.", {}

    values = args[:-1]
    model_type = args[-1]

    if any(v is None for v in values):
        return "Please fill in all input fields.", {}

    input_df = pd.DataFrame([values], columns=expected_features)
    input_scaled = scaler.transform(input_df)

    if model_type == "rf":
        pred = rf_model.predict(input_scaled)[0]
        return f"Predicted Grade Class (RF): {int(pred)}", {}

    elif model_type == "nn":
        probs = nn_model.predict(input_scaled)[0]
        pred_class = np.argmax(probs)

        fig = go.Figure(data=[
            go.Bar(x=[str(i) for i in range(len(probs))], y=probs, marker_color='orange')
        ])
        fig.update_layout(title="Neural Network Prediction Probabilities",
                          xaxis_title="Grade Class", yaxis_title="Probability",
                          template='plotly_white')

        return f"Predicted Grade Class (NN): {pred_class}", fig

    return "Invalid model selection", {}


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
