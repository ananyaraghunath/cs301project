# ============================================================
# Stage 3 Application: Student Performance Classification App
# Upload CSV, select target, visualize data, train model, predict
# ============================================================

import base64
import io

import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# Change this depending on which model won in Stage 2
# Options: "decision_tree" or "knn"
# ============================================================

BEST_CLASSIFIER = "decision_tree"


# Global storage for trained model
TRAINED_MODEL = {
    "model": None,
    "features": None,
    "target": None,
    "numeric_features": None,
    "categorical_features": None
}


# ============================================================
# Helper Functions
# ============================================================

def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def parse_uploaded_file(contents, filename):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    if filename.endswith(".csv"):
        df = pd.read_csv(
            io.StringIO(decoded.decode("utf-8")),
            sep=None,
            engine="python"
        )
    else:
        raise ValueError("Please upload a CSV file.")

    # Automatically create classification target for Student Performance dataset
    if "G3" in df.columns and "pass_fail" not in df.columns:
        df["pass_fail"] = df["G3"].apply(lambda grade: 1 if grade >= 10 else 0)

    return df


def dataframe_from_store(data):
    if data is None:
        return None
    return pd.read_json(io.StringIO(data), orient="split")


def empty_figure(message):
    fig = px.scatter()
    fig.update_layout(
        title=message,
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 16}
            }
        ]
    )
    return fig


def get_task_type(y):
    if y.dtype == "object" or y.dtype == "bool":
        return "classification"

    if y.nunique() <= 10:
        return "classification"

    return "regression"


def get_model_and_params():
    if BEST_CLASSIFIER == "knn":
        model = KNeighborsClassifier()

        params = {
            "model__n_neighbors": [3, 5, 7, 9, 11],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan"]
        }

        return model, params

    else:
        model = DecisionTreeClassifier(random_state=42)

        params = {
            "model__criterion": ["gini", "entropy"],
            "model__max_depth": [2, 3, 4, 5, 6, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4]
        }

        return model, params


def get_f1_average(y):
    unique_values = set(pd.Series(y).dropna().unique())

    if unique_values == {0, 1} or unique_values == {0.0, 1.0}:
        return "binary"

    return "weighted"


# ============================================================
# Dash App Setup
# ============================================================

app = Dash(__name__)
server = app.server


app.layout = html.Div(
    style={
        "fontFamily": "Arial",
        "margin": "0px",
        "padding": "0px",
        "backgroundColor": "white"
    },
    children=[

        dcc.Store(id="data-store"),

        # ----------------------------------------------------
        # Upload Section
        # ----------------------------------------------------
        html.Div(
            style={
                "backgroundColor": "#d9d9d9",
                "textAlign": "center",
                "padding": "18px",
                "fontWeight": "bold",
                "fontSize": "18px"
            },
            children=[
                dcc.Upload(
                    id="upload-data",
                    children=html.Div("Upload File"),
                    multiple=False,
                    style={"cursor": "pointer"}
                ),
                html.Div(id="upload-status", style={"fontSize": "14px", "marginTop": "8px"})
            ]
        ),

        # ----------------------------------------------------
        # Target Selection Section
        # ----------------------------------------------------
        html.Div(
            style={
                "backgroundColor": "#eeeeee",
                "textAlign": "center",
                "padding": "12px",
                "fontSize": "16px",
                "fontWeight": "bold"
            },
            children=[
                html.Label("Select Target: "),
                dcc.Dropdown(
                    id="target-dropdown",
                    placeholder="Select target",
                    style={
                        "width": "250px",
                        "display": "inline-block",
                        "verticalAlign": "middle",
                        "fontWeight": "normal"
                    }
                )
            ]
        ),

        # ----------------------------------------------------
        # Charts Section
        # ----------------------------------------------------
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "25px",
                "padding": "25px"
            },
            children=[

                html.Div(
                    style={
                        "border": "1px solid #eeeeee",
                        "padding": "15px",
                        "minHeight": "400px"
                    },
                    children=[
                        dcc.RadioItems(
                            id="categorical-radio",
                            inline=True,
                            style={"textAlign": "center", "marginBottom": "10px"}
                        ),
                        dcc.Graph(id="category-chart")
                    ]
                ),

                html.Div(
                    style={
                        "border": "1px solid #eeeeee",
                        "padding": "15px",
                        "minHeight": "400px"
                    },
                    children=[
                        dcc.Graph(id="correlation-chart")
                    ]
                )
            ]
        ),

        # ----------------------------------------------------
        # Feature Selection and Training Section
        # ----------------------------------------------------
        html.Div(
            style={
                "border": "1px solid #eeeeee",
                "margin": "20px",
                "padding": "20px",
                "textAlign": "center"
            },
            children=[
                html.H3("Select Features for Training"),

                dcc.Checklist(
                    id="feature-checklist",
                    inline=True,
                    style={
                        "textAlign": "center",
                        "marginBottom": "20px"
                    }
                ),

                html.Button(
                    "Train",
                    id="train-button",
                    n_clicks=0,
                    style={
                        "width": "220px",
                        "height": "35px",
                        "fontSize": "16px"
                    }
                ),

                html.Div(
                    id="training-output",
                    style={
                        "marginTop": "20px",
                        "fontWeight": "bold",
                        "fontSize": "16px"
                    }
                )
            ]
        ),

        # ----------------------------------------------------
        # Prediction Section
        # ----------------------------------------------------
        html.Div(
            style={
                "textAlign": "center",
                "padding": "25px"
            },
            children=[
                dcc.Input(
                    id="prediction-input",
                    type="text",
                    placeholder="Enter feature values separated by commas",
                    style={
                        "width": "450px",
                        "height": "30px",
                        "fontSize": "14px"
                    }
                ),

                html.Button(
                    "Predict",
                    id="predict-button",
                    n_clicks=0,
                    style={
                        "height": "35px",
                        "fontSize": "14px",
                        "marginLeft": "10px"
                    }
                ),

                html.Span(
                    id="prediction-output",
                    style={
                        "marginLeft": "15px",
                        "fontWeight": "bold",
                        "fontSize": "16px"
                    }
                )
            ]
        )
    ]
)


# ============================================================
# Callback: Upload File
# ============================================================

@app.callback(
    Output("data-store", "data"),
    Output("upload-status", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename")
)
def upload_file(contents, filename):
    if contents is None:
        return None, "Upload a CSV file to begin."

    try:
        df = parse_uploaded_file(contents, filename)
        return df.to_json(orient="split"), f"Uploaded: {filename} | Shape: {df.shape}"

    except Exception as e:
        return None, f"Error uploading file: {str(e)}"


# ============================================================
# Callback: Populate Target Dropdown
# ============================================================

@app.callback(
    Output("target-dropdown", "options"),
    Output("target-dropdown", "value"),
    Input("data-store", "data")
)
def update_target_dropdown(data):
    df = dataframe_from_store(data)

    if df is None:
        return [], None

    options = [{"label": col, "value": col} for col in df.columns]

    if "pass_fail" in df.columns:
        default_target = "pass_fail"
    else:
        default_target = df.columns[-1]

    return options, default_target


# ============================================================
# Callback: Update Feature Checklist and Categorical Radio Buttons
# ============================================================

@app.callback(
    Output("categorical-radio", "options"),
    Output("categorical-radio", "value"),
    Output("feature-checklist", "options"),
    Output("feature-checklist", "value"),
    Input("data-store", "data"),
    Input("target-dropdown", "value")
)
def update_feature_controls(data, target):
    df = dataframe_from_store(data)

    if df is None or target is None:
        return [], None, [], []

    categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if target in categorical_columns:
        categorical_columns.remove(target)

    categorical_options = [{"label": col, "value": col} for col in categorical_columns]
    categorical_value = categorical_columns[0] if len(categorical_columns) > 0 else None

    feature_columns = [col for col in df.columns if col != target]

    # Avoid data leakage because pass_fail is created from G3
    if target == "pass_fail" and "G3" in feature_columns:
        feature_columns.remove("G3")

    feature_options = [{"label": col, "value": col} for col in feature_columns]

    return categorical_options, categorical_value, feature_options, feature_columns


# ============================================================
# Callback: Update Charts
# ============================================================

@app.callback(
    Output("category-chart", "figure"),
    Output("correlation-chart", "figure"),
    Input("data-store", "data"),
    Input("target-dropdown", "value"),
    Input("categorical-radio", "value")
)
def update_charts(data, target, categorical_column):
    df = dataframe_from_store(data)

    if df is None or target is None:
        return empty_figure("Upload data first"), empty_figure("Upload data first")

    # -------------------------------
    # Left chart: class distribution or average target by category
    # -------------------------------

    if categorical_column is None:
        category_fig = empty_figure("No categorical variable available")
    else:
        task_type = get_task_type(df[target])

        if task_type == "classification":
            chart_df = (
                df.groupby([categorical_column, target])
                .size()
                .reset_index(name="count")
            )

            chart_df[target] = chart_df[target].astype(str)

            category_fig = px.bar(
                chart_df,
                x=categorical_column,
                y="count",
                color=target,
                barmode="group",
                title=f"Class Distribution of {target} by {categorical_column}"
            )

        else:
            chart_df = (
                df.groupby(categorical_column)[target]
                .mean()
                .reset_index()
            )

            category_fig = px.bar(
                chart_df,
                x=categorical_column,
                y=target,
                title=f"Average {target} by {categorical_column}",
                text=target
            )

            category_fig.update_traces(texttemplate="%{text:.2f}", textposition="inside")

    # -------------------------------
    # Right chart: correlation strength
    # -------------------------------

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if target in numeric_columns:
        numeric_columns.remove(target)

    # Avoid data leakage
    if target == "pass_fail" and "G3" in numeric_columns:
        numeric_columns.remove("G3")

    if len(numeric_columns) == 0:
        corr_fig = empty_figure("No numerical variables available")
    else:
        if pd.api.types.is_numeric_dtype(df[target]):
            target_values = df[target]
        else:
            target_values = pd.factorize(df[target])[0]

        correlations = []

        for col in numeric_columns:
            corr_value = abs(pd.Series(df[col]).corr(pd.Series(target_values)))

            if not np.isnan(corr_value):
                correlations.append({
                    "Variable": col,
                    "Correlation Strength": corr_value
                })

        corr_df = pd.DataFrame(correlations)

        if corr_df.empty:
            corr_fig = empty_figure("No valid correlations available")
        else:
            corr_df = corr_df.sort_values("Correlation Strength", ascending=False)

            corr_fig = px.bar(
                corr_df,
                x="Variable",
                y="Correlation Strength",
                title=f"Correlation Strength of Numerical Variables with {target}",
                text="Correlation Strength"
            )

            corr_fig.update_traces(texttemplate="%{text:.2f}", textposition="inside")

    return category_fig, corr_fig


# ============================================================
# Callback: Update Prediction Placeholder
# ============================================================

@app.callback(
    Output("prediction-input", "placeholder"),
    Input("feature-checklist", "value")
)
def update_prediction_placeholder(selected_features):
    if selected_features is None or len(selected_features) == 0:
        return "Select features first"

    return "Enter values in this order: " + ", ".join(selected_features)


# ============================================================
# Callback: Train Model
# ============================================================

@app.callback(
    Output("training-output", "children"),
    Input("train-button", "n_clicks"),
    State("data-store", "data"),
    State("target-dropdown", "value"),
    State("feature-checklist", "value")
)
def train_model(n_clicks, data, target, selected_features):
    if n_clicks == 0:
        return ""

    df = dataframe_from_store(data)

    if df is None:
        return "Please upload a dataset first."

    if target is None:
        return "Please select a target variable."

    if selected_features is None or len(selected_features) == 0:
        return "Please select at least one feature."

    # Avoid data leakage
    if target == "pass_fail" and "G3" in selected_features:
        selected_features.remove("G3")

    model_df = df[selected_features + [target]].copy()
    model_df = model_df.dropna(subset=[target])

    X = model_df[selected_features]
    y = model_df[target]

    task_type = get_task_type(y)

    if task_type != "classification":
        return "This app section is set up for classification. Please select a categorical target like pass_fail."

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder())
        ]
    )

    transformers = []

    if len(numeric_features) > 0:
        transformers.append(("num", numeric_transformer, numeric_features))

    if len(categorical_features) > 0:
        transformers.append(("cat", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    # Use the best model from Stage 2.
    # Change this if KNN won in your Stage 2 results.
    classifier = DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", classifier)
        ]
    )

    stratify_value = y if y.value_counts().min() >= 2 else None

    X_train_app, X_test_app, y_train_app, y_test_app = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_value
    )

    pipeline.fit(X_train_app, y_train_app)

    y_pred = pipeline.predict(X_test_app)

    accuracy = accuracy_score(y_test_app, y_pred)
    f1_average = get_f1_average(y_test_app)
    f1 = f1_score(y_test_app, y_pred, average=f1_average)

    TRAINED_MODEL["model"] = pipeline
    TRAINED_MODEL["features"] = selected_features
    TRAINED_MODEL["target"] = target
    TRAINED_MODEL["numeric_features"] = numeric_features
    TRAINED_MODEL["categorical_features"] = categorical_features

    return html.Div([
        html.Div("Best Model Used: Decision Tree Classifier"),
        html.Div(f"Accuracy: {accuracy:.3f}"),
        html.Div(f"F1-score: {f1:.3f}")
    ])


# ============================================================
# Callback: Predict
# ============================================================

@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    State("prediction-input", "value")
)
def make_prediction(n_clicks, input_values):
    if n_clicks == 0:
        return ""

    if TRAINED_MODEL["model"] is None:
        return "Train the model first."

    if input_values is None or input_values.strip() == "":
        return "Enter feature values first."

    features = TRAINED_MODEL["features"]
    numeric_features = TRAINED_MODEL["numeric_features"]

    values = [value.strip() for value in input_values.split(",")]

    if len(values) != len(features):
        return f"Input error: expected {len(features)} values, but received {len(values)}."

    row = {}

    try:
        for feature, value in zip(features, values):
            if feature in numeric_features:
                row[feature] = float(value)
            else:
                row[feature] = value

        input_df = pd.DataFrame([row])

        prediction = TRAINED_MODEL["model"].predict(input_df)[0]

        if TRAINED_MODEL["target"] == "pass_fail":
            label = "Pass" if int(prediction) == 1 else "Fail"
            return f"Predicted class: {label} ({prediction})"

        return f"Predicted class: {prediction}"

    except Exception as e:
        return f"Prediction error: {str(e)}"


# ============================================================
# Run App
# ============================================================

if __name__ == "__main__":
    app.run_server(debug=True)