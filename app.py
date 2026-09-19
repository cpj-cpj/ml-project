
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, cross_val_predict

RANDOM_STATE = 42
DATA_FILE = "master_dataset.csv"

st.set_page_config(page_title="Sydney Housing Price Predictor", page_icon="🏠", layout="centered")

@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_FILE, parse_dates=["sale_date"])
    data = df.copy()

    data["land_size_sqm"] = data.groupby("suburb")["land_size_sqm"].transform(
        lambda s: s.fillna(s.median())
    )
    data["sale_year"] = data["sale_date"].dt.year
    suburb_median_year = data.groupby("suburb")["sale_year"].transform("median")
    data["sale_year"] = data["sale_year"].fillna(suburb_median_year).astype(int)

    for suburb in data["suburb"].dropna().unique():
        col = f"bedrooms_if_{str(suburb).lower()}"
        data[col] = np.where(data["suburb"] == suburb, data["bedrooms"], 0)

    feature_cols = [
        "bedrooms", "bathrooms", "parking", "land_size_sqm",
        "land_size_missing", "sale_year", "sale_date_missing",
        "suburb", "property_type",
    ] + [c for c in data.columns if c.startswith("bedrooms_if_")]

    numeric_features = [
        "bedrooms", "bathrooms", "parking", "land_size_sqm",
        "land_size_missing", "sale_year", "sale_date_missing",
    ] + [c for c in feature_cols if c.startswith("bedrooms_if_")]

    categorical_features = ["suburb", "property_type"]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median"))
        ]), numeric_features),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
        ]), categorical_features)
    ])

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", rf)
    ])

    model = TransformedTargetRegressor(
        regressor=pipe,
        func=np.log1p,
        inverse_func=np.expm1
    )

    X = data[feature_cols]
    y = data["sale_price"]
    model.fit(X, y)

    return model, data, feature_cols

model, training_data, feature_cols = train_model()

st.title("🏠 Sydney Housing Price Prediction")
st.write(
    "Enter the characteristics of a property in Kellyville, Cherrybrook or Blacktown "
    "to obtain an estimated sale price from the Random Forest model developed in this project."
)

with st.form("prediction_form"):
    suburb = st.selectbox("Suburb", ["Kellyville", "Cherrybrook", "Blacktown"])
    property_type = st.selectbox("Property type", ["House", "Townhouse"])
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=4, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=8, value=2, step=1)
    parking = st.number_input("Car spaces", min_value=0, max_value=8, value=2, step=1)
    land_size = st.number_input(
        "Land size (sqm)", min_value=50.0, max_value=5000.0,
        value=650.0, step=10.0
    )
    sale_year = st.number_input(
        "Expected/current sale year", min_value=2023, max_value=2035,
        value=2026, step=1
    )

    submitted = st.form_submit_button("Predict sale price")

if submitted:
    row = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "parking": parking,
        "land_size_sqm": land_size,
        "land_size_missing": False,
        "sale_year": sale_year,
        "sale_date_missing": False,
        "suburb": suburb,
        "property_type": property_type,
    }

    for s in training_data["suburb"].dropna().unique():
        row[f"bedrooms_if_{str(s).lower()}"] = bedrooms if suburb == s else 0

    input_df = pd.DataFrame([row], columns=feature_cols)
    prediction = float(model.predict(input_df)[0])

    st.success(f"Estimated sale price: ${prediction:,.0f}")

    st.info(
        "This is a model-based estimate, not a formal property valuation. "
        "The training data contains only 232 manually collected sales and does not "
        "capture factors such as renovation quality, exact micro-location, views, "
        "property condition, school catchments or negotiation circumstances."
    )

st.divider()
st.caption("Model: Random Forest regression with log-transformed sale-price target and 5-fold CV evaluation.")
