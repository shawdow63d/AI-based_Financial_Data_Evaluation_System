import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import shap
import re

from sklearn.metrics import roc_curve, auc
from scipy.io import arff
from io import StringIO

st.set_page_config(page_title="AI Financial Dashboard", layout="wide")

st.title("AI Financial Analysis Dashboard")

# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_bankruptcy():

    artifacts = joblib.load("bankruptcy_model.pkl")

    return (
        artifacts["model"],
        artifacts["scaler"],
        artifacts["imputer"],
        artifacts["feature_names"]
    )

@st.cache_resource
def load_growth():

    model = joblib.load("model_xgb.pkl")
    scaler = joblib.load("scaler.pkl")
    imputer = joblib.load("imputer.pkl")

    return model, scaler, imputer

bank_model, bank_scaler, bank_imputer, bank_features = load_bankruptcy()
growth_model, growth_scaler, growth_imputer = load_growth()

# =========================================================
# BANKRUPTCY PREDICTION
# =========================================================

st.header("🏦 Bankruptcy Prediction")

file = st.file_uploader(
    "Upload financial dataset",
    type=["csv","xlsx"],
    key="bankruptcy"
)

if file:

    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.write("Preview Data")
    st.dataframe(df.head())

    if st.button("Run Bankruptcy Model"):

        X = df[bank_features]

        X = bank_imputer.transform(X)
        X = bank_scaler.transform(X)

        pred = bank_model.predict(X)
        prob = bank_model.predict_proba(X)[:,1]

        results = df.copy()

        results["Prediction"] = np.where(pred==1,"Bankrupt","Safe")
        results["Risk %"] = (prob*100).round(2)

        # =========================
        # METRICS
        # =========================

        col1,col2,col3 = st.columns(3)

        col1.metric("Total Companies",len(results))
        col2.metric("Bankrupt Risk",(pred==1).sum())
        col3.metric("Average Risk %",round(results["Risk %"].mean(),2))

        # =========================
        # BAR CHART
        # =========================

        st.subheader("Risk Distribution")

        fig = px.bar(
            results["Prediction"].value_counts(),
            title="Bankruptcy Risk Distribution"
        )

        st.plotly_chart(fig)

        # =========================
        # HISTOGRAM
        # =========================

        st.subheader("Risk Score Distribution")

        fig2 = px.histogram(results,x="Risk %",nbins=30)

        st.plotly_chart(fig2)

        # =========================
        # FEATURE IMPORTANCE
        # =========================

        st.subheader("Feature Importance")

        if hasattr(bank_model,"feature_importances_"):

            importance = pd.DataFrame({

                "feature": bank_features,
                "importance": bank_model.feature_importances_

            })

            importance = importance.sort_values(
                "importance",
                ascending=False
            ).head(20)

            fig3 = px.bar(
                importance,
                x="importance",
                y="feature",
                orientation="h"
            )

            st.plotly_chart(fig3)

        # =========================
        # SHAP EXPLAINABILITY
        # =========================

        st.subheader("Model Explainability (SHAP)")

        explainer = shap.TreeExplainer(bank_model)

        shap_values = explainer.shap_values(X)

        shap_df = pd.DataFrame(
            np.abs(shap_values).mean(0),
            index=bank_features,
            columns=["SHAP"]
        ).sort_values("SHAP",ascending=False).head(20)

        fig4 = px.bar(
            shap_df,
            x="SHAP",
            y=shap_df.index,
            orientation="h"
        )

        st.plotly_chart(fig4)

        # =========================
        # RESULTS TABLE
        # =========================

        st.subheader("Prediction Results")

        st.dataframe(results)

# =========================================================
# GROWTH MODEL
# =========================================================

st.divider()

st.header("📈 Growth Prediction")

def detect_year_prefix(filename):

    match = re.search(r"(\d)year",filename.lower())

    if match:
        return f"y{match.group(1)}__"

    return None

files = st.file_uploader(
    "Upload 1year 2year 3year 4year",
    type=["csv","xlsx","arff"],
    accept_multiple_files=True,
    key="growth"
)

if files:

    df_dict = {}

    for file in files:

        if file.name.endswith(".csv"):
            df_temp = pd.read_csv(file)

        elif file.name.endswith(".xlsx"):
            df_temp = pd.read_excel(file)

        else:

            content = file.read().decode("utf-8")

            data_raw,meta = arff.loadarff(StringIO(content))

            df_temp = pd.DataFrame(data_raw)

        if "class" in df_temp.columns:
            df_temp = df_temp.drop(columns=["class"])

        prefix = detect_year_prefix(file.name)

        df_temp.columns = [prefix + col for col in df_temp.columns]

        df_dict[prefix] = df_temp

    df = pd.concat(df_dict.values(),axis=1)

    st.write("Merged Dataset")

    st.dataframe(df.head())

    if st.button("Run Growth Model"):

        X = df.copy()

        required = list(growth_imputer.feature_names_in_)

        X = X[required]

        X = growth_imputer.transform(X)
        X = growth_scaler.transform(X)

        pred = growth_model.predict(X)

        results = df.copy()

        results["Growth Prediction"] = pred

        st.subheader("Growth Distribution")

        fig = px.bar(
            results["Growth Prediction"].value_counts()
        )

        st.plotly_chart(fig)

        st.dataframe(results)

        csv = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Results",
            csv,
            "growth_predictions.csv"
        )
