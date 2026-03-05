import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import re
from scipy.io import arff
from io import StringIO

st.set_page_config(page_title="AI Financial Dashboard", layout="wide")

st.title("AI Financial Analysis Dashboard")

# =====================================================
# LOAD MODELS
# =====================================================

@st.cache_resource
def load_bankruptcy():
    try:
        artifacts = joblib.load("bankruptcy_model.pkl")
        return (
            artifacts["model"],
            artifacts["scaler"],
            artifacts["imputer"],
            artifacts["feature_names"]
        )
    except:
        return None,None,None,None


@st.cache_resource
def load_growth():
    try:
        model = joblib.load("model_xgb.pkl")
        scaler = joblib.load("scaler.pkl")
        imputer = joblib.load("imputer.pkl")
        return model,scaler,imputer
    except:
        return None,None,None


bank_model,bank_scaler,bank_imputer,bank_features = load_bankruptcy()
growth_model,growth_scaler,growth_imputer = load_growth()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Bankruptcy Prediction","Growth Prediction"]
)

# =====================================================
# BANKRUPTCY PAGE (GIỮ NGUYÊN)
# =====================================================

if page == "Bankruptcy Prediction":

    st.header("🏦 Bankruptcy Risk Prediction")

    if bank_model is None:
        st.error("bankruptcy_model.pkl not found")
        st.stop()

    file = st.file_uploader(
        "Upload financial dataset",
        type=["csv","xlsx"]
    )

    if file:

        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        st.subheader("Preview data")
        st.dataframe(df.head())

        if st.button("Run Prediction"):

            try:

                X = df[bank_features]

                X = bank_imputer.transform(X)
                X = bank_scaler.transform(X)

                pred = bank_model.predict(X)
                prob = bank_model.predict_proba(X)[:,1]

                results = df.copy()

                results["Prediction"] = np.where(pred==1,"Bankrupt","Safe")
                results["Risk %"] = (prob*100).round(2)

                col1,col2,col3 = st.columns(3)

                col1.metric("Total Companies",len(results))
                col2.metric("Bankrupt Risk",(pred==1).sum())
                col3.metric("Average Risk %",round(results["Risk %"].mean(),2))

                st.subheader("Bankruptcy Risk Distribution")

                fig = px.bar(
                    results["Prediction"].value_counts(),
                    title="Bankrupt vs Safe"
                )

                st.plotly_chart(fig,use_container_width=True)

                st.subheader("Risk Score Distribution")

                fig2 = px.histogram(
                    results,
                    x="Risk %",
                    nbins=30
                )

                st.plotly_chart(fig2,use_container_width=True)

                st.subheader("Risk Percentage Pie Chart")

                fig3 = px.pie(
                    results,
                    names="Prediction"
                )

                st.plotly_chart(fig3,use_container_width=True)

                st.subheader("Risk % Box Plot")

                fig4 = px.box(
                    results,
                    y="Risk %",
                    color="Prediction"
                )

                st.plotly_chart(fig4,use_container_width=True)

                if hasattr(bank_model,"feature_importances_"):

                    st.subheader("Top Important Features")

                    importance = pd.DataFrame({
                        "feature":bank_features,
                        "importance":bank_model.feature_importances_
                    })

                    importance = importance.sort_values(
                        "importance",
                        ascending=False
                    ).head(20)

                    fig5 = px.bar(
                        importance,
                        x="importance",
                        y="feature",
                        orientation="h"
                    )

                    st.plotly_chart(fig5,use_container_width=True)

                st.subheader("Prediction Results")

                st.dataframe(results)

                csv = results.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Results CSV",
                    csv,
                    "bankruptcy_results.csv",
                    "text/csv"
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")

# =====================================================
# GROWTH PAGE
# =====================================================

if page == "Growth Prediction":

    st.header("📈 Financial Growth Prediction")

    if growth_model is None:
        st.error("Growth model files missing")
        st.stop()

    def detect_year_prefix(filename):

        name = filename.lower()

        if "1year" in name:
            return "y1__"
        elif "2year" in name:
            return "y2__"
        elif "3year" in name:
            return "y3__"
        elif "4year" in name:
            return "y4__"

        return ""

    files = st.file_uploader(
        "Upload 1year 2year 3year 4year datasets",
        type=["csv","xlsx","arff"],
        accept_multiple_files=True
    )

    if files:

        df_list = []

        for file in files:

            try:

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

                df_temp.columns = [f"{prefix}{col}" for col in df_temp.columns]

                df_list.append(df_temp)

            except Exception as e:
                st.error(f"File error: {e}")

        df = pd.concat(df_list,axis=1)

        st.subheader("Merged dataset")
        st.dataframe(df.head())

        if st.button("Run Growth Prediction"):

            try:

                X = df.copy()

                required = list(growth_imputer.feature_names_in_)

                for col in required:
                    if col not in X.columns:
                        X[col] = np.nan

                X = X[required]

                X = growth_imputer.transform(X)
                X = growth_scaler.transform(X)

                pred = growth_model.predict(X)

                results = df.copy()
                results["Growth Prediction"] = pred

                # =========================
                # STATS
                # =========================

                col1,col2,col3 = st.columns(3)

                col1.metric("Average Growth",round(results["Growth Prediction"].mean(),4))
                col2.metric("Max Growth",round(results["Growth Prediction"].max(),4))
                col3.metric("Min Growth",round(results["Growth Prediction"].min(),4))

                # =========================
                # PIE CHART
                # =========================

                results["Growth Level"] = pd.cut(
                    results["Growth Prediction"],
                    bins=[0,0.02,0.04,1],
                    labels=["Low","Medium","High"]
                )

                st.subheader("Growth Level Share")

                fig = px.pie(
                    results,
                    names="Growth Level"
                )

                st.plotly_chart(fig,use_container_width=True)

                # =========================
                # RESULTS
                # =========================

                st.subheader("Prediction Results")

                st.dataframe(results)

                csv = results.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Results CSV",
                    csv,
                    "growth_results.csv"
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")
