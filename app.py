import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.io import arff
from io import StringIO

st.set_page_config(page_title="AI Financial System", layout="wide")

st.title("AI Financial Analysis Platform")

# ============================================================
# SECTION 1 — BANKRUPTCY PREDICTION
# ============================================================

st.header("🏦 AI Dự báo Phá sản Doanh nghiệp")
st.write("Model sử dụng XGBoost để dự đoán nguy cơ phá sản.")

@st.cache_resource
def load_bankruptcy_artifacts():
    try:
        artifacts = joblib.load('bankruptcy_model.pkl')
        return artifacts
    except:
        return None

data = load_bankruptcy_artifacts()

if data:

    model = data['model']
    imputer = data['imputer']
    scaler = data['scaler']
    feature_names = data['feature_names']

    st.subheader("Upload dữ liệu")

    uploaded_file = st.file_uploader(
        "Upload CSV hoặc Excel",
        type=["csv","xlsx"],
        key="bankruptcy"
    )

    input_df = None

    if uploaded_file:

        if uploaded_file.name.endswith(".csv"):
            input_df = pd.read_csv(uploaded_file)
        else:
            input_df = pd.read_excel(uploaded_file)

        st.success("Upload thành công!")

    if input_df is not None:

        if st.button("Chạy dự báo phá sản"):

            X_input = input_df[feature_names]

            X_filled = imputer.transform(X_input)
            X_scaled = scaler.transform(X_filled)

            pred = model.predict(X_scaled)
            proba = model.predict_proba(X_scaled)[:,1]

            results = input_df.copy()

            results["Prediction"] = [
                "Bankrupt Risk" if p==1 else "Safe"
                for p in pred
            ]

            results["Risk %"] = (proba*100).round(2)

            st.subheader("Kết quả")

            st.dataframe(results)

            # =========================
            # CHART 1 PIE
            # =========================

            st.subheader("Biểu đồ phân bố nguy cơ")

            counts = results["Prediction"].value_counts()

            fig, ax = plt.subplots()
            ax.pie(counts, labels=counts.index, autopct='%1.1f%%')
            ax.set_title("Tỷ lệ doanh nghiệp nguy cơ phá sản")

            st.pyplot(fig)

            # =========================
            # CHART 2 HISTOGRAM
            # =========================

            st.subheader("Phân bố Risk Score")

            fig2, ax2 = plt.subplots()
            ax2.hist(results["Risk %"], bins=20)
            ax2.set_xlabel("Risk %")
            ax2.set_ylabel("Số công ty")

            st.pyplot(fig2)

else:
    st.warning("Không tìm thấy bankruptcy_model.pkl")


# ============================================================
# SECTION 2 — GROWTH PREDICTION
# ============================================================

st.divider()

st.header("📈 AI Dự báo Tăng trưởng Doanh nghiệp")

@st.cache_resource
def load_growth_artifacts():

    try:
        model = joblib.load("model_xgb.pkl")
        scaler = joblib.load("scaler.pkl")
        imputer = joblib.load("imputer.pkl")

        return model, scaler, imputer

    except:
        return None,None,None

growth_model, growth_scaler, growth_imputer = load_growth_artifacts()

def detect_year_prefix(filename):

    match = re.search(r"(\d)year", filename.lower())

    if match:
        return f"y{match.group(1)}__"

    return None


if growth_model:

    st.subheader("Upload dữ liệu nhiều năm")

    growth_files = st.file_uploader(
        "Upload file 1year 2year 3year 4year",
        type=["csv","xlsx","arff"],
        accept_multiple_files=True,
        key="growth"
    )

    growth_df = None

    if growth_files:

        df_dict = {}

        for file in growth_files:

            if file.name.endswith(".csv"):
                df_temp = pd.read_csv(file)

            elif file.name.endswith(".xlsx"):
                df_temp = pd.read_excel(file)

            else:

                content = file.read().decode("utf-8")
                data_raw, meta = arff.loadarff(StringIO(content))
                df_temp = pd.DataFrame(data_raw)

            if "class" in df_temp.columns:
                df_temp = df_temp.drop(columns=["class"])

            prefix = detect_year_prefix(file.name)

            df_temp.columns = [prefix+col for col in df_temp.columns]

            df_dict[prefix] = df_temp

        growth_df = pd.concat(df_dict.values(), axis=1)

        st.success("Merge dữ liệu thành công!")

    if growth_df is not None:

        st.subheader("Preview")

        st.dataframe(growth_df.head())

        if st.button("Chạy dự báo tăng trưởng"):

            df_input = growth_df.copy()

            required_columns = list(growth_imputer.feature_names_in_)

            df_input = df_input[required_columns]

            X = growth_imputer.transform(df_input)
            X = growth_scaler.transform(X)

            pred = growth_model.predict(X)

            results = growth_df.copy()

            results["Growth Prediction"] = pred

            st.subheader("Kết quả")

            st.dataframe(results)

            # =========================
            # BAR CHART
            # =========================

            st.subheader("Biểu đồ tăng trưởng")

            counts = results["Growth Prediction"].value_counts()

            fig3, ax3 = plt.subplots()

            ax3.bar(counts.index.astype(str), counts.values)

            ax3.set_xlabel("Growth class")
            ax3.set_ylabel("Số công ty")

            st.pyplot(fig3)

else:

    st.warning("Không tìm thấy model tăng trưởng")
