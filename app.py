import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.io import arff

st.title("📊 AI-Based Financial Data Evaluation System")

# Load model
model = joblib.load("model.pkl")

# Upload file
uploaded_file = st.file_uploader("Upload ARFF file", type=["arff"])

if uploaded_file is not None:

    data, meta = arff.loadarff(uploaded_file)
    df = pd.DataFrame(data)

    # convert bytes -> string
    for col in df.select_dtypes([object]):
        df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

    # =============================
    # Remove class column for display
    # =============================

    if "class" in df.columns:
        df_display = df.drop(columns=["class"])
    else:
        df_display = df.copy()

    st.subheader("📄 Dataset Preview")
    st.write(df_display.head())

    numeric_cols = df_display.select_dtypes(include=["float64", "int64"]).columns

    # =============================
    # Histogram
    # =============================

    st.subheader("📈 Histogram")

    column = st.selectbox("Select column", numeric_cols)

    fig, ax = plt.subplots()
    ax.hist(df_display[column], bins=20)
    ax.set_title(f"Histogram of {column}")

    st.pyplot(fig)

    # =============================
    # Boxplot
    # =============================

    st.subheader("📦 Boxplot")

    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df_display[numeric_cols], ax=ax2)

    st.pyplot(fig2)

    # =============================
    # Pie Chart
    # =============================

    st.subheader("🥧 Pie Chart")

    pie_column = st.selectbox("Select column for Pie", numeric_cols)

    pie_data = pd.cut(df_display[pie_column], bins=5).value_counts()

    fig3, ax3 = plt.subplots()
    ax3.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%")

    st.pyplot(fig3)

    # =============================
    # Correlation Heatmap
    # =============================

    st.subheader("🔥 Correlation Heatmap")

    corr = df_display[numeric_cols].corr()

    fig4, ax4 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax4)

    st.pyplot(fig4)

    # =============================
    # Prediction
    # =============================

    st.subheader("🤖 Bankruptcy Prediction")

    predictions = model.predict(df_display)

    result_df = df_display.copy()
    result_df["Prediction"] = predictions

    st.write(result_df.head())

    # =============================
    # Bankruptcy Distribution
    # =============================

    st.subheader("📊 Bankruptcy Prediction Distribution")

    pred_counts = pd.Series(predictions).value_counts()

    fig5, ax5 = plt.subplots()
    ax5.bar(pred_counts.index.astype(str), pred_counts.values)
    ax5.set_xlabel("Prediction")
    ax5.set_ylabel("Count")

    st.pyplot(fig5)
