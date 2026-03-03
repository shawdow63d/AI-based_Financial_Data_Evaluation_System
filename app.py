import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==============================
# CẤU HÌNH TRANG
# ==============================
st.set_page_config(page_title="AI Financial Analysis", layout="wide")
st.title("💰 AI Phân tích Tài chính Doanh nghiệp")
st.write("Hệ thống dự đoán Phá sản và Tăng trưởng bằng Machine Learning.")

# =====================================================
# PHẦN 1 — DỰ BÁO PHÁ SẢN
# =====================================================

st.header("🔴 AI Đánh giá Nguy cơ Phá sản")

@st.cache_resource
def load_bankruptcy_artifacts():
    try:
        return joblib.load("bankruptcy_model.pkl")
    except Exception as e:
        st.error(f"Không tìm thấy bankruptcy_model.pkl: {e}")
        return None

bankruptcy_data = load_bankruptcy_artifacts()

if bankruptcy_data:

    model = bankruptcy_data['model']
    imputer = bankruptcy_data['imputer']
    scaler = bankruptcy_data['scaler']
    feature_names = bankruptcy_data['feature_names']

    st.sidebar.header("📥 Nhập dữ liệu (Phá sản)")
    option = st.sidebar.radio(
        "Chọn cách nhập:",
        ["Upload file Excel/CSV (Phá sản)", "Demo Phá sản"]
    )

    input_df = None

    if option == "Upload file Excel/CSV (Phá sản)":
        uploaded_file = st.sidebar.file_uploader(
            "Tải file Phá sản",
            type=["csv", "xlsx"],
            key="bankruptcy_upload"
        )
        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                input_df = pd.read_csv(uploaded_file)
            else:
                input_df = pd.read_excel(uploaded_file)
            st.success("Tải file thành công!")

    else:
        if st.sidebar.button("Sinh dữ liệu mẫu (Phá sản)"):
            random_data = np.random.rand(1, len(feature_names))
            input_df = pd.DataFrame(random_data, columns=feature_names)

    if input_df is not None:

        if st.button("Chạy Dự báo Phá sản"):

            try:
                X = input_df[feature_names]
                X = imputer.transform(X)
                X = scaler.transform(X)

                pred = model.predict(X)
                proba = model.predict_proba(X)[:, 1]

                results = input_df.copy()
                results["Dự đoán"] = [
                    "NGUY CƠ PHÁ SẢN" if p == 1 else "An toàn"
                    for p in pred
                ]
                results["Tỉ lệ rủi ro (%)"] = (proba * 100).round(2)

                st.dataframe(results)

            except Exception as e:
                st.error(f"Lỗi phá sản: {e}")

# =====================================================
# PHẦN 2 — DỰ BÁO TĂNG TRƯỞNG
# Load riêng model, scaler, imputer
# =====================================================

st.divider()
st.header("📈 AI Dự báo Tăng trưởng")

@st.cache_resource
def load_growth_artifacts():
    try:
        model = joblib.load("model_xgb.pkl")
        scaler = joblib.load("scaler.pkl")
        imputer = joblib.load("imputer.pkl")
        return model, scaler, imputer
    except Exception as e:
        st.error(f"Lỗi load model tăng trưởng: {e}")
        return None, None, None

growth_model, growth_scaler, growth_imputer = load_growth_artifacts()

if growth_model:

    st.sidebar.header("📥 Nhập dữ liệu (Tăng trưởng)")

    growth_file = st.sidebar.file_uploader(
        "Tải file cho mô hình Tăng trưởng",
        type=["csv", "xlsx"],
        key="growth_upload"
    )

    growth_df = None

    if growth_file:
        if growth_file.name.endswith(".csv"):
            growth_df = pd.read_csv(growth_file)
        else:
            growth_df = pd.read_excel(growth_file)
        st.success("Tải file tăng trưởng thành công!")

    if growth_df is not None:

        if st.button("Chạy Dự báo Tăng trưởng"):

            try:
                # ⚠️ PHẢI đúng thứ tự cột lúc train
                X_growth = growth_df.copy()

                X_growth = growth_imputer.transform(X_growth)
                X_growth = growth_scaler.transform(X_growth)

                growth_pred = growth_model.predict(X_growth)

                results_growth = growth_df.copy()
                results_growth["Dự báo Tăng trưởng"] = growth_pred

                st.dataframe(results_growth)

            except Exception as e:
                st.error(f"Lỗi tăng trưởng: {e}")
