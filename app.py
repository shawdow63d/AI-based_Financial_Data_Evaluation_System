import streamlit as st
import pandas as pd
import joblib
import numpy as np
import json

# ==============================
# CẤU HÌNH TRANG
# ==============================
st.set_page_config(page_title="AI Đánh giá Tài chính Doanh nghiệp", layout="wide")
st.title("💰 AI Đánh giá Sức khỏe Tài chính Doanh nghiệp")
st.write("Hệ thống sử dụng Machine Learning để dự đoán nguy cơ phá sản và tăng trưởng doanh nghiệp.")

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_artifacts():
    try:
        # ----- Model Phá sản -----
        bankruptcy_data = joblib.load("bankruptcy_model.pkl")

        # ----- Model Tăng trưởng -----
        growth_model = joblib.load("model_xgb.pkl")
        growth_imputer = joblib.load("imputer.pkl")
        growth_scaler = joblib.load("scaler.pkl")

        with open("features_list.json", "r") as f:
            growth_features = json.load(f)

        return {
            "bankruptcy": bankruptcy_data,
            "growth_model": growth_model,
            "growth_imputer": growth_imputer,
            "growth_scaler": growth_scaler,
            "growth_features": growth_features
        }

    except Exception as e:
        st.error(f"Lỗi load model: {e}")
        return None


data = load_artifacts()

# ==============================
# NẾU LOAD OK
# ==============================
if data:

    # ----- Phá sản -----
    bankruptcy_model = data["bankruptcy"]["model"]
    bankruptcy_imputer = data["bankruptcy"]["imputer"]
    bankruptcy_scaler = data["bankruptcy"]["scaler"]
    bankruptcy_features = data["bankruptcy"]["feature_names"]

    # ----- Tăng trưởng -----
    growth_model = data["growth_model"]
    growth_imputer = data["growth_imputer"]
    growth_scaler = data["growth_scaler"]
    growth_features = data["growth_features"]

    # ==============================
    # SIDEBAR
    # ==============================
    st.sidebar.header("📥 Nhập dữ liệu")
    option = st.sidebar.radio("Chọn cách nhập:", ["Upload file Excel/CSV", "Nhập thủ công (Demo)"])

    input_df = None

    if option == "Upload file Excel/CSV":
        uploaded_file = st.sidebar.file_uploader("Tải lên file dữ liệu", type=["csv", "xlsx"])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".csv"):
                    input_df = pd.read_csv(uploaded_file)
                else:
                    input_df = pd.read_excel(uploaded_file)
                st.success("Tải file thành công!")
            except:
                st.error("Lỗi định dạng file!")

    else:
        st.info("Chế độ Demo: Sinh dữ liệu ngẫu nhiên.")
        if st.sidebar.button("Sinh dữ liệu mẫu"):
            random_data = np.random.rand(1, len(bankruptcy_features))
            input_df = pd.DataFrame(random_data, columns=bankruptcy_features)
            st.dataframe(input_df)

    # ==============================
    # XỬ LÝ DỰ ĐOÁN
    # ==============================
    if input_df is not None:

        if st.button("🚀 Chạy Phân tích"):

            try:
                results = input_df.copy()

                # =====================================================
                # 1️⃣ PHÂN TÍCH PHÁ SẢN
                # =====================================================
                X_bank = input_df[bankruptcy_features]
                X_bank_filled = bankruptcy_imputer.transform(X_bank)
                X_bank_scaled = bankruptcy_scaler.transform(X_bank_filled)

                prediction = bankruptcy_model.predict(X_bank_scaled)
                proba = bankruptcy_model.predict_proba(X_bank_scaled)[:, 1]

                results["Dự đoán Phá sản"] = [
                    "NGUY CƠ PHÁ SẢN" if p == 1 else "An toàn"
                    for p in prediction
                ]
                results["Tỉ lệ rủi ro (%)"] = (proba * 100).round(2)

                # =====================================================
                # 2️⃣ PHÂN TÍCH TĂNG TRƯỞNG
                # =====================================================
                X_growth = input_df[growth_features]
                X_growth_filled = growth_imputer.transform(X_growth)
                X_growth_scaled = growth_scaler.transform(X_growth_filled)

                growth_prediction = growth_model.predict(X_growth_scaled)

                results["Dự đoán Tăng trưởng (%)"] = (
                    np.array(growth_prediction).reshape(-1) * 100
                ).round(2)

                # =====================================================
                # HIỂN THỊ
                # =====================================================
                st.subheader("📊 Kết quả Phân tích Tổng hợp")

                risk_count = np.sum(prediction)
                st.metric("Số công ty báo động đỏ", int(risk_count))

                def color_danger(val):
                    if val == "NGUY CƠ PHÁ SẢN":
                        return "color: red; font-weight: bold"
                    return "color: green"

                st.dataframe(
                    results.style.applymap(color_danger, subset=["Dự đoán Phá sản"])
                )

            except KeyError as e:
                st.error(f"Thiếu cột dữ liệu: {e}")
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {e}")
