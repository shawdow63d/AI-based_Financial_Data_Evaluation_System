import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==============================
# CẤU HÌNH TRANG
# ==============================
st.set_page_config(page_title="AI Financial Analysis", layout="wide")
st.title("💰 AI Phân tích Tài chính Doanh nghiệp")
st.write("Hệ thống sử dụng Machine Learning để dự đoán Phá sản và Tăng trưởng.")

# =====================================================
# PHẦN 1 — DỰ BÁO PHÁ SẢN
# =====================================================

st.header("🔴 AI Đánh giá Nguy cơ Phá sản")

@st.cache_resource
def load_bankruptcy_artifacts():
    try:
        return joblib.load("bankruptcy_model.pkl")
    except Exception as e:
        st.error(f"Không tìm thấy file 'bankruptcy_model.pkl'. Lỗi: {e}")
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
        ["Upload file Excel/CSV (Phá sản)", "Nhập thủ công (Demo Phá sản)"]
    )

    input_df = None

    if option == "Upload file Excel/CSV (Phá sản)":
        uploaded_file = st.sidebar.file_uploader(
            "Tải lên file dữ liệu",
            type=["csv", "xlsx"],
            key="bankruptcy_upload"
        )
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    input_df = pd.read_csv(uploaded_file)
                else:
                    input_df = pd.read_excel(uploaded_file)
                st.success("Tải file thành công!")
            except:
                st.error("Lỗi định dạng file!")

    else:
        if st.sidebar.button("Sinh dữ liệu mẫu (Phá sản)"):
            random_data = np.random.rand(1, len(feature_names))
            input_df = pd.DataFrame(random_data, columns=feature_names)
            st.dataframe(input_df)

    if input_df is not None:

        st.subheader("📊 Kết quả Phân tích Phá sản")

        if st.button("Chạy Dự báo Phá sản"):

            try:
                X_input = input_df[feature_names]
                X_filled = imputer.transform(X_input)
                X_scaled = scaler.transform(X_filled)

                prediction = model.predict(X_scaled)
                proba = model.predict_proba(X_scaled)[:, 1]

                results = input_df.copy()
                results['Dự đoán'] = [
                    "NGUY CƠ PHÁ SẢN" if p == 1 else "An toàn"
                    for p in prediction
                ]
                results['Tỉ lệ rủi ro (%)'] = (proba * 100).round(2)

                risky_df = results[results['Dự đoán'] == 'NGUY CƠ PHÁ SẢN']
                risk_count = np.sum(prediction)

                st.metric("Số công ty báo động đỏ", int(risk_count))

                def color_danger(val):
                    return 'color: red; font-weight: bold' if val == "NGUY CƠ PHÁ SẢN" else 'color: green'

                st.subheader("⚠️ Danh sách Doanh nghiệp Báo động đỏ")
                if not risky_df.empty:
                    st.dataframe(
                        risky_df.style.applymap(color_danger, subset=['Dự đoán'])
                    )
                else:
                    st.success("Không có doanh nghiệp nào có nguy cơ phá sản.")

                st.subheader("📋 Dữ liệu toàn bộ")
                st.dataframe(results)

            except KeyError as e:
                st.error(f"File thiếu cột dữ liệu: {e}")
            except Exception as e:
                st.error(f"Lỗi dự báo phá sản: {e}")


# =====================================================
# PHẦN 2 — DỰ BÁO TĂNG TRƯỞNG (ĐỘC LẬP HOÀN TOÀN)
# =====================================================

st.divider()
st.header("📈 AI Dự báo Tăng trưởng Doanh nghiệp")

@st.cache_resource
def load_growth_artifacts():
    try:
        return joblib.load("model_xgb.pkl")
    except Exception as e:
        st.error(f"Không tìm thấy file 'model_xgb.pkl'. Lỗi: {e}")
        return None

growth_data = load_growth_artifacts()

if growth_data:

    growth_model = growth_data['model']
    growth_imputer = growth_data['imputer']
    growth_scaler = growth_data['scaler']
    growth_feature_names = growth_data['feature_names']

    st.sidebar.header("📥 Nhập dữ liệu (Tăng trưởng)")
    growth_option = st.sidebar.radio(
        "Chọn cách nhập cho Tăng trưởng:",
        ["Upload file Excel/CSV (Tăng trưởng)", "Nhập thủ công (Demo Tăng trưởng)"],
        key="growth_option"
    )

    growth_input_df = None

    if growth_option == "Upload file Excel/CSV (Tăng trưởng)":
        growth_uploaded_file = st.sidebar.file_uploader(
            "Tải file cho mô hình Tăng trưởng",
            type=["csv", "xlsx"],
            key="growth_upload"
        )

        if growth_uploaded_file:
            try:
                if growth_uploaded_file.name.endswith('.csv'):
                    growth_input_df = pd.read_csv(growth_uploaded_file)
                else:
                    growth_input_df = pd.read_excel(growth_uploaded_file)
                st.success("Tải file tăng trưởng thành công!")
            except:
                st.error("Lỗi file tăng trưởng!")

    else:
        if st.sidebar.button("Sinh dữ liệu mẫu (Tăng trưởng)", key="growth_demo"):
            random_data = np.random.rand(1, len(growth_feature_names))
            growth_input_df = pd.DataFrame(
                random_data,
                columns=growth_feature_names
            )
            st.dataframe(growth_input_df)

    if growth_input_df is not None:

        st.subheader("📊 Kết quả Dự báo Tăng trưởng")

        if st.button("Chạy Dự báo Tăng trưởng"):

            try:
                X_growth = growth_input_df[growth_feature_names]
                X_growth_filled = growth_imputer.transform(X_growth)
                X_growth_scaled = growth_scaler.transform(X_growth_filled)

                growth_prediction = growth_model.predict(X_growth_scaled)

                growth_results = growth_input_df.copy()
                growth_results['Dự báo Tăng trưởng'] = growth_prediction

                st.dataframe(growth_results)

            except KeyError as e:
                st.error(f"Thiếu cột cho model tăng trưởng: {e}")
            except Exception as e:
                st.error(f"Lỗi dự báo tăng trưởng: {e}")

