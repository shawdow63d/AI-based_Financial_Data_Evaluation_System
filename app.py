import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Cấu hình trang
st.set_page_config(page_title="Dự báo Phá sản Doanh nghiệp", layout="wide")
st.title("AI Đánh giá Sức khỏe Tài chính Doanh nghiệp")
st.write("Hệ thống sử dụng Machine Learning (XGBoost) để dự đoán nguy cơ phá sản.")

# Load model và các công cụ
@st.cache_resource
def load_artifacts():
    try:
        artifacts = joblib.load('bankruptcy_model.pkl')
        return artifacts
    except Exception as e:
        st.error(f"Không tìm thấy file 'bankruptcy_model.pkl'. Hãy chắc chắn nó ở cùng thư mục với app.py! Lỗi: {e}")
        return None

data = load_artifacts()

if data:
    model = data['model']
    imputer = data['imputer']
    scaler = data['scaler'] 
    feature_names = data['feature_names']

    # Sidebar nhập liệu
    st.sidebar.header("📥 Nhập dữ liệu")
    option = st.sidebar.radio("Chọn cách nhập:", ["Upload file Excel/CSV", "Nhập thủ công (Demo)"])
    
    input_df = None

    if option == "Upload file Excel/CSV":
        uploaded_file = st.sidebar.file_uploader("Tải lên file dữ liệu", type=["csv", "xlsx"])
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
        st.info("Chế độ Demo: Tự sinh dữ liệu ngẫu nhiên giả lập 1 công ty.")
        if st.sidebar.button("Sinh dữ liệu mẫu"):
            random_data = np.random.rand(1, len(feature_names))
            input_df = pd.DataFrame(random_data, columns=feature_names)
            st.write("Dữ liệu đầu vào (Mô phỏng):")
            st.dataframe(input_df)

    # Xử lý và Hiển thị
    if input_df is not None:
        st.subheader("📊 Kết quả Phân tích")
        
        if st.button("Chạy Dự báo ngay"):
            try:
                # 1. Tiền xử lý
                X_input = input_df[feature_names]
                X_filled = imputer.transform(X_input)
                X_scaled = scaler.transform(X_filled)
                
                # 2. Dự báo
                prediction = model.predict(X_scaled) 
                proba = model.predict_proba(X_scaled)[:, 1]
                
                # 3. Gán kết quả
                results = input_df.copy()
                results['Dự đoán'] = ["NGUY CƠ PHÁ SẢN" if p == 1 else "An toàn" for p in prediction]
                results['Tỉ lệ rủi ro (%)'] = (proba * 100).round(2)

                # --- ĐOẠN CODE ĐÃ SỬA THEO CÁCH 1 (TỐI ƯU HIỆU NĂNG) ---
                
                # A. Tách riêng danh sách nguy hiểm
                risky_df = results[results['Dự đoán'] == 'NGUY CƠ PHÁ SẢN']
                risk_count = np.sum(prediction)

                # B. Hiển thị số lượng
                st.metric("Số công ty báo động đỏ", int(risk_count))

                # C. Chỉ tô màu danh sách nguy hiểm (nhỏ và nhẹ)
                def color_danger(val):
                    return 'color: red; font-weight: bold' if val == "NGUY CƠ PHÁ SẢN" else 'color: green'

                st.subheader("⚠️ Danh sách Doanh nghiệp Báo động đỏ")
                if not risky_df.empty:
                    st.dataframe(risky_df.style.applymap(color_danger, subset=['Dự đoán']))
                else:
                    st.success("Tuyệt vời! Không tìm thấy doanh nghiệp nào có nguy cơ phá sản trong file này.")

                # D. Hiển thị bảng gốc dạng thường (xử lý được file lớn không bị lỗi)
                st.subheader("📋 Dữ liệu toàn bộ (Chi tiết)")
                st.dataframe(results) 
                
                # --- HẾT PHẦN SỬA ---

            except KeyError as e:
                st.error(f"File của bạn thiếu cột dữ liệu quan trọng: {e}")
            except Exception as e:
                st.error(f"Có lỗi xảy ra: {e}")
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


