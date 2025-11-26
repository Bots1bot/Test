import streamlit as st
import pandas as pd
import pickle

# ========================
# Load trained model
# ========================
with open('full_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🏡 Prediksi Harga Rumah Jabodetabek")
st.write("Aplikasi estimasi harga rumah berdasarkan data properti.")
st.write("⚠️ Hasil hanya estimasi, bukan acuan harga pasti.")

# ========================
# Sidebar Input
# ========================
st.sidebar.header("Input Properti Rumah")

bedrooms = st.sidebar.slider("Jumlah Kamar Tidur", 1, 8, 3)
bathrooms = st.sidebar.slider("Jumlah Kamar Mandi", 1, 4, 2)
land_size_m2 = st.sidebar.slider("Luas Tanah (m²)", 10.0, 400.0, 100.0)
building_size_m2 = st.sidebar.slider("Luas Bangunan (m²)", 10.0, 400.0, 90.0)
floors = st.sidebar.slider("Jumlah Lantai", 1, 3, 2)

city = st.sidebar.selectbox(
    "Kota",
    ["Bekasi", "Bogor", "Depok", "Jakarta Barat", "Jakarta Pusat",
     "Jakarta Selatan", "Jakarta Timur", "Jakarta Utara", "Tangerang"]
)

furnishing = st.sidebar.selectbox(
    "Kondisi Furnishing",
    ["baru", "furnished", "semi furnished", "unfurnished"]
)

# ========================
# Build DataFrame sesuai kolom model
# ========================
data = {
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'land_size_m2': land_size_m2,
    'building_size_m2': building_size_m2,
    'floors': floors,
    # one hot city
    'city_ Bekasi': 1 if city == "Bekasi" else 0,
    'city_ Bogor': 1 if city == "Bogor" else 0,
    'city_ Depok': 1 if city == "Depok" else 0,
    'city_ Jakarta Barat': 1 if city == "Jakarta Barat" else 0,
    'city_ Jakarta Pusat': 1 if city == "Jakarta Pusat" else 0,
    'city_ Jakarta Selatan': 1 if city == "Jakarta Selatan" else 0,
    'city_ Jakarta Timur': 1 if city == "Jakarta Timur" else 0,
    'city_ Jakarta Utara': 1 if city == "Jakarta Utara" else 0,
    'city_ Tangerang': 1 if city == "Tangerang" else 0,
    # one hot furnishing
    'furnishing_baru': 1 if furnishing == "baru" else 0,
    'furnishing_furnished': 1 if furnishing == "furnished" else 0,
    'furnishing_semi furnished': 1 if furnishing == "semi furnished" else 0,
    'furnishing_unfurnished': 1 if furnishing == "unfurnished" else 0
}

df = pd.DataFrame([data])

st.subheader("📌 Data Input")
st.write(df)

# ========================
# Prediction
# ========================
if st.sidebar.button("Prediksi Harga"):
    try:
        prediction = model.predict(df)[0]
        st.subheader("💰 Estimasi Harga Properti")
        st.success(f"Rp {prediction:,.2f}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
