
import streamlit as st
import pandas as pd
import pickle

# ================================
# Muat Full Pipeline Model
# ================================
try:
    with open('full_model.pkl', 'rb') as file:
        full_model = pickle.load(file)
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ================================
# Streamlit UI
# ================================
st.title("🏡 Prediksi Harga Rumah Jabodetabek")
st.write("Aplikasi estimasi harga rumah berdasarkan data properti.")
st.write("⚠️ *Hasil hanya estimasi dan bukan acuan harga pasti.*")

st.sidebar.header("Input Properti Rumah")

def user_input_features():
    bedrooms = st.sidebar.slider("Jumlah Kamar Tidur", 1, 8, 3)
    bathrooms = st.sidebar.slider("Jumlah Kamar Mandi", 1, 4, 2)
    land_size_m2 = st.sidebar.slider("Luas Tanah (m²)", 10.0, 400.0, 100.0)
    building_size_m2 = st.sidebar.slider("Luas Bangunan (m²)", 10.0, 400.0, 90.0)
    carports = st.sidebar.slider("Jumlah Carport", 0, 3, 1)
    floors = st.sidebar.slider("Jumlah Lantai", 1, 3, 2)
    building_age = st.sidebar.slider("Usia Bangunan (tahun)", 0, 15, 0)
    garages = st.sidebar.slider("Jumlah Garasi", 0, 2, 0)

    city = st.sidebar.selectbox("Kota", [
        "Bekasi", "Bogor", "Depok", "Jakarta Barat", "Jakarta Pusat",
        "Jakarta Selatan", "Jakarta Timur", "Jakarta Utara", "Tangerang"
    ])

    furnishing = st.sidebar.selectbox("Perabotan", [
        "unfurnished", "semi furnished", "furnished", "baru"
    ])

    data = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'land_size_m2': land_size_m2,
        'building_size_m2': building_size_m2,
        'carports': carports,
        'floors': floors,
        'building_age': building_age,
        'garages': garages,
        'city': city,
        'furnishing': furnishing
    }
    return pd.DataFrame(data, index=[0])

df_input = user_input_features()

st.subheader("📌 Parameter Input")
st.write(df_input)

# ================================
# Prediksi
# ================================
if st.sidebar.button("Prediksi Harga"):
    try:
        prediction = full_model.predict(df_input)[0]
        st.subheader("💰 Estimasi Harga Rumah")
        st.success(f"Rp {prediction:,.2f}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
