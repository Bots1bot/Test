import streamlit as st
import pandas as pd
import pickle

# ====== PAGE CONFIG ======
st.set_page_config(
    page_title="Prediksi Harga Rumah Jabodetabek",
    page_icon="🏠",
    layout="wide"
)

# ====== LOAD MODEL ======
try:
    with open('full_model.pkl', 'rb') as f:
        full_model = pickle.load(f)
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# ====== HEADER ======
st.markdown("<h1 style='text-align:center;'>🏠 Prediksi Harga Rumah Jabodetabek</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:gray;'>Estimasi harga rumah berdasarkan fitur properti</h4>",
            unsafe_allow_html=True)
st.write("---")

# ====== INPUT FORM ======
st.sidebar.title("⚙️ Parameter Properti Rumah")

def input_user():
    bedrooms = st.sidebar.number_input("🛏️ Jumlah Kamar Tidur", 1, 8, 3)
    bathrooms = st.sidebar.number_input("🛁 Jumlah Kamar Mandi", 1, 4, 2)
    land_size_m2 = st.sidebar.number_input("🌿 Luas Tanah (m²)", 10.0, 400.0, 120.0)
    building_size_m2 = st.sidebar.number_input("🏡 Luas Bangunan (m²)", 10.0, 400.0, 90.0)
    floors = st.sidebar.number_input("⬆️ Jumlah Lantai", 1, 3, 2)

    city = st.sidebar.selectbox(
        "📍 Kota",
        ["Bekasi", "Bogor", "Depok", "Jakarta Barat", "Jakarta Pusat",
         "Jakarta Selatan", "Jakarta Timur", "Jakarta Utara", "Tangerang"]
    )

    furnishing = st.sidebar.selectbox(
        "🛋️ Kondisi Perabotan",
        ["baru", "furnished", "semi furnished", "unfurnished"]
    )

    data = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'land_size_m2': land_size_m2,
        'building_size_m2': building_size_m2,
        'floors': floors,
        'city': city,
        'furnishing': furnishing
    }

    return pd.DataFrame(data, index=[0])


df_input = input_user()

# ====== SHOW INPUT ======
with st.expander("📌 Data Input Anda"):
    st.dataframe(df_input, use_container_width=True)

# ====== VALIDATION ======
def validate(df):
    if df["building_size_m2"].iloc[0] > df["land_size_m2"].iloc[0]:
        st.error("⚠️ Luas bangunan tidak boleh lebih besar dari luas tanah!")
        return False
    return True

# ====== PREDIKSI ======
predict_button = st.button("🔍 Prediksi Harga Rumah", use_container_width=True)

if predict_button:
    if validate(df_input):
        try:
            price = full_model.predict(df_input)[0]

            # CARD KEREN UNTUK HASIL
            st.success("🎯 Prediksi berhasil!")
            st.markdown(
                f"""
                <div style="
                    background-color:#f0f8ff;
                    padding:25px;
                    border-radius:15px;
                    text-align:center;
                    border: 3px solid #1f77b4;
                ">
                    <h2 style="color:#1f77b4;">Estimasi Harga Rumah</h2>
                    <h1 style="color:#0d4d8b;">Rp {price:,.2f}</h1>
                    <p style="color:gray;">⚠️ Estimasi berdasarkan data model — bukan harga final pasar.</p>
                </div>
                """, unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")

st.write("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Developed with ❤️ using Machine Learning</p>",
    unsafe_allow_html=True
)
