
import streamlit as st
import pandas as pd
import joblib

# Load the preprocessor, PCA, and the trained model
preprocessor = joblib.load('preprocessor.pkl')
pca = joblib.load('pca.pkl')
gbr_model = joblib.load('gbr_model.pkl')

st.title('House Price Prediction App')
st.write('Enter the details of the house to predict its price in Jabodetabek.')

# Input fields for features
city = st.selectbox('City', ['Bekasi', 'Bogor', 'Depok', 'Jakarta Barat', 'Jakarta Pusat', 'Jakarta Selatan', 'Jakarta Timur', 'Jakarta Utara', 'Tangerang'])
bedrooms = st.slider('Number of Bedrooms', 1, 10, 3)
bathrooms = st.slider('Number of Bathrooms', 1, 10, 2)
land_size_m2 = st.number_input('Land Size (m2)', min_value=10.0, max_value=1000.0, value=100.0, step=10.0)
building_size_m2 = st.number_input('Building Size (m2)', min_value=10.0, max_value=1000.0, value=80.0, step=10.0)
floors = st.slider('Number of Floors', 1, 5, 2)
furnishing = st.selectbox('Furnishing', ['unfurnished', 'furnished', 'semi furnished', 'baru'])

# Create a DataFrame from user input
input_data = pd.DataFrame([{
    'city': city,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'land_size_m2': land_size_m2,
    'building_size_m2': building_size_m2,
    'floors': floors,
    'furnishing': furnishing
}])

if st.button('Predict Price'):
    try:
        # Ensure all columns expected by the preprocessor are present
        # This ensures consistency even if some features are not used in the input form.
        # The `features_to_use_for_X` from the notebook was: ['city', 'bedrooms', 'bathrooms', 'land_size_m2', 'building_size_m2', 'floors', 'furnishing']
        # The preprocessor expects these in the original order.

        # The order of columns in X_features was:
        # ['city', 'bedrooms', 'bathrooms', 'land_size_m2', 'building_size_m2', 'floors', 'furnishing']
        # Re-order input_data columns to match the training data's feature order
        ordered_input_data = input_data[['city', 'bedrooms', 'bathrooms', 'land_size_m2', 'building_size_m2', 'floors', 'furnishing']]

        # Transform input data using the preprocessor
        transformed_input = preprocessor.transform(ordered_input_data)

        # Apply PCA
        pca_transformed_input = pca.transform(transformed_input)

        # Predict price
        predicted_price = gbr_model.predict(pca_transformed_input)[0]

        st.success(f'Predicted House Price: Rp {predicted_price:,.2f}')
    except Exception as e:
        st.error(f'An error occurred during prediction: {e}')
