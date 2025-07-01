import streamlit as st
import numpy as np
import pickle
import json

# Load model
with open('Model/bangalore_home_price_model.pickle', 'rb') as file:
    model = pickle.load(file)

# Load columns.json
with open('Model/columns.json', 'r') as file:
    data_columns = json.load(file)
    columns = data_columns['data_columns']
    locations = columns[3:]  # Skip ['total_sqft', 'bath', 'bhk']

# Streamlit UI
st.title("🏡 Bangalore Home Price Prediction")

location = st.selectbox("Select Location", locations)
sqft = st.number_input("Total Square Feet", min_value=300.0)
bath = st.slider("Bathrooms", 1, 5, 2)
bhk = st.slider("Bedrooms (BHK)", 1, 5, 2)

if st.button("Predict Price"):
    x = np.zeros(len(columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if location in columns:
        loc_index = columns.index(location)
        x[loc_index] = 1

    predicted_price = model.predict([x])[0]
    st.success(f"Estimated Price: ₹ {predicted_price:,.2f} Lakhs")
