import streamlit as st
import numpy as np
import pickle

st.title("💎 Diamond Price Predictor")

# ------------------ LOAD MODELS ------------------ #
try:
    with open("xgb_model.pkl", "rb") as f:
        price_model = pickle.load(f)

    with open("kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# ------------------ INPUT ------------------ #
carat = st.number_input("Carat")
x = st.number_input("x")
y = st.number_input("y")
z = st.number_input("z")

cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color", ["J", "I", "H", "G", "F", "E", "D"])
clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])

# ------------------ ENCODING ------------------ #
cut_map = {"Fair":0, "Good":1, "Very Good":2, "Premium":3, "Ideal":4}
color_map = {"J":0, "I":1, "H":2, "G":3, "F":4, "E":5, "D":6}
clarity_map = {"I1":0, "SI2":1, "SI1":2, "VS2":3, "VS1":4, "VVS2":5, "VVS1":6, "IF":7}

# ------------------ FEATURES ------------------ #
volume = x*y*z
dimension_ratio = x/y if y != 0 else 0

features = np.array([[carat, cut_map[cut], color_map[color], clarity_map[clarity], volume, dimension_ratio, z]])

# ------------------ PREDICT ------------------ #
if st.button("Predict"):
    try:
        price = price_model.predict(features)[0]
        scaled = scaler.transform(features)
        cluster = kmeans.predict(scaled)[0]

        st.success(f"Price: ₹ {price*83:,.2f}")
        st.write(f"Cluster: {cluster}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
