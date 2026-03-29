import streamlit as st
import numpy as np
import pickle

# ------------------ LOAD MODELS ------------------ #
with open("xgb_model.pkl", "rb") as f:
    price_model = pickle.load(f)

with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)



# ------------------ TITLE ------------------ #
st.title("💎 Diamond Price & Market Segment Predictor")

st.write("Enter diamond details below:")

# ------------------ INPUTS ------------------ #
col1, col2 = st.columns(2)

with col1:
    carat = st.number_input("Carat", min_value=0.0, step=0.1)
    x = st.number_input("Length (x)", min_value=0.0)
    y = st.number_input("Width (y)", min_value=0.0)
    z = st.number_input("Depth (z)", min_value=0.0)

with col2:
    cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
    color = st.selectbox("Color", ["J", "I", "H", "G", "F", "E", "D"])
    clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])

# ------------------ ENCODING ------------------ #
cut_map = {"Fair":0, "Good":1, "Very Good":2, "Premium":3, "Ideal":4}
color_map = {"J":0, "I":1, "H":2, "G":3, "F":4, "E":5, "D":6}
clarity_map = {"I1":0, "SI2":1, "SI1":2, "VS2":3, "VS1":4, "VVS2":5, "VVS1":6, "IF":7}

cut_val = cut_map[cut]
color_val = color_map[color]
clarity_val = clarity_map[clarity]

# ------------------ FEATURE ENGINEERING ------------------ #
volume = x * y * z
dimension_ratio = x / y if y != 0 else 0

features = np.array([[carat, cut, color, clarity, depth, dimension_ratio,volume]])


# ------------------ BUTTON ------------------ #
if st.button("🔍 Predict"):

    # -------- PRICE PREDICTION -------- #
    price_usd = price_model.predict(features)[0]
    price_inr = price_usd * 83

    # -------- CLUSTER PREDICTION -------- #
    scaled_features = scaler.transform(features)
    cluster = kmeans.predict(scaled_features)[0]

    # -------- CLUSTER NAME -------- #
    if cluster == 0:
        cluster_name = "💰 Affordable Small Diamonds"
    elif cluster == 1:
        cluster_name = "⚖️ Mid-range Balanced Diamonds"
    else:
        cluster_name = "💎 Premium Heavy Diamonds"

    # ------------------ OUTPUT ------------------ #
    st.subheader("📊 Results")

    st.success(f"💰 Predicted Price: ₹ {price_inr:,.2f}")
    st.info(f"📦 Cluster: {cluster}")
    st.warning(f"🏷️ Segment: {cluster_name}")
