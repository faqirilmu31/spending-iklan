# app.py

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

# --- Load Model dan Scaler ---
model = load_model('best_cnn_model1.h5', compile=False)

with open('scaler_cnn.pkl', 'rb') as f:
    scaler = pickle.load(f)

# --- Streamlit UI ---
st.title("🚀 Prediksi Lead 15 Hari dari Spending Iklan")
st.write("Masukkan 15 data spending terakhir (spending hari ini, kemarin, dst):")

# --- Input Spending User ---
user_input = st.text_area(
    "Input 30 angka spending, pisahkan dengan koma (,)",
    placeholder="contoh: 1000000,1100000,1200000,..."
)

if st.button("Prediksi Lead"):
    try:
        # Parsing input
        spending_list = list(map(float, user_input.strip().split(',')))

        if len(spending_list) != 30:
            st.warning("⚠️ Jumlah input harus tepat 30 angka!")
        else:
            # Preprocessing
            spending_scaled = scaler.transform(np.array(spending_list).reshape(1, -1))
            spending_scaled = spending_scaled.reshape((1, 30, 1))

            # Predict
            prediction = model.predict(spending_scaled)

            # Display
            st.success("✅ Prediksi berhasil! Berikut hasilnya:")
            pred_df = pd.DataFrame({
                "Hari ke-": list(range(15)),
                "Prediksi Lead": [round(prediction[0][i]) for i in range(15)]
            })
            st.dataframe(pred_df, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
