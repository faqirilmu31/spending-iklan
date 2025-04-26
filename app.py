# app.py

import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# Load model dan scaler
model = load_model('best_lstm_model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit App
st.title("Prediksi Lead 15 Hari ke Depan dari Spending Iklan")

st.write("Masukkan 15 spending terakhir (spending hari ini, kemarin, hingga 14 hari lalu):")

# Input box
user_input = st.text_area("Input 15 angka spending, pisahkan dengan koma", 
                          placeholder="contoh: 1000000,1100000,1200000,...")

if st.button("Prediksi Lead"):
    try:
        # Parsing input
        spending_list = list(map(float, user_input.strip().split(',')))

        if len(spending_list) != 15:
            st.warning("⚠️ Jumlah input harus tepat 15 angka!")
        else:
            # Preprocessing input
            spending_scaled = scaler.transform(np.array(spending_list).reshape(1, -1))
            spending_scaled = spending_scaled.reshape((1, 15, 1))

            # Predict
            prediction = model.predict(spending_scaled)

            # Tampilkan hasil
            st.success("✅ Prediksi berhasil! Berikut hasilnya:")
            pred_df = pd.DataFrame({
                f'Lead Lag {i} (Hari ke-{i})': [round(prediction[0][i])] for i in range(15)
            })
            st.table(pred_df)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
