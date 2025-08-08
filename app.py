import streamlit as st
import pandas as pd
import joblib

model = joblib.load('model.pkl')
st.set_page_config(page_title="Student Performance Prediction", layout="centered")
st.title('🎓 Prediksi Indeks Performa Siswa')

with st.form("prediksi_form"):
    hours = st.slider('Jam Belajar per Hari', 0.0, 10.0, step=0.1)
    prev_scores = st.slider('Nilai Sebelumnya', 0, 100, step=1)
    extra = st.selectbox('Ekstrakurikuler (Yes/No)', ['Yes', 'No'])
    sleep = st.slider('Jam Tidur per Hari', 0.0, 12.0, step=0.1)
    papers = st.slider('Jumlah Soal Latihan', 0, 10, step=1)
    submitted = st.form_submit_button("Prediksi")

if submitted:
    input_df = pd.DataFrame({
        'Hours Studied': [hours],
        'Previous Scores': [prev_scores],
        'Extracurricular Activities': [extra],
        'Sleep Hours': [sleep],
        'Sample Question Papers Practiced': [papers]
    })
    pred = model.predict(input_df)[0]
    st.success(f"Prediksi Performance Index: **{pred:.2f}**")
