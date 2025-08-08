import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Set tampilan halaman
st.set_page_config(page_title="Student Performance Prediction", layout="centered")
st.title('🎓 Prediksi Indeks Performa Siswa')

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Student_Performance.csv")
    return df

df = load_data()

# Encode kolom kategorikal
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

# Pisahkan fitur dan target
X = df[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
y = df['Performance Index']

# Split dan latih model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

st.write("Masukkan data berikut untuk memprediksi performa siswa:")

# Input form
with st.form("prediction_form"):
    hours = st.slider('Hours Studied', 0.0, 10.0, step=0.1)
    previous_scores = st.slider('Previous Scores', 0, 100, step=1)
    extra = st.selectbox('Extracurricular Activities', ['Yes', 'No'])
    sleep = st.slider('Sleep Hours', 0.0, 12.0, step=0.1)
    papers = st.slider('Sample Question Papers Practiced', 0, 10, step=1)
    submitted = st.form_submit_button("Predict")

# Prediksi
if submitted:
    input_df = pd.DataFrame({
        'Hours Studied': [hours],
        'Previous Scores': [previous_scores],
        'Extracurricular Activities': [1 if extra == 'Yes' else 0],
        'Sleep Hours': [sleep],
        'Sample Question Papers Practiced': [papers]
    })

    prediction = model.predict(input_df)[0]
    st.success(f"🎯 Prediksi Performance Index Siswa: **{prediction:.2f}**")
