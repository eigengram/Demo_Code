import streamlit as st
import numpy as np
import tensorflow as tf

# Load the model (Adjust the path as necessary)
MODEL_PATH = 'model'
model = tf.keras.models.load_model(MODEL_PATH)

def predict_heart_disease(model, input_features):
    input_dict = {name: tf.convert_to_tensor([value]) for name, value in input_features.items()}
    predictions = model.predict(input_dict)
    return predictions[0][0]

# Streamlit UI
st.title('Heart Disease Prediction App')

# Collecting user inputs for each feature
sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
cp = st.selectbox('Chest Pain Type', options=range(0, 5))
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
restecg = st.selectbox('Resting Electrocardiographic Results', options=range(0, 3))
exang = st.selectbox('Exercise Induced Angina', options=[0, 1])
ca = st.selectbox('Number of Major Vessels Colored by Flourosopy', options=range(0, 5))
thal = st.selectbox('Thalassemia', options=['normal', 'fixed defect', 'reversable defect'])
age = st.slider('Age', min_value=0, max_value=100, value=50, step=1)
trestbps = st.slider('Resting Blood Pressure', min_value=90, max_value=200, value=120, step=1)
chol = st.slider('Serum Cholestrol in mg/dl', min_value=100, max_value=600, value=240, step=1)
thalach = st.slider('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150, step=1)
oldpeak = st.slider('ST depression induced by exercise relative to rest', min_value=0.0, max_value=10.0, value=2.3, step=0.1)
slope = st.selectbox('The Slope of The Peak Exercise ST Segment', options=range(0, 3))

# Button to make prediction
if st.button('Predict Heart Disease'):
    input_features = {
        'sex': np.array([sex], dtype=np.int64),
        'cp': np.array([cp], dtype=np.int64),
        'fbs': np.array([fbs], dtype=np.int64),
        'restecg': np.array([restecg], dtype=np.int64),
        'exang': np.array([exang], dtype=np.int64),
        'ca': np.array([ca], dtype=np.int64),
        'thal': np.array([thal], dtype=np.object),
        'age': np.array([age], dtype=np.float32),
        'trestbps': np.array([trestbps], dtype=np.float32),
        'chol': np.array([chol], dtype=np.float32),
        'thalach': np.array([thalach], dtype=np.float32),
        'oldpeak': np.array([oldpeak], dtype=np.float32),
        'slope': np.array([slope], dtype=np.int64)
    }

    probability = predict_heart_disease(model, input_features)
    st.write(f'This particular patient has a {probability:.1%} probability of having heart disease.')
