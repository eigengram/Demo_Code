import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os

# Path to the trained model
model_path = 'breast_cancer_ultrasound_model/model.h5'

# Load the trained model
model = tf.keras.models.load_model(model_path)

def predict(image):
    # Preprocess the image to get it into the right format for the model
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)

    # Make prediction
    prediction = model.predict(image)
    return prediction[0][0]

# Directory for saving uploaded images
save_images_dir = "uploaded_images"

# Check if the directory exists, if not, create it
if not os.path.exists(save_images_dir):
    os.makedirs(save_images_dir)

# Streamlit web interface
st.title("Breast Cancer Detection from Ultrasound Images")
st.write("""
This application predicts whether breast cancer is present in ultrasound images. 
Please upload your ultrasound image using the upload area below.
""")

# File uploader for the user to add their own image
uploaded_file = st.file_uploader("Upload an ultrasound image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Uncomment the lines below to enable saving of uploaded images
    # with open(os.path.join(save_images_dir, uploaded_file.name), "wb") as f:
    #     f.write(uploaded_file.getbuffer())

    if st.button('Predict'):
        prediction = predict(image)

        if prediction >= 0.5:
            st.write(f"Prediction: Malignant with a probability of {prediction:.2f}")
        else:
            st.write(f"Prediction: Benign with a probability of {1 - prediction:.2f}")
