import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
import pickle
import os
from PIL import Image
import matplotlib.pyplot as plt

# Earlier - directly in colab
# Initialize the ResNet50 model
#model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Older
# Initialize the ResNet50 model with caching to avoid reloading weights on every call
@st.cache(allow_output_mutation=True)
def get_model():
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    return model

# New
#@st.experimental_memo  # Updated based on new Streamlit caching
#def get_model():
    #model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    #return model

model = get_model()





# New - by manually downlaoding it
# Initialize the model with no weights
#model = ResNet50(include_top=False, pooling='avg', weights=None)

# Path to your weights file
#weights_path = r"C:\Users\Sinha\All_Python_Codes\streamlit_applications\reverse_image_search_patents\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

# Load the weights
#model.load_weights(weights_path)


def extract_features(img, model):
    """Extract features from an image using ResNet50."""
    img = img.convert('RGB')  # Ensure image is in RGB format
    img = img.resize((224, 224))  # Resize image to match model expected size
    img_array = img_to_array(img)  # Convert the image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image
    features = model.predict(img_array)  # Predict to extract features
    return features.flatten()

def load_feature_vectors(vector_folder):
    """Load feature vectors from the specified folder."""
    vectors = []
    paths = []
    for filename in os.listdir(vector_folder):
        if filename.endswith('.pkl'):
            with open(os.path.join(vector_folder, filename), 'rb') as f:
                path, vector = pickle.load(f)
                paths.append(path)
                vectors.append(vector)
    return paths, np.array(vectors)


# Update these paths to the relative paths
vector_folder = "./vectors"  # Relative path to the vectors folder
images_folder = "./images/design-patent-images"  # Relative path to the images folder


# Specify your vector folder and images folder path here
#vector_folder = r"C:\Users\Sinha\All_Python_Codes\streamlit_applications\reverse_image_search_patents\design-patent-images-vectordb"
#images_folder = r"C:\Users\Sinha\All_Python_Codes\streamlit_applications\reverse_image_search_patents\design-patent-images"

image_paths, features_array = load_feature_vectors(vector_folder)
nn_model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
nn_model.fit(features_array)

st.title('Reverse Image Search')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    try:
        uploaded_image = Image.open(uploaded_file)
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
        query_features = extract_features(uploaded_image, model)
        distances, indices = nn_model.kneighbors([query_features])
        
        for i, idx in enumerate(indices[0]):
            filename = os.path.basename(image_paths[idx])
            distance = distances[0][i]  # Use distance directly
            local_image_path = os.path.join(images_folder, filename)
            if os.path.exists(local_image_path):
                st.image(local_image_path, caption=f"{filename} (Distance: {distance:.2f})", use_column_width=True)
            else:
                st.error(f"Image {filename} not found in the images folder.")
    except Exception as e:
        st.error(f"An error occurred: {e}")