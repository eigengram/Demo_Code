import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Address OpenMP runtime issue


import streamlit as st
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
import faiss
import os
from PIL import Image

# Replace experimental_singleton with cache_resource
@st.cache_resource
def load_model_and_processor():
    model = CLIPModel.from_pretrained("./clip_model")
    processor = CLIPProcessor.from_pretrained("./clip_model")
    return model, processor

model, processor = load_model_and_processor()

# Function to load embeddings and create a FAISS index
@st.experimental_singleton
def load_faiss_index(embeddings_folder):
    embeddings_files = [os.path.join(embeddings_folder, f) for f in os.listdir(embeddings_folder) if f.endswith('.npy')]
    embeddings = np.vstack([np.load(f) for f in embeddings_files])
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index, embeddings_files

embeddings_folder = './embeddings'  # Relative path to the embeddings folder
images_folder = './images/design-patent-images'  # Relative path to the images folder
index, embeddings_files = load_faiss_index(embeddings_folder)

def search_images(text_query, top_n=10):
    inputs = processor(text=text_query, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs).cpu().numpy()
    
    D, I = index.search(text_features, top_n)
    results = []
    for i, idx in enumerate(I[0]):
        embedding_file_name = os.path.basename(embeddings_files[idx])
        image_file_name = embedding_file_name.replace('.npy', '')
        # Assuming the images have the same name as embeddings but without .npy
        image_path = os.path.join(images_folder, image_file_name)
        distance = D[0][i]
        results.append((image_path, distance))
    return results

st.title('Image Search from Text')

text_query = st.text_input("Enter your search query:", "")
if text_query:
    st.write(f"Searching for: {text_query}")
    results = search_images(text_query, top_n=10)
    
    for image_path, distance in results:
        try:
            image = Image.open(image_path)
            st.image(image, caption=f"{os.path.basename(image_path)} (Distance: {distance:.2f})", use_column_width=True)
        except FileNotFoundError:
            st.error(f"Image not found: {image_path}")
