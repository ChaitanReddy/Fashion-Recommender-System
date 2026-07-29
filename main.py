import streamlit as st
import os
from PIL import Image, UnidentifiedImageError
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import io
import cv2

# --- Load embeddings ---
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# --- Load ResNet50 model ---
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

st.title("Fashion Recommender System")

if not os.path.exists("uploads"):
    os.makedirs("uploads")

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

uploaded_file = st.file_uploader("📤 Upload a clothing image (JPG/PNG/WebP)")

if uploaded_file is not None:
    try:
        # ✅ Read bytes & reset pointer
        uploaded_file.seek(0)
        image_bytes = uploaded_file.read()
        image_stream = io.BytesIO(image_bytes)

        try:
            # Try Pillow first
            img = Image.open(image_stream)
            img.verify()  # validate file
            img = Image.open(io.BytesIO(image_bytes))  # reopen for display
        except UnidentifiedImageError:
            # Fallback to OpenCV decode if Pillow fails
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img_cv is None:
                raise UnidentifiedImageError("OpenCV also failed to decode image.")
            img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Save a copy locally
        save_path = os.path.join("uploads", uploaded_file.name)
        img.save(save_path)

        # --- Feature extraction ---
        features = feature_extraction(save_path, model)

        # --- Get recommendations ---
        indices = recommend(features, feature_list)

        # --- Display recommendations ---
        st.subheader("🛍️ Recommended Similar Products:")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            with col:
                st.image(filenames[indices[0][i]], use_container_width=True)

    except Exception as e:
        st.error(f"❌ Invalid or corrupted image file: {e}")
