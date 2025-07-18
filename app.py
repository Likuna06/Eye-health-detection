import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

# Paths
MODEL_PATH = "model.h5"
CLASS_INDEX_PATH = "class_indices.txt"

# --------------------------
# Load model and class names
# --------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    class_map = {}
    with open(CLASS_INDEX_PATH, "r", encoding="utf-8") as f:
        for line in f:
            idx, label = line.strip().split("\t")
            class_map[int(idx)] = label
    return class_map

model = load_model()
class_names = load_class_names()

# --------------------------
# App UI
# --------------------------
st.set_page_config(page_title="Eye Disease Detection", page_icon="👁", layout="centered")
st.title("👁 Eye Disease Detection")
st.write("Upload an eye image to predict the disease type.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    IMG_SIZE = (224, 224)
    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            preds = model.predict(img_array)
            pred_idx = np.argmax(preds)
            pred_label = class_names[pred_idx]
            confidence = preds[0][pred_idx] * 100

        st.success(f"✅ Predicted: *{pred_label}* ({confidence:.2f}% confidence)")
        st.progress(int(confidence))