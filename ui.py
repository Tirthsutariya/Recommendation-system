import cv2
import numpy as np
import os
import streamlit as st
import faiss
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Paths
DATASET_FOLDER = './BACKUP MARBLE'
INDEX_FILE = 'faiss_index.bin'
FILENAMES_FILE = 'filenames.pkl'
HISTOGRAMS_FILE = 'histograms.pkl'

# Image display size
IMAGE_DISPLAY_SIZE = (200, 200)

st.set_page_config(page_title="Marble Image Similarity Finder", layout="wide")
st.title("🟢 Marble Image Similarity Finder")
st.markdown("Upload an image to find visually similar marble patterns.")

# ORB feature descriptor settings
DESCRIPTOR_LENGTH = 500
ORB = cv2.ORB_create()

# Feature extraction function
def extract_features(image_path):
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_color = cv2.imread(image_path)

    if image_gray is None or image_color is None:
        return None, None

    keypoints, descriptors = ORB.detectAndCompute(image_gray, None)

    if descriptors is None or len(descriptors) == 0:
        descriptors = np.zeros((DESCRIPTOR_LENGTH, 32), dtype=np.float32)
    else:
        if descriptors.shape[0] < DESCRIPTOR_LENGTH:
            padding = np.zeros((DESCRIPTOR_LENGTH - descriptors.shape[0], descriptors.shape[1]), dtype=np.float32)
            descriptors = np.vstack([descriptors, padding])
        else:
            descriptors = descriptors[:DESCRIPTOR_LENGTH]

    hist = cv2.calcHist([image_color], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    return descriptors, hist

# Compute and store FAISS index
def build_faiss_index():
    dataset_features = []
    dataset_histograms = []
    filenames = []

    for filename in os.listdir(DATASET_FOLDER):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(DATASET_FOLDER, filename)
            descriptors, hist = extract_features(image_path)

            if descriptors is not None:
                dataset_features.append(descriptors.flatten())
                dataset_histograms.append(hist)
                filenames.append(filename)

    if len(dataset_features) == 0:
        st.error("No valid images found in the dataset. Please check your dataset folder.")
        return None, None, None

    dataset_features = np.array(dataset_features).astype('float32')

    # Create and save FAISS index
    index = faiss.IndexFlatL2(dataset_features.shape[1])
    index.add(dataset_features)

    faiss.write_index(index, INDEX_FILE)

    # Save filenames and histograms
    with open(FILENAMES_FILE, 'wb') as f:
        pickle.dump(filenames, f)
    with open(HISTOGRAMS_FILE, 'wb') as f:
        pickle.dump(dataset_histograms, f)

    return index, dataset_histograms, filenames

# Load FAISS index if it exists
def load_faiss_index():
    if os.path.exists(INDEX_FILE) and os.path.exists(FILENAMES_FILE) and os.path.exists(HISTOGRAMS_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(FILENAMES_FILE, 'rb') as f:
            filenames = pickle.load(f)
        with open(HISTOGRAMS_FILE, 'rb') as f:
            dataset_histograms = pickle.load(f)
        return index, dataset_histograms, filenames
    else:
        return build_faiss_index()

# Calculate histogram similarity
def calculate_hist_similarity(upload_hist, dataset_hist):
    if upload_hist is None or dataset_hist is None:
        return 0
    return cosine_similarity([upload_hist], [dataset_hist])[0][0]

# Load FAISS index
index, dataset_histograms, filenames = load_faiss_index()

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and index is not None:
    image = Image.open(uploaded_file)
    
    resized_image = image.resize(IMAGE_DISPLAY_SIZE)
    resized_image = image.resize((200, 200))
    st.image(resized_image, caption="📷 Uploaded Image", width=200)
    
    # Save uploaded image temporarily
    upload_path = "temp_uploaded.jpg"
    image.save(upload_path)
    upload_descriptors, upload_hist = extract_features(upload_path)
    
    if upload_descriptors is not None:
        upload_descriptors = upload_descriptors.flatten().astype('float32').reshape(1, -1)

        # Retrieve top 10 matches from FAISS
        _, indices = index.search(upload_descriptors, k=min(10, len(filenames)))

        # Compute histogram similarities
        results = []
        for dataset_idx in indices[0]:
            if dataset_idx >= len(filenames):
                continue

            hist_similarity = calculate_hist_similarity(upload_hist, dataset_histograms[dataset_idx])
            results.append((dataset_idx, hist_similarity))

        # Sort by histogram similarity (descending)
        results = sorted(results, key=lambda x: x[1], reverse=True)

        # Display the most similar images
        st.subheader("🎯 Top Recommended Marble Patterns")
        cols = st.columns(3)
        
        for idx, (dataset_idx, hist_similarity) in enumerate(results[:6]):  # Display top 6 sorted results
            img_path = os.path.join(DATASET_FOLDER, filenames[dataset_idx])
            img = Image.open(img_path).resize(IMAGE_DISPLAY_SIZE)

            with cols[idx % 3]:
                st.image(img, caption=f"🔹 Score: {hist_similarity:.2f}", width=200)