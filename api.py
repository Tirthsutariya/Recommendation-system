import os
import cv2
import numpy as np
import faiss
import pickle
import logging
import uvicorn
from fastapi import FastAPI, File, UploadFile
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
DATASET_FOLDER = './BACKUP MARBLE'
INDEX_FILE = 'faiss_index.bin'
FILENAMES_FILE = 'filenames.pkl'
HISTOGRAMS_FILE = 'histograms.pkl'

# ORB feature descriptor settings
DESCRIPTOR_LENGTH = 500
ORB = cv2.ORB_create()

# FastAPI app
app = FastAPI()

def extract_features(image: np.ndarray):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = ORB.detectAndCompute(image_gray, None)
    
    if descriptors is None or len(descriptors) == 0:
        descriptors = np.zeros((DESCRIPTOR_LENGTH, 32), dtype=np.float32)
    else:
        if descriptors.shape[0] < DESCRIPTOR_LENGTH:
            padding = np.zeros((DESCRIPTOR_LENGTH - descriptors.shape[0], descriptors.shape[1]), dtype=np.float32)
            descriptors = np.vstack([descriptors, padding])
        else:
            descriptors = descriptors[:DESCRIPTOR_LENGTH]
    
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    return descriptors, hist

def build_faiss_index():
    dataset_features = []
    dataset_histograms = []
    filenames = []
    
    for filename in os.listdir(DATASET_FOLDER):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(DATASET_FOLDER, filename)
            image = cv2.imread(image_path)
            
            if image is None:
                continue
            
            descriptors, hist = extract_features(image)
            dataset_features.append(descriptors.flatten())
            dataset_histograms.append(hist)
            filenames.append(filename)
    
    if not dataset_features:
        logging.error("No valid images found in the dataset.")
        return None, None, None
    
    dataset_features = np.array(dataset_features).astype('float32')
    index = faiss.IndexFlatL2(dataset_features.shape[1])
    index.add(dataset_features)
    faiss.write_index(index, INDEX_FILE)
    
    with open(FILENAMES_FILE, 'wb') as f:
        pickle.dump(filenames, f)
    with open(HISTOGRAMS_FILE, 'wb') as f:
        pickle.dump(dataset_histograms, f)
    
    return index, dataset_histograms, filenames

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

def calculate_hist_similarity(upload_hist, dataset_hist):
    if upload_hist is None or dataset_hist is None:
        return 0
    return cosine_similarity([upload_hist], [dataset_hist])[0][0]

# Load FAISS index
index, dataset_histograms, filenames = load_faiss_index()

@app.post("/search/")
async def search_similar_images(file: UploadFile = File(...)):
    if index is None:
        logging.error("FAISS index is not loaded.")
        return {"error": "Index not loaded"}
    
    image = Image.open(BytesIO(await file.read()))
    image = np.array(image.convert('RGB'))[:, :, ::-1]  # Convert PIL to OpenCV BGR
    descriptors, upload_hist = extract_features(image)
    
    if descriptors is None:
        return {"error": "Could not extract features"}
    
    descriptors = descriptors.flatten().astype('float32').reshape(1, -1)
    _, indices = index.search(descriptors, k=min(10, len(filenames)))
    results = []
    
    for dataset_idx in indices[0]:
        if dataset_idx >= len(filenames):
            continue
        hist_similarity = calculate_hist_similarity(upload_hist, dataset_histograms[dataset_idx])
        results.append({"filename": filenames[dataset_idx], "score": float(hist_similarity)})
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return {"matches": results[:6]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)