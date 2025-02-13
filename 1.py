import cv2
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Path to the uploaded marble image
UPLOAD_IMAGE_PATH = './FIND THIS MARBLE/onyx.jpg'

# Path to your dataset folder
DATASET_FOLDER = './BACKUP MARBLE'

# Function to extract features using ORB (Oriented FAST and Rotated BRIEF) and color histograms
def extract_features(image_path):
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_color = cv2.imread(image_path)

    # Extract ORB features
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image_gray, None)

    # Extract color histogram features
    hist = cv2.calcHist([image_color], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    return descriptors, hist

# Load the uploaded image features
upload_descriptors, upload_hist = extract_features(UPLOAD_IMAGE_PATH)

# Prepare dataset image features
dataset_features = {}
for filename in os.listdir(DATASET_FOLDER):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(DATASET_FOLDER, filename)
        descriptors, hist = extract_features(image_path)
        if descriptors is not None:
            dataset_features[filename] = (descriptors, hist)

# Function to calculate similarity between feature descriptors and color histograms
def calculate_similarity(upload_descriptors, dataset_descriptors, upload_hist, dataset_hist):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(upload_descriptors, dataset_descriptors)
    orb_similarity = len(matches)

    # Calculate color histogram similarity using cosine similarity
    hist_similarity = cosine_similarity([upload_hist], [dataset_hist])[0][0]

    # Combine ORB and color histogram similarity (weighted sum)
    combined_similarity = 0.5 * orb_similarity + 0.5 * hist_similarity * 100  # Scaling histogram similarity
    return combined_similarity

# Compare the uploaded image with dataset images
similarities = {}
for filename, (descriptors, hist) in dataset_features.items():
    similarity_score = calculate_similarity(upload_descriptors, descriptors, upload_hist, hist)
    similarities[filename] = similarity_score

# Sort images based on similarity (higher score = more similar)
recommended_images = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:5]

# Display the top 5 recommended images
print("Top 5 Recommended Images:")
plt.figure(figsize=(15, 5))
for idx, (filename, score) in enumerate(recommended_images):
    print(f"{idx + 1}: {filename} with similarity score {score:.2f}")

    # Display the images
    img = cv2.imread(os.path.join(DATASET_FOLDER, filename))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   
    plt.subplot(1, 5, idx + 1)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(f"{score:.2f}")

plt.show()