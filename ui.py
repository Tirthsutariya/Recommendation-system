# # # # # import cv2
# # # # # import numpy as np
# # # # # import os
# # # # # import streamlit as st
# # # # # from sklearn.metrics.pairwise import cosine_similarity
# # # # # from PIL import Image

# # # # # # Path to your dataset folder
# # # # # DATASET_FOLDER = './BACKUP MARBLE'

# # # # # def extract_features(image_path):
# # # # #     image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# # # # #     image_color = cv2.imread(image_path)
    
# # # # #     if image_gray is None or image_color is None:
# # # # #         return None, None

# # # # #     # Extract ORB features
# # # # #     orb = cv2.ORB_create()
# # # # #     keypoints, descriptors = orb.detectAndCompute(image_gray, None)
    
# # # # #     # Extract color histogram features
# # # # #     hist = cv2.calcHist([image_color], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
# # # # #     hist = cv2.normalize(hist, hist).flatten()

# # # # #     return descriptors, hist

# # # # # def calculate_similarity(upload_descriptors, dataset_descriptors, upload_hist, dataset_hist):
# # # # #     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# # # # #     matches = bf.match(upload_descriptors, dataset_descriptors)
# # # # #     orb_similarity = len(matches)
    
# # # # #     # Calculate color histogram similarity using cosine similarity
# # # # #     hist_similarity = cosine_similarity([upload_hist], [dataset_hist])[0][0]
    
# # # # #     # Combine ORB and color histogram similarity (weighted sum)
# # # # #     combined_similarity = 0.5 * orb_similarity + 0.5 * hist_similarity * 100
# # # # #     return combined_similarity

# # # # # # Streamlit UI
# # # # # st.title("Marble Image Similarity Finder")
# # # # # uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# # # # # dataset_features = {}
# # # # # for filename in os.listdir(DATASET_FOLDER):
# # # # #     if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
# # # # #         image_path = os.path.join(DATASET_FOLDER, filename)
# # # # #         descriptors, hist = extract_features(image_path)
# # # # #         if descriptors is not None:
# # # # #             dataset_features[filename] = (descriptors, hist)

# # # # # if uploaded_file is not None:
# # # # #     image = Image.open(uploaded_file)
# # # # #     st.image(image, caption="Uploaded Image", use_column_width=True)
    
# # # # #     # Save uploaded image temporarily
# # # # #     upload_path = "temp_uploaded.jpg"
# # # # #     image.save(upload_path)
# # # # #     upload_descriptors, upload_hist = extract_features(upload_path)
    
# # # # #     if upload_descriptors is not None:
# # # # #         similarities = {}
# # # # #         for filename, (descriptors, hist) in dataset_features.items():
# # # # #             similarity_score = calculate_similarity(upload_descriptors, descriptors, upload_hist, hist)
# # # # #             similarities[filename] = similarity_score
        
# # # # #         recommended_images = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:5]
        
# # # # #         st.subheader("Top 5 Recommended Images:")
# # # # #         cols = st.columns(5)
# # # # #         for idx, (filename, score) in enumerate(recommended_images):
# # # # #             img_path = os.path.join(DATASET_FOLDER, filename)
# # # # #             img = Image.open(img_path)
# # # # #             with cols[idx]:
# # # # #                 st.image(img, caption=f"{score:.2f}", use_column_width=True)



# # # # # import cv2
# # # # # import numpy as np
# # # # # import os
# # # # # import streamlit as st
# # # # # from sklearn.metrics.pairwise import cosine_similarity
# # # # # from PIL import Image

# # # # # # Path to your dataset folder
# # # # # DATASET_FOLDER = './BACKUP MARBLE'
# # # # # IMAGE_DISPLAY_SIZE = (200, 200)  # Fixed display size for consistency

# # # # # st.set_page_config(page_title="Marble Image Similarity Finder", layout="wide")
# # # # # st.title("🟢 Marble Image Similarity Finder")
# # # # # st.markdown("Upload an image to find visually similar marble patterns.")

# # # # # @st.cache_data
# # # # # def extract_features(image_path):
# # # # #     image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# # # # #     image_color = cv2.imread(image_path)
    
# # # # #     if image_gray is None or image_color is None:
# # # # #         return None, None

# # # # #     # Extract ORB features
# # # # #     orb = cv2.ORB_create()
# # # # #     keypoints, descriptors = orb.detectAndCompute(image_gray, None)
    
# # # # #     # Extract color histogram features
# # # # #     hist = cv2.calcHist([image_color], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
# # # # #     hist = cv2.normalize(hist, hist).flatten()

# # # # #     return descriptors, hist

# # # # # @st.cache_data
# # # # # def load_dataset_features():
# # # # #     dataset_features = {}
# # # # #     for filename in os.listdir(DATASET_FOLDER):
# # # # #         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
# # # # #             image_path = os.path.join(DATASET_FOLDER, filename)
# # # # #             descriptors, hist = extract_features(image_path)
# # # # #             if descriptors is not None:
# # # # #                 dataset_features[filename] = (descriptors, hist)
# # # # #     return dataset_features

# # # # # def calculate_similarity(upload_descriptors, dataset_descriptors, upload_hist, dataset_hist):
# # # # #     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# # # # #     matches = bf.match(upload_descriptors, dataset_descriptors)
# # # # #     orb_similarity = len(matches)
    
# # # # #     # Calculate color histogram similarity using cosine similarity
# # # # #     hist_similarity = cosine_similarity([upload_hist], [dataset_hist])[0][0]
    
# # # # #     # Combine ORB and color histogram similarity (weighted sum)
# # # # #     combined_similarity = 0.5 * orb_similarity + 0.5 * hist_similarity * 100
# # # # #     return combined_similarity

# # # # # dataset_features = load_dataset_features()

# # # # # uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

# # # # # if uploaded_file is not None:
# # # # #     image = Image.open(uploaded_file)
# # # # #     st.image(image, caption="📷 Uploaded Image", use_column_width=False, width=300)
    
# # # # #     # Save uploaded image temporarily
# # # # #     upload_path = "temp_uploaded.jpg"
# # # # #     image.save(upload_path)
# # # # #     upload_descriptors, upload_hist = extract_features(upload_path)
    
# # # # #     if upload_descriptors is not None:
# # # # #         similarities = {}
# # # # #         for filename, (descriptors, hist) in dataset_features.items():
# # # # #             similarity_score = calculate_similarity(upload_descriptors, descriptors, upload_hist, hist)
# # # # #             similarities[filename] = similarity_score
        
# # # # #         recommended_images = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:6]
        
# # # # #         st.subheader("🎯 Top 6 Recommended Images:")
        
# # # # #         cols = st.columns(3)
# # # # #         for idx, (filename, score) in enumerate(recommended_images):
# # # # #             img_path = os.path.join(DATASET_FOLDER, filename)
# # # # #             img = Image.open(img_path).resize(IMAGE_DISPLAY_SIZE)
# # # # #             with cols[idx % 3]:
# # # # #                 st.image(img, caption=f"🔹 Score: {score:.2f}", use_column_width=False, width=200)


# # # # import cv2
# # # # import numpy as np
# # # # import os
# # # # import streamlit as st
# # # # from sklearn.metrics.pairwise import cosine_similarity
# # # # from PIL import Image

# # # # # Path to your dataset folder
# # # # DATASET_FOLDER = './BACKUP MARBLE'
# # # # IMAGE_DISPLAY_SIZE = (200, 200)  # Fixed display size for consistency

# # # # st.set_page_config(page_title="Marble Image Similarity Finder", layout="wide")
# # # # st.title("🟢 Marble Image Similarity Finder")
# # # # st.markdown("Upload an image to find visually similar marble patterns.")

# # # # @st.cache_data
# # # # def extract_features(image_path):
# # # #     image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# # # #     image_color = cv2.imread(image_path)
    
# # # #     if image_gray is None or image_color is None:
# # # #         return None, None

# # # #     # Extract ORB features
# # # #     orb = cv2.ORB_create()
# # # #     keypoints, descriptors = orb.detectAndCompute(image_gray, None)
    
# # # #     # Extract color histogram features
# # # #     hist = cv2.calcHist([image_color], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
# # # #     hist = cv2.normalize(hist, hist).flatten()

# # # #     return descriptors, hist

# # # # @st.cache_data
# # # # def load_dataset_features():
# # # #     dataset_features = {}
# # # #     for filename in os.listdir(DATASET_FOLDER):
# # # #         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
# # # #             image_path = os.path.join(DATASET_FOLDER, filename)
# # # #             descriptors, hist = extract_features(image_path)
# # # #             if descriptors is not None:
# # # #                 dataset_features[filename] = (descriptors, hist)
# # # #     return dataset_features

# # # # def calculate_similarity(upload_descriptors, dataset_descriptors, upload_hist, dataset_hist):
# # # #     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# # # #     matches = bf.match(upload_descriptors, dataset_descriptors)
# # # #     orb_similarity = len(matches)
    
# # # #     # Calculate color histogram similarity using cosine similarity
# # # #     hist_similarity = cosine_similarity([upload_hist], [dataset_hist])[0][0]
    
# # # #     # Combine ORB and color histogram similarity (weighted sum)
# # # #     combined_similarity = 0.5 * orb_similarity + 0.5 * hist_similarity * 100
# # # #     return combined_similarity

# # # # dataset_features = load_dataset_features()

# # # # uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

# # # # if uploaded_file is not None:
# # # #     image = Image.open(uploaded_file)
    
# # # #     # Resize the image to a smaller size
# # # #     resized_image = image.resize((50, 50))
    
# # # #     st.image(resized_image, caption="📷 Uploaded Image", use_container_width=True, width=50)
    
# # # #     # Save resized image temporarily
# # # #     upload_path = "temp_uploaded.jpg"
# # # #     resized_image.save(upload_path)
# # # #     upload_descriptors, upload_hist = extract_features(upload_path)
    
# # # #     if upload_descriptors is not None:
# # # #         similarities = {}
# # # #         for filename, (descriptors, hist) in dataset_features.items():
# # # #             similarity_score = calculate_similarity(upload_descriptors, descriptors, upload_hist, hist)
# # # #             similarities[filename] = similarity_score
        
# # # #         recommended_images = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:6]
        
# # # #         st.subheader("🎯 Top 6 Recommended Images:")
        
# # # #         cols = st.columns(3)
# # # #         for idx, (filename, score) in enumerate(recommended_images):
# # # #             img_path = os.path.join(DATASET_FOLDER, filename)
# # # #             img = Image.open(img_path).resize(IMAGE_DISPLAY_SIZE)
# # # #             with cols[idx % 3]:
# # # #                 st.image(img, caption=f"🔹 Score: {score:.2f}", use_container_width=True, width=200)


# # # import cv2
# # # import numpy as np
# # # import os
# # # import streamlit as st
# # # import faiss
# # # from sklearn.metrics.pairwise import cosine_similarity
# # # from PIL import Image

# # # # Path to your dataset folder
# # # DATASET_FOLDER = './BACKUP MARBLE'
# # # IMAGE_DISPLAY_SIZE = (200, 200)  # Fixed display size for consistency

# # # st.set_page_config(page_title="Marble Image Similarity Finder", layout="wide")
# # # st.title("🟢 Marble Image Similarity Finder")
# # # st.markdown("Upload an image to find visually similar marble patterns.")

# # # # Helper function to extract ORB features and color histogram
# # # def extract_features(image_path):
# # #     image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# # #     image_color = cv2.imread(image_path)
    
# # #     if image_gray is None or image_color is None:
# # #         return None, None

# # #     # Extract ORB features
# # #     orb = cv2.ORB_create()
# # #     keypoints, descriptors = orb.detectAndCompute(image_gray, None)
    
# # #     # Extract color histogram features
# # #     hist = cv2.calcHist([image_color], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
# # #     hist = cv2.normalize(hist, hist).flatten()

# # #     return descriptors, hist

# # # # Build FAISS index for descriptors
# # # @st.cache_data
# # # def build_faiss_index():
# # #     dataset_features = []
# # #     filenames = []
    
# # #     # Iterate through dataset images to extract features and store filenames
# # #     for filename in os.listdir(DATASET_FOLDER):
# # #         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
# # #             image_path = os.path.join(DATASET_FOLDER, filename)
# # #             descriptors, hist = extract_features(image_path)
# # #             if descriptors is not None:
# # #                 dataset_features.append(descriptors.flatten())
# # #                 filenames.append(filename)

# # #     # Convert list of descriptors to numpy array
# # #     dataset_features = np.array(dataset_features).astype('float32')

# # #     # Create FAISS index (L2 distance index)
# # #     index = faiss.IndexFlatL2(dataset_features.shape[1])
# # #     index.add(dataset_features)
    
# # #     return index, filenames

# # # # Calculate similarity with color histogram
# # # def calculate_hist_similarity(upload_hist, dataset_hist):
# # #     hist_similarity = cosine_similarity([upload_hist], [dataset_hist])[0][0]
# # #     return hist_similarity

# # # # Load FAISS index and filenames
# # # index, filenames = build_faiss_index()

# # # uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

# # # if uploaded_file is not None:
# # #     image = Image.open(uploaded_file)
    
# # #     # Resize the image to a smaller size
# # #     resized_image = image.resize((50, 50))
    
# # #     st.image(resized_image, caption="📷 Uploaded Image", use_container_width=True, width=50)
    
# # #     # Save resized image temporarily
# # #     upload_path = "temp_uploaded.jpg"
# # #     resized_image.save(upload_path)
# # #     upload_descriptors, upload_hist = extract_features(upload_path)
    
# # #     if upload_descriptors is not None:
# # #         # Search for similar images using FAISS index
# # #         upload_descriptors = upload_descriptors.flatten().astype('float32').reshape(1, -1)
# # #         _, indices = index.search(upload_descriptors, k=6)  # Top 6 similar images
        
# # #         # Get the recommended images and their scores
# # #         recommended_images = []
# # #         for idx in indices[0]:
# # #             img_path = os.path.join(DATASET_FOLDER, filenames[idx])
# # #             img = Image.open(img_path).resize(IMAGE_DISPLAY_SIZE)
# # #             # Calculate histogram similarity for each image
# # #             _, dataset_hist = extract_features(img_path)
# # #             hist_similarity = calculate_hist_similarity(upload_hist, dataset_hist)
# # #             recommended_images.append((img, hist_similarity))
        
# # #         # Display the top recommended images with their similarity scores
# # #         st.subheader("🎯 Top 6 Recommended Images:")
        
# # #         cols = st.columns(3)
# # #         for idx, (img, score) in enumerate(recommended_images):
# # #             with cols[idx % 3]:
# # #                 st.image(img, caption=f"🔹 Score: {score:.2f}", use_container_width=True, width=200)


# # import cv2
# # import numpy as np
# # import os
# # import streamlit as st
# # import faiss
# # from sklearn.metrics.pairwise import cosine_similarity
# # from PIL import Image

# # # Path to your dataset folder
# # DATASET_FOLDER = './BACKUP MARBLE'
# # IMAGE_DISPLAY_SIZE = (200, 200)  # Fixed display size for consistency

# # st.set_page_config(page_title="Marble Image Similarity Finder", layout="wide")
# # st.title("🟢 Marble Image Similarity Finder")
# # st.markdown("Upload an image to find visually similar marble patterns.")

# # # Fixed descriptor length
# # DESCRIPTOR_LENGTH = 500

# # # Helper function to extract ORB features and color histogram
# # def extract_features(image_path):
# #     image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# #     image_color = cv2.imread(image_path)
    
# #     if image_gray is None or image_color is None:
# #         return None, None

# #     # Extract ORB features
# #     orb = cv2.ORB_create()
# #     keypoints, descriptors = orb.detectAndCompute(image_gray, None)
    
# #     # Extract color histogram features
# #     hist = cv2.calcHist([image_color], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
# #     hist = cv2.normalize(hist, hist).flatten()

# #     if descriptors is not None:
# #         # Pad or truncate descriptors to a fixed length
# #         if descriptors.shape[0] < DESCRIPTOR_LENGTH:
# #             # Pad with zeros
# #             padding = np.zeros((DESCRIPTOR_LENGTH - descriptors.shape[0], descriptors.shape[1]), dtype=np.float32)
# #             descriptors = np.vstack([descriptors, padding])
# #         else:
# #             # Truncate to the fixed length
# #             descriptors = descriptors[:DESCRIPTOR_LENGTH]
    
# #     return descriptors, hist

# # # Build FAISS index for descriptors
# # @st.cache_data
# # def build_faiss_index():
# #     dataset_features = []
# #     filenames = []
    
# #     # Iterate through dataset images to extract features and store filenames
# #     for filename in os.listdir(DATASET_FOLDER):
# #         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
# #             image_path = os.path.join(DATASET_FOLDER, filename)
# #             descriptors, hist = extract_features(image_path)
# #             if descriptors is not None:
# #                 dataset_features.append(descriptors.flatten())
# #                 filenames.append(filename)

# #     # Convert list of descriptors to numpy array
# #     dataset_features = np.array(dataset_features).astype('float32')

# #     # Create FAISS index (L2 distance index)
# #     index = faiss.IndexFlatL2(dataset_features.shape[1])
# #     index.add(dataset_features)
    
# #     return index, filenames

# # # Calculate similarity with color histogram
# # def calculate_hist_similarity(upload_hist, dataset_hist):
# #     hist_similarity = cosine_similarity([upload_hist], [dataset_hist])[0][0]
# #     return hist_similarity

# # # Load FAISS index and filenames
# # index, filenames = build_faiss_index()

# # uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

# # if uploaded_file is not None:
# #     image = Image.open(uploaded_file)
    
# #     # Resize the image to a smaller size
# #     resized_image = image.resize((50, 50))
    
# #     st.image(resized_image, caption="📷 Uploaded Image", use_container_width=True, width=50)
    
# #     # Save resized image temporarily
# #     upload_path = "temp_uploaded.jpg"
# #     resized_image.save(upload_path)
# #     upload_descriptors, upload_hist = extract_features(upload_path)
    
# #     if upload_descriptors is not None:
# #         # Search for similar images using FAISS index
# #         upload_descriptors = upload_descriptors.flatten().astype('float32').reshape(1, -1)
# #         _, indices = index.search(upload_descriptors, k=6)  # Top 6 similar images
        
# #         # Get the recommended images and their scores
# #         recommended_images = []
# #         for idx in indices[0]:
# #             img_path = os.path.join(DATASET_FOLDER, filenames[idx])
# #             img = Image.open(img_path).resize(IMAGE_DISPLAY_SIZE)
# #             # Calculate histogram similarity for each image
# #             _, dataset_hist = extract_features(img_path)
# #             hist_similarity = calculate_hist_similarity(upload_hist, dataset_hist)
# #             recommended_images.append((img, hist_similarity))
        
# #         # Display the top recommended images with their similarity scores
# #         st.subheader("🎯 Top 6 Recommended Images:")
        
# #         cols = st.columns(3)
# #         for idx, (img, score) in enumerate(recommended_images):
# #             with cols[idx % 3]:
# #                 st.image(img, caption=f"🔹 Score: {score:.2f}", use_container_width=True, width=200)


# import cv2
# import numpy as np
# import os
# import streamlit as st
# import faiss
# from sklearn.metrics.pairwise import cosine_similarity
# from PIL import Image

# # Path to your dataset folder
# DATASET_FOLDER = './BACKUP MARBLE'
# IMAGE_DISPLAY_SIZE = (200, 200)  # Display size for images

# st.set_page_config(page_title="Marble Image Similarity Finder", layout="wide")
# st.title("🟢 Marble Image Similarity Finder")
# st.markdown("Upload an image to find visually similar marble patterns.")

# # ORB feature descriptor settings
# DESCRIPTOR_LENGTH = 500
# ORB = cv2.ORB_create()

# # Helper function to extract ORB features and color histogram
# def extract_features(image_path):
#     image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     image_color = cv2.imread(image_path)
    
#     if image_gray is None or image_color is None:
#         return None, None

#     # Extract ORB features
#     keypoints, descriptors = ORB.detectAndCompute(image_gray, None)
    
#     # Ensure descriptors are not None
#     if descriptors is None or len(descriptors) == 0:
#         descriptors = np.zeros((DESCRIPTOR_LENGTH, 32), dtype=np.float32)  # Default zero array
#     else:
#         # Pad or truncate to match DESCRIPTOR_LENGTH
#         if descriptors.shape[0] < DESCRIPTOR_LENGTH:
#             padding = np.zeros((DESCRIPTOR_LENGTH - descriptors.shape[0], descriptors.shape[1]), dtype=np.float32)
#             descriptors = np.vstack([descriptors, padding])
#         else:
#             descriptors = descriptors[:DESCRIPTOR_LENGTH]

#     # Extract color histogram
#     hist = cv2.calcHist([image_color], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#     hist = cv2.normalize(hist, hist).flatten()

#     return descriptors, hist

# # Build FAISS index for dataset
# @st.cache_data
# def build_faiss_index():
#     dataset_features = []
#     dataset_histograms = []
#     filenames = []
    
#     for filename in os.listdir(DATASET_FOLDER):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             image_path = os.path.join(DATASET_FOLDER, filename)
#             descriptors, hist = extract_features(image_path)
            
#             if descriptors is not None:
#                 dataset_features.append(descriptors.flatten())
#                 dataset_histograms.append(hist)
#                 filenames.append(filename)

#     if len(dataset_features) == 0:
#         st.error("No valid images found in the dataset. Please check your dataset folder.")
#         return None, None, None

#     dataset_features = np.array(dataset_features).astype('float32')
    
#     # Create FAISS index
#     index = faiss.IndexFlatL2(dataset_features.shape[1])
#     index.add(dataset_features)
    
#     return index, dataset_histograms, filenames

# # Calculate color histogram similarity
# def calculate_hist_similarity(upload_hist, dataset_hist):
#     if upload_hist is None or dataset_hist is None:
#         return 0
#     return cosine_similarity([upload_hist], [dataset_hist])[0][0]

# # Load FAISS index
# index, dataset_histograms, filenames = build_faiss_index()

# # Upload image
# uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None and index is not None:
#     image = Image.open(uploaded_file)
    
#     # Resize uploaded image for display
#     resized_image = image.resize(IMAGE_DISPLAY_SIZE)
#     st.image(resized_image, caption="📷 Uploaded Image", use_column_width=True)
    
#     # Save uploaded image temporarily
#     upload_path = "temp_uploaded.jpg"
#     image.save(upload_path)
#     upload_descriptors, upload_hist = extract_features(upload_path)
    
#     if upload_descriptors is not None:
#         upload_descriptors = upload_descriptors.flatten().astype('float32').reshape(1, -1)

#         # Search for top similar images
#         _, indices = index.search(upload_descriptors, k=min(6, len(filenames)))

#         # Display results
#         st.subheader("🎯 Top Recommended Marble Patterns")
#         cols = st.columns(3)
        
#         for idx, dataset_idx in enumerate(indices[0]):
#             if dataset_idx >= len(filenames):
#                 continue  # Skip if index is out of range

#             img_path = os.path.join(DATASET_FOLDER, filenames[dataset_idx])
#             img = Image.open(img_path).resize(IMAGE_DISPLAY_SIZE)
            
#             # Compute histogram similarity
#             hist_similarity = calculate_hist_similarity(upload_hist, dataset_histograms[dataset_idx])
            
#             # Display image with score
#             with cols[idx % 3]:
#                 st.image(img, caption=f"🔹 Score: {hist_similarity:.2f}", use_column_width=True)


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

        # Search for top similar images
        _, indices = index.search(upload_descriptors, k=min(6, len(filenames)))

        st.subheader("🎯 Top Recommended Marble Patterns")
        cols = st.columns(3)
        
        for idx, dataset_idx in enumerate(indices[0]):
            if dataset_idx >= len(filenames):
                continue

            img_path = os.path.join(DATASET_FOLDER, filenames[dataset_idx])
            img = Image.open(img_path).resize(IMAGE_DISPLAY_SIZE)

            hist_similarity = calculate_hist_similarity(upload_hist, dataset_histograms[dataset_idx])

            with cols[idx % 3]:
                st.image(img, caption=f"🔹 Score: {hist_similarity:.2f}", width=200)
