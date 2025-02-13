# Recommendation-system

## **Marble Image Similarity Matching**

## **Overview**
This project helps in finding similar **marble tile images** from a dataset based on an uploaded image. It extracts image features and compares them using ORB descriptors and color histograms, recommending the **top 5 most similar images**.

## **How It Works**

1. **Feature Extraction**  
   - Uses **ORB (Oriented FAST and Rotated BRIEF)** to extract keypoint descriptors.  
   - Computes **color histograms** to capture color distribution.

2. **Similarity Calculation**  
   - Uses a **Brute-Force Matcher (BFMatcher)** to compare ORB keypoint descriptors.  
   - Uses **Cosine Similarity** to compare color histograms.  
   - Combines both metrics to compute a final similarity score.

3. **Recommendation System**  
   - Sorts dataset images based on similarity score.  
   - Displays the **top 5 most similar images** with similarity scores.

## **Project Structure**
```
/Marble-Similarity-Checker
│-- script.py               # Main Python script
│-- FIND THIS MARBLE/       # Folder containing the uploaded image
│   ├── onyx.jpg            # Image to find similar matches
│-- BACKUP MARBLE/          # Folder containing the dataset images
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── ...
│-- README.md               # Project documentation
```

## **Requirements**
Ensure you have the following Python libraries installed:
```sh
pip install opencv-python numpy scikit-learn matplotlib
```

## **Usage**
Run the script in your Python environment:
```sh
python script.py
```

## **Input & Output**
- **Input:** An image file located in `FIND THIS MARBLE/` folder.
- **Output:**
  - The script prints the **top 5 recommended images** with similarity scores.
  - Displays the images using **Matplotlib**.

## **Example Output**
```
Top 5 Recommended Images:
1: marble1.jpg with similarity score 85.40
2: marble2.jpg with similarity score 82.75
3: marble3.jpg with similarity score 78.32
4: marble4.jpg with similarity score 75.90
5: marble5.jpg with similarity score 73.45
```

## **License**
This project is open-source. Feel free to modify and improve it!

---

Let me know if you need any changes! 🚀

