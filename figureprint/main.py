import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def match_orb_bf(img1_path, img2_path, is_fingerprint=True):
    print(f"Loading images: {img1_path}, {img2_path}")
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("WARNING: Error loading images.")
        return "Error", None

    if is_fingerprint:
        print("Applying fingerprint preprocessing (Otsu threshholding with inversion)")
        _, img1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    orb = cv2.ORB_create(nfeatures=1000)

    print("Finding keypoints and descriptors")
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    print(f"DEBUG: Keypoints detected: {len(kp1)} in img1, {len(kp2)} in img2")

    if des1 is None or des2 is None:
        print("WARNING: No descriptors found in one of the images.")
        return "No Match", None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test (keep only good matches)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    # Visualize

    threshold = 20 # Arbitrary, but from guide
    if len(good_matches) > threshold:
        print(f"Result: Match (good matches: {len(good_matches)} > {threshold})")
        return "Match", len(good_matches)
    else:
        print(f"Result: No Match (good matches: {len(good_matches)} <= {threshold})")
        return "No Match", len(good_matches)


def match_sift_flann(img1_path, img2_path, is_fingerprint=True):
    print(f"Loading images: {img1_path}, {img2_path}")
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print(f"WARNING: Error loading images.")
        return "Error", None

    if is_fingerprint:
        print("Applying fingerprint preprocessing (Otsu threshholding with inversion)")
        _, img1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    sift = cv2.SIFT_create(nfeatures=1000)

    print("Finding keypoints and descriptors")
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    print(f"DEBUG: Keypoints detected: {len(kp1)} in img1, {len(kp2)} in img2")
    if des1 is None or des2 is None:
        print("WARNING: No descriptors found in one of the images.")
        return "No Match", None
    
    # FLANN parameters
    index_params = dict(algorithm=1, trees=5) # Using KDTree
    search_params = dict(checks=50) # number of checks for nearest neightbor
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test (keep only good matches)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    threshold = 20  # Arbitrary, but from guide
    if len(good_matches) > threshold:
        print(f"Result: Match (good matches: {len(good_matches)} > {threshold})")
        return "Match", len(good_matches)
    else:
        print(f"Result: No Match (good matches: {len(good_matches)} <= {threshold})")
        return "No Match", len(good_matches)


def process_dataset(dataset_root='./data_check/', approaches=['orb_bf', 'sift_flann']):
    results = {approach: {'correct': 0, 'total': 0} for approach in approaches}

    for folder_type in ['same', 'different']:
        for i in range(1, 11):
            folder = f"{folder_type}_{i}"
            folder_path = os.path.join(dataset_root, folder)
            if not os.path.exists(folder_path):
                print(f"DEBUG: Folder {folder_path} does not exist. Skipping")
                continue

            tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
            if len(tif_files) != 2:
                print(f"DEBUG: Expected 2 .tif files in {folder}, found {len(tif_files)}. Skipping")
                continue

            img1_path = os.path.join(folder_path, tif_files[0])
            img2_path = os.path.join(folder_path, tif_files[1])

            expected = "Match" if folder_type == 'same' else "No Match"
            print(f"\nProcessing {folder}: Expected = {expected}")

            for approach in approaches:
                if approach == 'orb_bf':
                    result, _ = match_orb_bf(img1_path, img2_path, is_fingerprint=True)
                elif approach == 'sift_flann':
                    result, _ = match_sift_flann(img1_path, img2_path, is_fingerprint=True)
                
                if result == expected:
                    results[approach]['correct'] += 1
                results[approach]['total'] += 1

        for approach in approaches:
            accuracy = (results[approach]['correct'] / results[approach]['total']) * 100 if results[approach]['total'] > 0 else 0
            print(f"\n{approach.upper()} Accuracy: {accuracy:.2f}% ({results[approach]['correct']}/{results[approach]['total']})")


def process_UIA(uia_root='./UIA/'):
    img1_path = os.path.join(uia_root, 'front3.jpg')
    img2_path = os.path.join(uia_root, 'front1.png')

    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print("DEBUG: UIA images not found. Skipping UIA processing.")
        return

    print("\nProcessing UIA images (non-fingerprint, no preprocessing)")
    print("\nORB_BF Approach:")
    match_orb_bf(img1_path, img2_path, is_fingerprint=False)
    print("\nSIFT_FLANN Approach:")
    match_sift_flann(img1_path, img2_path, is_fingerprint=False)

if __name__ == "__main__":
    process_dataset()
    process_UIA()