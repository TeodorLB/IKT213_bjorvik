import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def match_orb_bf(img1_path, img2_path, is_fingerprint=True, display_inline=False):
    print(f"Loading images: {img1_path}, {img2_path}")
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("WARNING: Error loading images.")
        return "Error", 0, None

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
        return "No Match", 0, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test (keep only good matches)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    # Visualize: Draw matches
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, singlePointColor=(0,255,0), matchColor=(255,0,0))

    if display_inline and match_img is not None:
        plt.figure(figsize=(10,5))
        plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
        plt.title(f"ORB_BF Good matches: {len(good_matches)}")
        plt.axis('off')
        plt.show()

    threshold = 20 # Arbitrary, but from guide
    if len(good_matches) > threshold:
        print(f"Result: Match (good matches: {len(good_matches)} > {threshold})")
        return "Match", len(good_matches), match_img
    else:
        print(f"Result: No Match (good matches: {len(good_matches)} <= {threshold})")
        return "No Match", len(good_matches), match_img


def match_sift_flann(img1_path, img2_path, is_fingerprint=True, display_inline=False):
    print(f"Loading images: {img1_path}, {img2_path}")
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print(f"WARNING: Error loading images.")
        return "Error", 0, None

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
        return "No Match", 0, None
    
    # FLANN parameters
    index_params = dict(algorithm=1, trees=5) # Using KDTree
    search_params = dict(checks=50) # number of checks for nearest neightbor
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test (keep only good matches)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, singlePointColor=(0,255,0), matchColor=(255,0,0))

    if display_inline and match_img is not None:
        plt.figure(figsize=(10,5))
        plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
        plt.title(f"SIFT_FLANN Good matches: {len(good_matches)}")
        plt.axis('off')
        plt.show()

    threshold = 20  # Arbitrary, but from guide
    if len(good_matches) > threshold:
        print(f"Result: Match (good matches: {len(good_matches)} > {threshold})")
        return "Match", len(good_matches), match_img
    else:
        print(f"Result: No Match (good matches: {len(good_matches)} <= {threshold})")
        return "No Match", len(good_matches), match_img


def process_dataset(dataset_root='./data_check/', approaches=['orb_bf', 'sift_flann'], display_inline=False, results_folder='./results/'):
    os.makedirs(results_folder, exist_ok=True)
    results = {approach: {'correct': 0, 'total': 0, 'y_true': [], 'y_pred': [], 'match_counts': {}} for approach in approaches}

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
            expected_label = 1 if expected == "Match" else 0 # For confusion matrix
            print(f"\nProcessing {folder}: Expected = {expected}")

            for approach in approaches:
                if approach == 'orb_bf':
                    result, count, match_img = match_orb_bf(img1_path, img2_path, is_fingerprint=True, display_inline=display_inline)
                elif approach == 'sift_flann':
                    result, count, match_img = match_sift_flann(img1_path, img2_path, is_fingerprint=True, display_inline=display_inline)
                
                predicted_label = 1 if result == "Match" else 0
                if result == expected:
                    results[approach]['correct'] += 1
                results[approach]['total'] += 1
                results[approach]['y_true'].append(expected_label)
                results[approach]['y_pred'].append(predicted_label)
                results[approach]['match_counts'][folder] = count
                
                # Save match image
                if match_img is not None:
                    result_str = "match" if result == "Match" else "no_match"
                    match_img_filename = f"{folder}_{approach}_{result_str}.png"
                    match_img_path = os.path.join(results_folder, match_img_filename)
                    cv2.imwrite(match_img_path, match_img)
                    print(f"DEBUG: {approach} Match image saved to {match_img_path}")

        for approach in approaches:
            accuracy = (results[approach]['correct'] / results[approach]['total']) * 100 if results[approach]['total'] > 0 else 0
            print(f"\n{approach.upper()} Accuracy: {accuracy:.2f}% ({results[approach]['correct']}/{results[approach]['total']})")

            # Confusion Matrix
            if results[approach]['y_true']:
                labels = ["No Match (0)", "Match (1)"]
                cm = confusion_matrix(results[approach]['y_true'], results[approach]['y_pred'])
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
                fig, ax = plt.subplots(figsize=(6,5))
                disp.plot(cmap="Blues", values_format="d", ax=ax)
                plt.title(f"Confusion Matrix {approach.upper()}")

                cm_filename = f"confusion_matrix_{approach}.png"
                cm_path = os.path.join(results_folder, cm_filename)
                fig.savefig(cm_path)
                print(f"DEBUG: {approach} Confusion matrix saved to {cm_path}")
                plt.show()

            # Match counts plot
            if results[approach]['match_counts']:
                plt.figure(figsize=(10,5))
                folders = sorted(results[approach]['match_counts'].keys())
                counts = [results[approach]["match_counts"][f] for f in folders]
                colors = ['green' if "same" in f.lower() else 'red' for f in folders]
                plt.bar(folders, counts, color=colors)
                plt.axhline(y=20, color='black', linestyle='--', label='Threshold (20)')
                plt.xticks(rotation=45)
                plt.ylabel('Number of Good Matches')
                plt.title(f'Match Counts per Folder ({approach.upper()}) (Green=Same, Red=Different)')
                plt.legend()
            
                tuning_filename = f"match_counts_{approach}.png"
                tuning_path = os.path.join(results_folder, tuning_filename)
                plt.savefig(tuning_path)
                print(f"DEBUG: {approach} Match counts plot saved to {tuning_path}")
                plt.show()
                
    return results # For markdown code

def process_UIA(uia_root='./UIA/', display_inline=False, results_folder='./results/'):
    img1_path = os.path.join(uia_root, 'front3.jpg')
    img2_path = os.path.join(uia_root, 'front1.png')

    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print("DEBUG: UIA images not found. Skipping UIA processing.")
        return

    print("\nProcessing UIA images (non-fingerprint, no preprocessing)")
    uia_results = {}

    print("\nORB_BF Approach:")
    result_orb, count_orb, match_img_orb = match_orb_bf(img1_path, img2_path, is_fingerprint=False, display_inline=display_inline)
    uia_results['orb_bf'] = {'result': result_orb, 'count': count_orb}
    if match_img_orb is not None:
        result_str = "match" if result_orb == "Match" else "no_match"
        match_img_filename = f"UIA_orb_bf_{result_str}.png"
        match_img_path = os.path.join(results_folder, match_img_filename)
        cv2.imwrite(match_img_path, match_img_orb)
        print(f"DEBUG: orb_bf UIA Match image saved to {match_img_path}")
    print("\nSIFT_FLANN Approach:")
    result_sift, count_sift, match_img_sift = match_sift_flann(img1_path, img2_path, is_fingerprint=False, display_inline=display_inline)
    uia_results['sift_flann'] = {'result': result_sift, 'count': count_sift}
    if match_img_sift is not None:
        result_str = "match" if result_sift == "Match" else "no_match"
        match_img_filename = f"UIA_sift_flann_{result_str}.png"
        match_img_path = os.path.join(results_folder, match_img_filename)
        cv2.imwrite(match_img_path, match_img_sift)
        print(f"DEBUG: sift_flann UIA Match image saved to {match_img_path}")

def generate_markdown_report(results_folder, dataset_results, uia_results, approaches=['orb_bf', 'sift_flann'], threshold=20):
    report_path = os.path.join(results_folder, 'report.md')
    with open(report_path, 'w') as f:
        f.write("# Image Matching Report (ORB_BF and SIFT_FLANN)\n\n")
        f.write(f"**Threshold Used**: {threshold}\n\n")

        for approach in approaches:
            f.write(f"## {approach.upper()} Approach\n\n")
            if dataset_results[approach]['total'] > 0:
                accuracy = (dataset_results[approach]['correct'] / dataset_results[approach]['total']) * 100
                f.write(f"- **Data_Check Accuracy**: {accuracy:.2f}% ({dataset_results[approach]['correct']}/{dataset_results[approach]['total']})\n\n")
                f.write("### Confusion Matrix (Data_Check)\n")
                f.write(f"![Confusion Matrix {approach.upper()}](confusion_matrix_{approach}.png)\n\n")
                f.write("### Match Counts Plot (Data_Check)\n")
                f.write(f"![Match Counts {approach.upper()}](match_counts_plot_{approach}.png)\n\n")
                f.write("### Per-Pair Results (Data_Check)\n")
                f.write("| Folder | Expected | Predicted | Good Matches | Visualization |\n")
                f.write("|--------|----------|-----------|--------------|---------------|\n")
                for folder, count in sorted(dataset_results[approach]['match_counts'].items()):
                    expected = "Match" if "same" in folder.lower() else "No Match"
                    predicted = "Match" if count > threshold else "No Match"
                    img_file = f"{folder}_{approach}_{'match' if predicted == 'Match' else 'no_match'}.png"
                    f.write(f"| {folder} | {expected} | {predicted} | {count} | ![Match]({img_file}) |\n")
                f.write("\n")

            if uia_results and approach in uia_results:
                f.write("### UiA Images Results\n")
                res = uia_results[approach]
                f.write(f"- **Predicted**: {res['result']}\n")
                f.write(f"- **Good Matches**: {res['count']}\n")
                img_file = f"uia_{approach}_{'match' if res['result'] == 'Match' else 'no_match'}.png"
                f.write(f"![UiA Match]({img_file})\n\n")

    print(f"Generated Markdown report at: {report_path}")


if __name__ == "__main__":
    results_folder = './results/'
    dataset_results = process_dataset(display_inline=False, results_folder=results_folder) # True for pop ups
    uia_results = process_UIA(display_inline=True, results_folder=results_folder)
    generate_markdown_report(results_folder, dataset_results, uia_results) # Write once in code, templatable