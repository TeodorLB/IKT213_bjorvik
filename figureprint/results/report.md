# Image Matching Report (ORB_BF and SIFT_FLANN)

**Threshold Used**: 20

## ORB_BF Approach

- **Data_Check Accuracy**: 80.00% (16/20)

### Confusion Matrix (Data_Check)
![Confusion Matrix ORB_BF](confusion_matrix_orb_bf.png)

### Match Counts Plot (Data_Check)
![Match Counts ORB_BF](match_counts_plot_orb_bf.png)

### Per-Pair Results (Data_Check)
| Folder | Expected | Predicted | Good Matches | Visualization |
|--------|----------|-----------|--------------|---------------|
| different_1 | No Match | No Match | 6 | ![Match](different_1_orb_bf_no_match.png) |
| different_10 | No Match | No Match | 19 | ![Match](different_10_orb_bf_no_match.png) |
| different_2 | No Match | No Match | 1 | ![Match](different_2_orb_bf_no_match.png) |
| different_3 | No Match | No Match | 3 | ![Match](different_3_orb_bf_no_match.png) |
| different_4 | No Match | No Match | 7 | ![Match](different_4_orb_bf_no_match.png) |
| different_5 | No Match | No Match | 1 | ![Match](different_5_orb_bf_no_match.png) |
| different_6 | No Match | No Match | 2 | ![Match](different_6_orb_bf_no_match.png) |
| different_7 | No Match | No Match | 1 | ![Match](different_7_orb_bf_no_match.png) |
| different_8 | No Match | No Match | 14 | ![Match](different_8_orb_bf_no_match.png) |
| different_9 | No Match | No Match | 10 | ![Match](different_9_orb_bf_no_match.png) |
| same_1 | Match | Match | 43 | ![Match](same_1_orb_bf_match.png) |
| same_10 | Match | Match | 34 | ![Match](same_10_orb_bf_match.png) |
| same_2 | Match | No Match | 18 | ![Match](same_2_orb_bf_no_match.png) |
| same_3 | Match | No Match | 5 | ![Match](same_3_orb_bf_no_match.png) |
| same_4 | Match | No Match | 9 | ![Match](same_4_orb_bf_no_match.png) |
| same_5 | Match | Match | 47 | ![Match](same_5_orb_bf_match.png) |
| same_6 | Match | Match | 23 | ![Match](same_6_orb_bf_match.png) |
| same_7 | Match | Match | 29 | ![Match](same_7_orb_bf_match.png) |
| same_8 | Match | Match | 53 | ![Match](same_8_orb_bf_match.png) |
| same_9 | Match | No Match | 16 | ![Match](same_9_orb_bf_no_match.png) |

## SIFT_FLANN Approach

- **Data_Check Accuracy**: 85.00% (17/20)

### Confusion Matrix (Data_Check)
![Confusion Matrix SIFT_FLANN](confusion_matrix_sift_flann.png)

### Match Counts Plot (Data_Check)
![Match Counts SIFT_FLANN](match_counts_plot_sift_flann.png)

### Per-Pair Results (Data_Check)
| Folder | Expected | Predicted | Good Matches | Visualization |
|--------|----------|-----------|--------------|---------------|
| different_1 | No Match | No Match | 9 | ![Match](different_1_sift_flann_no_match.png) |
| different_10 | No Match | No Match | 6 | ![Match](different_10_sift_flann_no_match.png) |
| different_2 | No Match | No Match | 5 | ![Match](different_2_sift_flann_no_match.png) |
| different_3 | No Match | No Match | 7 | ![Match](different_3_sift_flann_no_match.png) |
| different_4 | No Match | No Match | 12 | ![Match](different_4_sift_flann_no_match.png) |
| different_5 | No Match | No Match | 5 | ![Match](different_5_sift_flann_no_match.png) |
| different_6 | No Match | No Match | 11 | ![Match](different_6_sift_flann_no_match.png) |
| different_7 | No Match | No Match | 11 | ![Match](different_7_sift_flann_no_match.png) |
| different_8 | No Match | No Match | 6 | ![Match](different_8_sift_flann_no_match.png) |
| different_9 | No Match | No Match | 18 | ![Match](different_9_sift_flann_no_match.png) |
| same_1 | Match | Match | 57 | ![Match](same_1_sift_flann_match.png) |
| same_10 | Match | Match | 50 | ![Match](same_10_sift_flann_match.png) |
| same_2 | Match | Match | 41 | ![Match](same_2_sift_flann_match.png) |
| same_3 | Match | No Match | 12 | ![Match](same_3_sift_flann_no_match.png) |
| same_4 | Match | Match | 21 | ![Match](same_4_sift_flann_match.png) |
| same_5 | Match | Match | 36 | ![Match](same_5_sift_flann_match.png) |
| same_6 | Match | No Match | 8 | ![Match](same_6_sift_flann_no_match.png) |
| same_7 | Match | Match | 30 | ![Match](same_7_sift_flann_match.png) |
| same_8 | Match | Match | 82 | ![Match](same_8_sift_flann_match.png) |
| same_9 | Match | No Match | 19 | ![Match](same_9_sift_flann_no_match.png) |

