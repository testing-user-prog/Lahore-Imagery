import cv2
import numpy as np

def matchfeatures(image_name1, image_name2,feature_count=10):
    """
    Detects ORB features in both images and matches them.
    Returns two lists of (x, y) coordinates for the UNMATCHED keypoints
    in image1 and image2 respectively — these are the points of interest
    (potential change regions) that did not find a match in the other image.
    """
    img1 = cv2.imread(f'{image_name1}', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(f'{image_name2}', cv2.IMREAD_GRAYSCALE)

    # Initialize ORB detector limited to the top 10 strongest features
    orb = cv2.ORB_create(nfeatures=feature_count)

    # Detect keypoints (locations) and compute binary descriptors (fingerprints) for both images
    # Possible Problem 1: Prominent features of images are identified independently
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Match binary descriptors using Hamming distance with mutually consistent matching (cross-check)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #Problem 2: queryIdx and TrainIDx images issue
    matches = bf.match(des1, des2)
    # Sort matches in ascending order of distance (best/strongest matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Collect indices of keypoints that successfully found a match in each image
    #Problem 3: Different parts of the image can be matched who have entirely different coordinates
    matched_indices_img1 = {m.queryIdx for m in matches}
    matched_indices_img2 = {m.trainIdx for m in matches}
    #Problem 4: Problem 3 can lead to problem 4 i.e the distinct points at entirely differnet position
    # Extract (x, y) integer pixel coordinates for unmatched keypoints in image 1
    unmatched_coords_img1 = [
        (int(kp1[idx].pt[0]), int(kp1[idx].pt[1]))
        for idx in range(len(kp1))
        if idx not in matched_indices_img1
    ]

    # Extract (x, y) integer pixel coordinates for unmatched keypoints in image 2
    unmatched_coords_img2 = [
        (int(kp2[idx].pt[0]), int(kp2[idx].pt[1]))
        for idx in range(len(kp2))
        if idx not in matched_indices_img2
    ]

    # Return lists of unmatched coordinate pairs for downstream region marking
    return unmatched_coords_img1, unmatched_coords_img2