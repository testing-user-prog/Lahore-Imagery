import cv2
import numpy as np

def matchfeatures(image_name1, image_name2):
    """
    Detects ORB features in both images and matches them.
    Returns two lists of (x, y) coordinates for the UNMATCHED keypoints
    in image1 and image2 respectively — these are the points of interest
    (potential change regions) that did not find a match in the other image.
    """
    img1 = cv2.imread(f'{image_name1}', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(f'{image_name2}', cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create(nfeatures=10)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    matched_indices_img1 = {m.queryIdx for m in matches}
    matched_indices_img2 = {m.trainIdx for m in matches}

    unmatched_coords_img1 = [
        (int(kp1[idx].pt[0]), int(kp1[idx].pt[1]))
        for idx in range(len(kp1))
        if idx not in matched_indices_img1
    ]

    unmatched_coords_img2 = [
        (int(kp2[idx].pt[0]), int(kp2[idx].pt[1]))
        for idx in range(len(kp2))
        if idx not in matched_indices_img2
    ]

    return unmatched_coords_img1, unmatched_coords_img2