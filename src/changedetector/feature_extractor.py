import cv2
import numpy as np


def _is_plausible_transform(transform, image_shape):
    """
    Validates whether an affine transformation matrix falls within realistic geometric bounds.
    Checks scaling factor (0.8x to 1.2x), rotation angle (<= 10 deg), and total translation shift.
    """
    # Calculate uniform scaling factor using Euclidean norm of the first column
    scale = np.hypot(transform[0, 0], transform[1, 0])
    
    # Calculate rotation angle in degrees from the transformation matrix
    rotation = abs(np.degrees(np.arctan2(transform[1, 0], transform[0, 0])))
    
    # Extract dimensions to bound maximum allowed spatial translation shift
    height, width = image_shape
    max_translation = max(10, 0.1 * max(width, height))
    translation = np.hypot(transform[0, 2], transform[1, 2])
    
    # Return True only if scale, rotation, and translation fall within plausible limits
    return (
        0.8 <= scale <= 1.2
        and rotation <= 10
        and translation <= max_translation
    )


def _select_change_locations(
    keypoints1, indices1, keypoints2, indices2, transform, limit, min_distance=12
):
    """
    Selects top unmatched keypoint locations based on feature strength (response),
    applies Non-Maximum Suppression (spatial filtering), and maps points back to both image domains.
    """
    selected = []
    
    # Extract response (strength score) and (x, y) coordinates for unmatched indices in Image 1
    candidates = [(keypoints1[index].response, keypoints1[index].pt) for index in indices1]
    
    # Append response and coordinates for unmatched indices in Image 2
    candidates.extend(
        (keypoints2[index].response, keypoints2[index].pt) for index in indices2
    )
    
    # Sort candidate keypoints by strength in descending order (strongest corners first)
    for _, (x, y) in sorted(candidates, key=lambda candidate: candidate[0], reverse=True):
        # Enforce spatial separation (min_distance) to prevent duplicate/overlapping detections
        if all((x - sx) ** 2 + (y - sy) ** 2 >= min_distance**2 for sx, sy in selected):
            selected.append((x, y))
            # Stop once the requested feature limit is reached
            if len(selected) == limit:
                break

    # Convert Image 1 spatial selection into integer pixel coordinates
    coordinates1 = [(int(x), int(y)) for x, y in selected]
    
    # Apply affine transform to project Image 1 coordinates accurately onto Image 2 domain
    coordinates2 = [
        tuple(
            np.rint(transform[:, :2] @ np.array([x, y]) + transform[:, 2])
            .astype(int)
            .tolist()
        )
        for x, y in selected
    ]
    return coordinates1, coordinates2


def matchfeatures(image_name1, image_name2, feature_count=10):
    """
    Return paired unmatched keypoint coordinates for each image.

    The images are registered before comparing feature locations, so points
    from unrelated parts of the scene cannot be treated as correspondences.
    Each returned coordinate pair identifies the same scene location in both
    source images.
    """
    # Input validation: Ensure a positive target count for change keypoints
    if feature_count <= 0:
        raise ValueError("feature_count must be a positive integer")

    # Load both target images as single-channel grayscale matrices
    img1 = cv2.imread(image_name1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_name2, cv2.IMREAD_GRAYSCALE)
    
    # Check if files exist and were successfully loaded
    if img1 is None:
        raise FileNotFoundError(f"Could not load image at: {image_name1}")
    if img2 is None:
        raise FileNotFoundError(f"Could not load image at: {image_name2}")
        
    # Enforce identical image spatial dimensions for registration pipeline
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions to compare feature locations")

    # Initialize ORB detector dynamically scaled to sample enough candidate keypoints
    orb = cv2.ORB_create(nfeatures=max(500, min(feature_count * 50, 2000)))
    keypoints1 = orb.detect(img1, None)
    keypoints2 = orb.detect(img2, None)
    
    # Early exit if either image fails to yield detectable keypoints
    if not keypoints1 or not keypoints2:
        return [], []

    try:
        # Perform sub-pixel image registration using Enhanced Correlation Coefficient (ECC) algorithm
        _, transform = cv2.findTransformECC(
            img1.astype(np.float32) / 255,
            img2.astype(np.float32) / 255,
            np.eye(2, 3, dtype=np.float32),
            cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6),
            None,
            5,
        )
    except cv2.error as error:
        raise RuntimeError(
            f"Could not align images '{image_name1}' and '{image_name2}'"
        ) from error

    # Verify that computed affine matrix represents a realistic camera/satellite alignment
    if not _is_plausible_transform(transform, img1.shape):
        raise RuntimeError(
            f"Image alignment between '{image_name1}' and '{image_name2}' "
            "is outside the expected range"
        )

    # Warp Image 2 using calculated inverse affine transform matrix to align perfectly with Image 1
    # the transform parameter gives us the transformation matrix computed from above
    # WARP inverse allows backward mapping from so missing values can be placed from on the destination image from the source directly
    aligned_img2 = cv2.warpAffine(
        img2,
        transform,
        (img1.shape[1], img1.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
    )
    
    # Detect ORB keypoints on the aligned Image 2 surface
    aligned_keypoints2 = orb.detect(aligned_img2, None)
    
    # Extract (x, y) float coordinate arrays for pair-wise spatial matrix operations
    points1 = np.float32([keypoint.pt for keypoint in keypoints1])
    points2 = np.float32([keypoint.pt for keypoint in aligned_keypoints2])

    # Compute full Pairwise Euclidean Distance Matrix between all keypoints across both images
    distances = np.linalg.norm(points1[:, None, :] - points2[None, :, :], axis=2)
    
    # Identify indices of keypoints that fall within an 8-pixel radius (matched scene features)
    matched_indices_img1 = set(np.flatnonzero(distances.min(axis=1) <= 8))
    matched_indices_img2 = set(np.flatnonzero(distances.min(axis=0) <= 8))

    # Filter out matched points to isolate unmatched change candidates and return transformed coordinate pairs
    return _select_change_locations(
        keypoints1,
        set(range(len(keypoints1))) - matched_indices_img1,
        aligned_keypoints2,
        set(range(len(aligned_keypoints2))) - matched_indices_img2,
        transform,
        feature_count,
    )