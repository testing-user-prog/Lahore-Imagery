import cv2

def markregions(image_path, coordinates, save_path, radius=20, color=(0, 0, 255), thickness=2):
    """
    Loads the original color image at image_path, draws a circle at each
    (x, y) coordinate in the coordinates list, and saves the annotated
    image to save_path.

    Args:
        image_path  (str):              Path to the source image file.
        coordinates (list[tuple]):      List of (x, y) pixel coordinates to mark.
        save_path   (str):              Path where the annotated image will be saved.
        radius      (int):              Radius of each marker circle (default 20).
        color       (tuple):            BGR color of the marker circle (default red).
        thickness   (int):              Circle border thickness (default 2).
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image at: {image_path}")

    for (x, y) in coordinates:
        cv2.circle(image, (x, y), radius, color, thickness)

    cv2.imwrite(save_path, image)
    print(f"Marked image saved to: {save_path}")
