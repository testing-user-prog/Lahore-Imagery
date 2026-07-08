import cv2
import numpy as np

def matchfeatures(image_name1, image_name2,pathtosave):
    img1_color = cv2.imread(f'{image_name1}')
    img2_color = cv2.imread(f'{image_name2}')
    
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=10)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    matched_indices_img2 = {m.trainIdx for m in matches}
    output_image = img2_color.copy()
    
    for idx, kp in enumerate(kp2):
        if idx not in matched_indices_img2:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            cv2.circle(output_image, (x, y), 20, (0, 0, 255), 2)

    cv2.imwrite(pathtosave, output_image)