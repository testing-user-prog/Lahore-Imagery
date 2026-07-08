import cv2
import numpy as np  
def estimateimageblur(img_path):
    img=cv2.imread(img_path,0)
    return cv2.Laplacian(img,cv2.CV_64F).var()
    