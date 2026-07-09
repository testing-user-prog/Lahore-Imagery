import cv2
import numpy as np  
def estimateimageblur(img_path):
    img=cv2.imread(img_path,0)
    return cv2.Laplacian(img,cv2.CV_64F).var()
def getabsvarbyvalues(var1,var2):
    #will return true if var1<=var2
    #will return false if var1>var2
    if var1>=var2:
        return True,var1-var2
    return False,var2-var1

