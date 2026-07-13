import cv2
import numpy as np  
def estimateimageblurfrompicture(img):
    return cv2.Laplacian(img,cv2.CV_64F).var()

def estimateimageblurbypath(img_path):
    img=cv2.imread(img_path,0)
    return estimateimageblurfrompicture(img)
    
def getabsvarbyvalues(var1,var2):
    #will return true if var1<=var2
    #will return false if var1>var2
    if var1>=var2:
        return True,var1-var2
    return False,var2-var1


def getscale_factor(var1, var2):
    if var1 <= var2:
        return 1.0
    predicted_scale = np.sqrt(var2 / var1)
    return max(0.05, min(0.99, predicted_scale))
        

