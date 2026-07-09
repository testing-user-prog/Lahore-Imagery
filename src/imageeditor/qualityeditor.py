import cv2
import numpy as np
def imageblurbyvariancediff(img_path,var_diff,path_to_save):
    img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    sigma=np.sqrt(var_diff)
    kernel_size=int(2*np.ceil(2*sigma)+1)
    if kernel_size%2==0:
        kernel_size+=1
    kernal_size=max(3,kernel_size)
    blurred_img=cv2.GaussianBlur(img,(kernel_size,kernel_size),sigma)
    cv2.imwrite(path_to_save,blurred_img)
    return blurred_img

