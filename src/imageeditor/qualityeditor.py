import cv2
import numpy as np
def GaussianImageBlur(img_path,var_diff,path_to_save):
    img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    sigma=np.sqrt(var_diff)
    kernel_size=int(2*np.ceil(2*sigma)+1)
    if kernel_size%2==0:
        kernel_size+=1
    kernal_size=max(3,kernel_size)
    blurred_img=cv2.GaussianBlur(img,(kernel_size,kernel_size),sigma)
    cv2.imwrite(path_to_save,blurred_img)
    return blurred_img

#Pixelation
def decreaseimagequality(img1,img2,score1,score2,scale_factor): #from main score1>score2
    #for this function the score of img1 should be greater than the score of the image 2
    try:
        if score1<score2:
            raise Exception("Parameters passed in the incorrect order!")
    except Exception as error:
        print(f'An error caught: {error}')
    h,w=img1.shape

    low_res_w=int(w*scale_factor)
    low_res_h=int(h*scale_factor)
    img_small=cv2.resize(img1,(low_res_w,low_res_h),interpolation=cv2.INTER_AREA)
    return cv2.resize(img_small,(w,h),interpolation=cv2.INTER_NEAREST)




