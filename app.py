from src.dataloader.sensor_config import saveimage
import json
import os
from datetime import date
from dateutil.relativedelta import relativedelta
import shutil
from src.authenticator.authenticate import authenticateuser
from src.changedetector.feature_matcher import matchfeatures
from src.imageprocessor.qualityprocessor import estimateimageblurfrompicture
from src.imageprocessor.qualityprocessor import getabsvarbyvalues,getscale_factor
from src.imageeditor.qualityeditor import GaussianImageBlur, decreaseimagequality
import cv2
# these variables would be later input from the frontend
start_year = 1984
end_year = 2015
geometry_points = [74.4189, 31.4697]

# Load configurations
with open('config.json', 'r') as f:
    config = json.load(f)

authenticateuser(config['project_name'])

year_startdate = date(start_year, 1, 1)
curr_year = start_year

# if os.path.exists(config['save_path']):
#     shutil.rmtree(config['save_path'])

# while curr_year <= end_year:
#     next_yearfirst = year_startdate + relativedelta(years=1)
#     year_enddate = next_yearfirst - relativedelta(days=1)

#     saveimage(
#         start_time=str(year_startdate),
#         end_time=str(year_enddate),
#         geometry_points=geometry_points,
#         image_savepath=config['save_path'],
#         imagename=str(curr_year) + '.png',
#         config=config
#     )

#     curr_year += 1
#     year_startdate = next_yearfirst

# we will ask the user for the image input like the years whose image he want to compare now hardcoded for 2014 and the 2015
# grey1 and grey2 is the grey scaled version of image of interest
# 1. Load images directly into grayscale
grey1 = cv2.imread('src/dataset/2014.png', cv2.IMREAD_GRAYSCALE)  # Blurry baseline
grey2 = cv2.imread('src/dataset/2015.png', cv2.IMREAD_GRAYSCALE)  # Sharp target to degrade

# 2. Get initial variance scores
var1 = estimateimageblurfrompicture(grey1)
var2 = estimateimageblurfrompicture(grey2)
print('The gray score values are: ')
print(var1,var2)
returnstatus, abs_var = getabsvarbyvalues(var1, var2)
print(returnstatus, abs_var)

threshold = 0.125
scale_factor = getscale_factor(var2,var1)

# Create a copy of grey2 to safely mutate inside the loop
grey2_processed = grey2.copy()

# 3. Corrected Loop Condition: Check the score of the *processed* image against the baseline (var1)
while scale_factor > threshold and estimateimageblurfrompicture(grey2_processed) > var1:
    
    # Degrade the sharp 2015 image (grey2) and store it back into grey2_processed
    grey2_processed = decreaseimagequality(grey2, estimateimageblurfrompicture(grey2_processed), scale_factor)
    
    # Drop the scale factor down by 5% for the next iteration
    scale_factor *= 0.95

# 4. Corrected Saving Logic: Save the processed 2015 image to its correct name
cv2.imwrite("src/dataset/res2015.png", grey2_processed) # This is the newly matched 2015 image
cv2.imwrite("src/dataset/res2014.png", grey1)           # This is your pristine, untouched 2014 baseline