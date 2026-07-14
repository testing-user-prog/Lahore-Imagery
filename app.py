from src.dataloader.sensor_config import saveimage
import json
import os
from datetime import date
from dateutil.relativedelta import relativedelta
import shutil
from src.authenticator.authenticate import authenticateuser
from src.changedetector.feature_extractor import matchfeatures
from src.imageprocessor.qualityprocessor import estimateimageblurfrompicture
from src.imageprocessor.qualityprocessor import getabsvarbyvalues,getscale_factor
from src.imageeditor.qualitychanger import GaussianImageBlur, decreaseimagequality
from src.imageeditor.markups import markregions
import cv2
# these variables would be later input from the frontend
start_year = 1984
end_year = 2015
c1 = "2014"
c2 = "2015"
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

# we will ask the user for the image input like the years whose image he want to compare
# grey1 and grey2 is the grey scaled version of image of interest
# 1. Load images directly into grayscale using c1 and c2
grey1 = cv2.imread(f'src/dataset/{c1}.png', cv2.IMREAD_GRAYSCALE)  # Blurry baseline
grey2 = cv2.imread(f'src/dataset/{c2}.png', cv2.IMREAD_GRAYSCALE)  # Sharp target to degrade
if grey1 is None:
    # Fallback to absolute/relative path check
    grey1 = cv2.imread(f'src/dataset/{c1}.png', cv2.IMREAD_GRAYSCALE)
if grey2 is None:
    grey2 = cv2.imread(f'src/dataset/{c2}.png', cv2.IMREAD_GRAYSCALE)

# 2. Get initial variance scores
var1 = estimateimageblurfrompicture(grey1)
var2 = estimateimageblurfrompicture(grey2)
print('The gray score values are: ')
print(var1, var2)
returnstatus, abs_var = getabsvarbyvalues(var1, var2)
print(returnstatus, abs_var)

# Initial estimate of scale factor
scale_factor = getscale_factor(var2, var1)

# Create a copy of grey2 to safely mutate inside the loop
grey2_processed = grey2.copy()

# We set a minimum threshold to prevent downscaling to less than 5% of the original size
min_scale_threshold = 0.05
prev_scale_factor = 1.0

# 3. Dynamic Feedback Loop: Adjust scale factor based on the variance ratio
# This formula automatically adapts the step size for different years/sensors
while scale_factor > min_scale_threshold and estimateimageblurfrompicture(grey2_processed) > var1:
    current_var = estimateimageblurfrompicture(grey2_processed)
    
    # If the scale factor is no longer changing significantly, stop to prevent infinite looping
    if abs(prev_scale_factor - scale_factor) < 1e-4:
        break
        
    prev_scale_factor = scale_factor
    
    # Degrade the sharp image (grey2) and store it back into grey2_processed
    grey2_processed = decreaseimagequality(grey2, current_var, scale_factor)
    
    # Dynamic formula: update the scale factor based on the ratio of target variance to current variance.
    # We use a dampening exponent (e.g., 0.4) to smoothly converge without overshooting.
    scale_factor *= (var1 / current_var) ** 0.4

# 4. Corrected Saving Logic: Save the processed image using c1 and c2 variables
cv2.imwrite(f"src/dataset/res{c2}.png", grey2_processed)  # This is the newly matched image
cv2.imwrite(f"src/dataset/res{c1}.png", grey1)            # This is your pristine, untouched baseline

# 5. Feature Matching: detect unmatched keypoints (regions of potential change)
coords_img1, coords_img2 = matchfeatures(
    f'src/dataset/res{c1}.png',
    f'src/dataset/res{c2}.png'
)
print(f'Unmatched coords in {c1}: {coords_img1}')
print(f'Unmatched coords in {c2}: {coords_img2}')

# 6. Mark the detected change regions on the original (color) images
markregions(
    image_path=f'src/dataset/{c1}.png',
    coordinates=coords_img1,
    save_path=f'src/dataset/marked_{c1}.png'
)
markregions(
    image_path=f'src/dataset/{c2}.png',
    coordinates=coords_img2,
    save_path=f'src/dataset/marked_{c2}.png'
)