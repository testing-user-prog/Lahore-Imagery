from src.dataloader.sensor_config import saveimage
import json
import os
from datetime import date
from dateutil.relativedelta import relativedelta
import shutil
from src.authenticator.authenticate import authenticateuser
from src.changedetector.feature_matcher import matchfeatures
from src.imageprocessor.qualityprocessor import estimateimageblur
from src.imageprocessor.qualityprocessor import getabsvarbyvalues
from src.imageeditor.qualityeditor import imageblurbyvariancediff
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

# we will ask the user for the image input like the years whose image he want to compare

var1=estimateimageblur('src/dataset/2014.png')
var2=estimateimageblur('src/dataset/2015.png')
returnstatus,abs_var=getabsvarbyvalues(var1,var2)
print(returnstatus,abs_var)
imageblurbyvariancediff('src/dataset/2015.png',abs_var,'src/dataset/res.png')
# matchfeatures('src/dataset/2014.png', 'src/dataset/2015.png',f'{config['save_path']}/1.png')
# print('Task Completed')

