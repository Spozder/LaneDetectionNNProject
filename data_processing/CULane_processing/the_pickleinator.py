"""Pull in road images - currently uses 1 in 10 images
in order to account for time series data (i.e. very similar
images over a short period of time due to 30 fps video.
Also saves a pickle file for later use.
"""

import os
import glob
import cv2
import pickle
import re
import numpy as np

top = 150
bottom = 150

# Load road image locations in order
def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def pull_images(images):
	fNames = glob.glob('driver_23_30frame/05171117_0771.MP4/*.jpg')
	fNames.sort()
	for fName in fNames:
		img = cv2.imread(fName)
		img = img[top:img.shape[0]-bottom+1,:,:]
		images.append(img)


# List for images
road_images = []

# Pull in the desired images       
pull_images(road_images)

# Save the images to a pickle file
pickle.dump(road_images,open('bridge_original_images_cropped.p', "wb" ))