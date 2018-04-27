from moviepy.editor import ImageSequenceClip
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pickle
from skimage import transform
import cv2

# Create video from director of jpg files

def videoOfFramesInDir(dir, fps=30):
	frameFNamesList = glob.glob(os.path.join(dir, '*.jpg'))
	frameFNamesList.sort()
	frameFNamesList = frameFNamesList[0:77]
	return ImageSequenceClip(frameFNamesList, fps=fps)

def makePickleFileOfFramesInDir(dir, n, pName):
	frameFNamesList = glob.glob(os.path.join(dir, '*.jpg'))
	frameFNamesList.sort()
	frameFNamesList = frameFNamesList[0:n]
	images = np.array([mpimg.imread(fName) for fName in frameFNamesList])
	pickle.dump(images,open(pName, "wb" ))

# Combine label data with video
def detectLanesInFramesInDir(dir, fps=30):
	frameFNamesList = glob.glob(os.path.join(dir, '*.jpg'))
	frameFNamesList.sort()
	images = np.array([mpimg.imread(fName) for fName in frameFNamesList])
	predictedLanes = predictLanesFromImages(images)



#clip = videoOfFramesInDir('/media/spencer/76A4-7403/dataset/driver_23_30frame/05171117_0771.MP4', 10)
#clip.write_videofile('bridge.mp4', audio=False)

frames = np.array(pickle.load(open('bridge_RNN_output.p', 'rb')))
for f in frames:
	f_i = f < 0.05
	f[f_i] = 0
frames = [np.array(cv2.resize(f, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)) for f in frames]

print(len(frames))
print(frames[0].shape)

plt.imshow(frames[0])
plt.show()

input()

clip = ImageSequenceClip(frames, fps=2)
#clip.write_videofile('bridgernn.avi', audio=False, codec='png')

#makePickleFileOfFramesInDir('/media/spencer/76A4-7403/dataset/driver_23_30frame_resized_nolabel/05171117_0771.MP4', 77, 'bridge.p')