from moviepy.editor import ImageSequenceClip
import os
import numpy as np
import cv2
import pickle
from scipy.misc import imresize
from scipy import stats

# from keras.models import load_model

# Load Keras model
# model = load_model('full_CNN_model.h5')

def combine_original_with_predictions(image, cnn_label, rnn_label):
	""" Takes in a road image, re-sizes for the model,
	predicts the lane to be drawn from the model in G color,
	recreates an RGB image of a lane and merges with the
	original road image.
	"""

	# CNN
	# Generate fake R & B color dimensions, stack with G
	blanks = np.zeros_like(cnn_label).astype(np.uint8)
	cnn_lane_drawn = np.dstack((blanks, cnn_label, blanks)).astype(np.uint8)

	# RNN
	if rnn_label is not None:
		blanks = np.zeros_like(rnn_label).astype(np.uint8)
		rnn_lane_drawn = np.dstack((blanks, rnn_label, blanks))
		rnn_lane_drawn = imresize(rnn_lane_drawn, (80, 160, 3))

		print(rnn_lane_drawn.dtype)
		print(cnn_lane_drawn.dtype)

		cnn_lane_drawn = cv2.addWeighted(cnn_lane_drawn, 1, rnn_lane_drawn, 1, 0)
		cv2.imshow('merge',cnn_lane_drawn)
		cv2.waitKey()

	# Re-size to match the original image
	cnn_lane_drawn = imresize(cnn_lane_drawn, (291, 1640, 3))

	# Merge the lane drawing(s) onto the original image
	result = cv2.addWeighted(image, 1, cnn_lane_drawn, 1, 0)
	print(result.shape)
	print(type(result))
	cv2.imshow('result',result)
	cv2.waitKey()
	return result

def makeConfusionMatrix(original_label, network_label):
	threshold = 0.5
	#print(np.max(network_label))
	original_label = original_label.squeeze()
	network_label = network_label.squeeze()
	if (original_label.shape != network_label.shape):
		network_label = imresize(network_label, original_label.shape)
	rounded_original_label = original_label > threshold
	rounded_network_label = network_label > threshold
	#print("1s in rounded_original_label: {}".format(np.sum(rounded_original_label == 1)))
	#print("1s in rounded_network_labelel: {}".format(np.sum(rounded_network_label == 1)))
	true_positives = np.sum(np.logical_and(rounded_network_label == 1, rounded_original_label == 1))
	false_positives = np.sum(np.logical_and(rounded_network_label == 1, rounded_original_label == 0))
	false_negatives = np.sum(np.logical_and(rounded_network_label == 0, rounded_original_label == 1))
	true_negatives = np.sum(np.logical_and(rounded_network_label == 0, rounded_original_label == 0))
	#print("Number of true positives in frame: {}".format(true_positives))
	#print("Number of false positives in frame: {}".format(false_positives))
	#print("Number of false negatives in frame: {}".format(false_negatives))
	#print("Number of true negatives in frame: {}".format(true_negatives))
	return np.array([[true_positives, false_positives], [false_negatives, true_negatives]])


# Where to save the output video
# vid_output = 'proj_reg_vid.mp4'

# Location of the input videoon
# clip1 = VideoFileClip("project_video.mp4")

# vid_clip = clip1.fl_image(road_lines)
# vid_clip.write_videofile(vid_output, audio=False)

if __name__ == '__main__':
	np.set_printoptions(suppress=True)
	num_frames = 52851
	confusionMatricesRNN = []
	confusionMatricesCNN = []
	original_labels = np.array(pickle.load(open('road_labels.p', 'rb'))[0:num_frames]) / 200
	cnn_labels = np.array(pickle.load(open('CNN_road_labels.p', 'rb'))[0:num_frames])
	rnn_labels = np.array(pickle.load(open('RNN_road_labels.p', 'rb'))[0:num_frames]).reshape(-1, 40, 80, 1)
	for i in range(num_frames):
		confusionMatricesRNN.append(makeConfusionMatrix(original_labels[i], rnn_labels[i]))
	for i in range(num_frames):
		confusionMatricesCNN.append(makeConfusionMatrix(original_labels[i], cnn_labels[i]))

	print("Average Confusion Matrix for RNN:")
	print(np.mean(np.array(confusionMatricesRNN), axis=0))

	print("Average Confusion Matrix for CNN:")
	print(np.mean(np.array(confusionMatricesCNN), axis=0))

	# clip = ImageSequenceClip(rnn_cnn_original, fps=5)
	# clip.write_videofile('bridgecombined.avi', audio=False, codec='rawvideo')