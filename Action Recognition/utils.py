from numpy import *
from cv2 import *
from sklearn import *
import copy
from math import *
from os import *
from glob import *

def get_categories():
	return {0:'Diving-Side', 1:'Golf-Swing-Back', 2:'Golf-Swing-Front',3:'Golf-Swing-Side',4:'Kicking-Front',5:'Kicking-Side',6:'Lifting',7:'Riding-Horse',8:'Run-Side',9:'SkateBoarding-Front', 10:'Swing-Bench', 11:'Swing-SideAngle', 12:'Walk-Front'}
	

def magnitude(p1, p2):
	return sqrt(1.0*((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2))

def angle(p1, p2):
	return atan2(p2[1]-p1[1], p2[0]-p1[0])
	
# numb_features = 16
def flow_feature_extractor(video_path, numb_features):

	feature_parameters = dict(maxCorners = numb_features, qualityLevel = 0.3, minDistance = 7, blockSize = 7)
	lucas_kanade_parameters = dict(winSize = (15,15), maxLevel = 2, criteria = (TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, 10, 0.03))
	
	
	cap = VideoCapture(video_path)
	
	ret, im = cap.read()
	initial_image = copy.deepcopy(im)
	
	gray_im = cvtColor(im, COLOR_BGR2GRAY)
	
	#[[[x1,y2]], [[x2,y2]]]
	corner_cordinates = goodFeaturesToTrack(gray_im, mask=None, **feature_parameters)
	initial_coordinates = copy.deepcopy(corner_cordinates)
	new_coordinates = None
	
	while True:
		
		ret, new_im = cap.read()
		
		if not ret:
			break
		
		new_gray_im = cvtColor(new_im, COLOR_BGR2GRAY)
		
		#logic
		new_coordinates, st, err = calcOpticalFlowPyrLK(gray_im, new_gray_im, corner_cordinates, None, **lucas_kanade_parameters)
		
		#good_news = new_coordinates[st==1]
		
		
		gray_im = new_gray_im.copy()
		corner_cordinates = new_coordinates.reshape(-1,1,2)
	
	
	'''print 'Initial Coordinates:'	
	print initial_coordinates
	
	print 'New Coordinates:'
	print new_coordinates
	line(initial_image, tuple(initial_coordinates[15][0]), tuple(new_coordinates[15][0]), (0,0,255), 10)
	imwrite('hello.jpg', initial_image)'''
	
	magnitudes = []
	angles = []
	for k in range(len(initial_coordinates)):
		magnitudes.append(magnitude(initial_coordinates[k][0], new_coordinates[k][0]))
		angles.append(angle(initial_coordinates[k][0], new_coordinates[k][0]))

	return magnitudes + angles


def flow_feature_extractor_avg(video_path, numb_features):

	feature_parameters = dict(maxCorners = numb_features, qualityLevel = 0.3, minDistance = 7, blockSize = 7)
	lucas_kanade_parameters = dict(winSize = (15,15), maxLevel = 2, criteria = (TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, 10, 0.03))
	
	
	cap = VideoCapture(video_path)
	
	ret, im = cap.read()
	initial_image = copy.deepcopy(im)
	
	gray_im = cvtColor(im, COLOR_BGR2GRAY)
	
	#[[[x1,y2]], [[x2,y2]]]
	corner_cordinates = goodFeaturesToTrack(gray_im, mask=None, **feature_parameters)
	initial_coordinates = copy.deepcopy(corner_cordinates)
	new_coordinates = None
	
	while True:
		
		ret, new_im = cap.read()
		
		if not ret:
			break
		
		new_gray_im = cvtColor(new_im, COLOR_BGR2GRAY)
		
		#logic
		new_coordinates, st, err = calcOpticalFlowPyrLK(gray_im, new_gray_im, corner_cordinates, None, **lucas_kanade_parameters)
		
		#good_news = new_coordinates[st==1]
		
		
		gray_im = new_gray_im.copy()
		corner_cordinates = new_coordinates.reshape(-1,1,2)
	
	
	'''print 'Initial Coordinates:'	
	print initial_coordinates
	
	print 'New Coordinates:'
	print new_coordinates
	line(initial_image, tuple(initial_coordinates[15][0]), tuple(new_coordinates[15][0]), (0,0,255), 10)
	imwrite('hello.jpg', initial_image)'''
	
	magnitudes = []
	angles = []
	for k in range(len(initial_coordinates)):
		magnitudes.append(magnitude(initial_coordinates[k][0], new_coordinates[k][0]))
		angles.append(angle(initial_coordinates[k][0], new_coordinates[k][0]))

	return [sum(magnitudes)/(len(magnitudes)*1.0),  sum(angles)/(len(angles)*1.0)]


#main_path is the path to the directory that contains all the category directories like Lifting, Run-Side ... that is the ucf_action directory
#This function returns a dictionary that maps the name of the category to a list of video paths. Example {'Diving-Side':[video1_path, video2_path ...], ...}
def get_all_video_paths(ucf_action_dir_path):
	videos = {}

	category_dirs = listdir(ucf_action_dir_path)
	
	for category in category_dirs:
		videos[category] = []
		directories = listdir(ucf_action_dir_path +'/'+category)
		
		for directory in directories:
			videos[category] = videos[category] + glob(ucf_action_dir_path +'/'+category+'/'+directory+'/*.avi')
	
	return videos
	
#print flow_feature_extractor('ucf_action/Golf-Swing-Back/001/3283-8_700741.avi',16)
#print flow_feature_extractor_avg('ucf_action/Golf-Swing-Back/001/3283-8_700741.avi',100)
#print get_all_video_paths('/home/andy/Documents/ucf_action')

