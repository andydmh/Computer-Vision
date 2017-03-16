from numpy import *
from cv2 import *
from sklearn import *
import copy
from math import *
from os import *
from glob import *
import matplotlib.pyplot as plt
from matplotlib import colors
import pylab as plt
import random
from sklearn.neural_network import MLPClassifier

#Mapping category numbers to category names 
(diving_number, diving_str) = (1, 'diving')#
(golf_swing_number, golf_swing_str) = (2, 'golf swing')#
(kicking_number, kicking_str) = (3, 'kicking')#
(lifting_number, lifting_str) = (4, 'lifting')#
(riding_horse_number, riding_horse_str) = (5, 'riding horse')#
(running_number, running_str) = (6, 'running')#
(skateboarding_number, skateboarding_str) = (7, 'skateboarding')#
(swing_bench_number, swing_bench_str) = (8, 'swing bench')#
(swing_side_number, swing_side_str) = (9, 'swing side')#
(walking_number, walking_str) = (10, 'walking')#



#---------------------------Get all values-------------------------------------------------------

#returns all the possible names for categories
def get_all_categories():
	return [diving_str, golf_swing_str, kicking_str, lifting_str, riding_horse_str, running_str, \
			skateboarding_str, swing_bench_str, swing_side_str, walking_str]

#returns all possible category numbers 
def get_all_category_numbers():
	return [1,2,3,4,5,6,7,8,9,10]

#TODO: Refactor directory categories to variables
#returns all directory categories which are more than actual categories since some sports/actions have different modalities
def get_all_directory_categories():
	return ['Diving-Side', 'Golf-Swing-Back', 'Golf-Swing-Front', 'Golf-Swing-Side', \
			'Kicking-Front','Kicking-Side', 'Lifting','Riding-Horse','Run-Side', \
			'SkateBoarding-Front', 'Swing-Bench', 'Swing-SideAngle', 'Walk-Front']




#--------------------------------------------Mapping-----------------------------------------
#given a category number it returns the category name
def map_categories_to_colors(category_number):
	return {diving_number:'c', golf_swing_number:'gold', kicking_number:'black', \
			lifting_number:'green',riding_horse_number:'b',running_number:'plum', \
			skateboarding_number:'fuchsia',swing_bench_number:'salmon',swing_side_number:'orange', \
			walking_number:'navy'}[category_number]

#given a category directory it returns the category number
def map_directory_categories_to_numbers(category):
	return {'Diving-Side':diving_number, 'Golf-Swing-Back':golf_swing_number, 'Golf-Swing-Front':golf_swing_number, \
			'Golf-Swing-Side':golf_swing_number,'Kicking-Front':kicking_number,'Kicking-Side':kicking_number, \
			'Lifting':lifting_number,'Riding-Horse':riding_horse_number,'Run-Side':running_number, \
			'SkateBoarding-Front':skateboarding_number, 'Swing-Bench':swing_bench_number, \
			'Swing-SideAngle':swing_side_number, 'Walk-Front':walking_number}[category]

#given a number it returns the category name
def map_numbers_to_categories(number):
	return {diving_number:diving_str, golf_swing_number:golf_swing_str, kicking_number:kicking_str, \
			lifting_number:lifting_str, riding_horse_number:riding_horse_str, running_number: running_str, \
			skateboarding_number: skateboarding_str, swing_bench_number: swing_bench_str, \
			swing_side_number: swing_side_str, walking_number:walking_str}[number]	



#-----------------------------------------Helper Math Functions-------------------------------------
#returns the pythagorean distance
def magnitude(p1, p2):
	return sqrt(1.0*((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2))

#TODO: remember to plot with and without negative angles
#returns the direction of the vector formed by joining 2 poits 
def angle(p1, p2):
	angle = atan2(p2[1]-p1[1], p2[0]-p1[0])
	#To make sure that the average of all of the angles is not 0
	if angle < 0:
		angle = angle + 2*pi
	
	return angle



#----------------------------------------------Feature Extractors------------------------------------	
# min number of features in one of the videos = 16
#This function finds all the flows and directions of interest points in the video
def get_flow_directions_and_magnitudes_helper(video_path, numb_features):

	feature_parameters = dict(maxCorners = numb_features, qualityLevel = 0.3, minDistance = 7, blockSize = 7)
	lucas_kanade_parameters = dict(winSize = (15,15), maxLevel = 2, criteria = (TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, 10, 0.03))
	
	
	cap = VideoCapture(video_path)
	
	ret, im = cap.read()
	initial_image = copy.deepcopy(im)
	
	gray_im = cvtColor(im, COLOR_BGR2GRAY)
	
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
		
		
		gray_im = new_gray_im.copy()
		corner_cordinates = new_coordinates.reshape(-1,1,2)
	
	magnitudes = []
	angles = []
	for k in range(len(initial_coordinates)):
		magnitudes.append(magnitude(initial_coordinates[k][0], new_coordinates[k][0]))
		angles.append(angle(initial_coordinates[k][0], new_coordinates[k][0]))

	return (magnitudes, angles)

flow_feature_extractor_description = 'It uses all the flow direction and magnitude of interest points as features. It will return a total of 200 features, 100 magnitudes and 100 directions. One for every key point. If there are not 100 key points 0s will be added as features.'
#returns all the magnitudes and directions as a feature vector [M1, M2, ..., Mn, O1, O2, ...., On] 
def flow_feature_extractor(video_path, numb_features):
	(magnitudes, angles) = get_flow_directions_and_magnitudes_helper(video_path, numb_features)
	return magnitudes + angles

#This function is used to fix the issue in which different feature vectors have different sizes due to the number of interest points in their respective videos
def fix_different_numbers_of_features(feature_vectors):
	max = -1
	new_feature_vectors = []
	
	for feature_vector in feature_vectors:
		if len(feature_vector) > max:
			max = len(feature_vector)
	
	for feature_vector in feature_vectors:	
		magnitudes = feature_vector[0:len(feature_vector)/2]
		angles = feature_vector[len(feature_vector)/2:len(feature_vector)]
		
		zeros_to_add = (max - len(feature_vector))/2
		zeros_list = [0]*zeros_to_add
		
		
		magnitudes = magnitudes + zeros_list
		angles = angles + zeros_list
		
		new_feature_vectors.append(magnitudes + angles)
	return new_feature_vectors 


flow_feature_extractor_avg_description = 'It finds the average of all the flow magnitudes and angles:' + \
										 ' [ag_magnitude, avg_angle]'
#It returns a feature vector that consists on [avg_magnitude, avg_angle]		
def flow_feature_extractor_avg(video_path, numb_features):
	(magnitudes, angles) = get_flow_directions_and_magnitudes_helper(video_path, numb_features)
	return [sum(magnitudes)/(len(magnitudes)*1.0),  sum(angles)/(len(angles)*1.0)]

flow_feature_extractor_avg_max_min_description = 'From the optical flow of the video it returns [avg_magnitude, max_magnitude, min_magnitude, avg_angle, max_angle, min_angle]'
#It returns a feture vector that consists on [avg_magnitude, max_magnitude, min_magnitude, avg_angle, min_angle, max_angle]
def flow_feature_extractor_avg_max_min(video_path, numb_features):
	(magnitudes, angles) = get_flow_directions_and_magnitudes_helper(video_path, numb_features)
	
	avg_magnitude = sum(magnitudes)/(len(magnitudes)*1.0)
	max_magnitude = amax(magnitudes) 
	min_magnitude = amin(magnitudes)
	
	avg_angle = sum(angles)/(len(angles)*1.0)
	max_angle = amax(angles)
	min_angle = amin(angles)
	
	return [avg_magnitude, max_magnitude, min_magnitude,  avg_angle, max_angle, min_angle]
	
	


flow_feature_extractor_max_description = 'From the optical flow of the video it returns [max_magnitude, max_angle]'
#It returns a feature vector that consits on [max_magnitude, max_angle]
def flow_feature_extractor_max(video_path, numb_features):
	(magnitudes, angles) = get_flow_directions_and_magnitudes_helper(video_path, numb_features)
	
	#avg_magnitude = sum(magnitudes)/(len(magnitudes)*1.0)
	max_magnitude = amax(magnitudes) 
	#min_magnitude = amin(magnitudes)
	
	#avg_angle = sum(angles)/(len(angles)*1.0)
	max_angle = amax(angles)
	#min_angle = amin(angles)
	
	return [max_magnitude, max_angle]
#-------------------------------------File Structure Helper Functions------------------------------------------------

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

#----------------------------------------Display Data Helper Functions---------------------------------------
#TODO: Add lables
#This function creates a scatter graph of points and lable them according to their category
def plot_points(points, categories):
	dx = {}
	dy = {}
	
	for i in range(len(points)):
		x = points[i][0]
		y = points[i][1]
		
		color = map_categories_to_colors(categories[i])
		lab = map_numbers_to_categories(categories[i]) 
		
		if (lab, color) in dx:
			dx[(lab, color)].append(x)
			dy[(lab, color)].append(y)
			
		else:
			dx[(lab, color)] = [x]
			dy[(lab, color)] = [y]
	
	for tup in dx:
		plt.scatter(dx[tup],dy[tup], c=tup[1], label=tup[0])
	
	plt.legend()
	plt.grid(True)
	plt.show()

#This function creates a histogram out of the data provided in dictionary
def create_histogram(dictionary):
	counts = range(1, len(dictionary)+1)
	
	labels = []
	
	values = []
	
	for label in dictionary:
		labels.append(label)
		values.append(dictionary[label])
	
	plt.bar(counts, values, align='center')
	plt.xticks(values, labels)
	plt.show()
	

#---------------------------------------Learning Helpers-------------------------------------------
#This is an abstract class to be used by all the classifiers. The names of all functions inside of it are self descriptive 
class Classifier(object):
	
	def __init__(self, features, categories):
		self.categories = categories
		self.features = features
		self.n = 0
		self.results = []
		self.build_confusion_matrix()
	
	def build_confusion_matrix(self):
		self.confusion_matrix = {}
		
		for trueCategry in get_all_category_numbers():
			for predictedCategory in get_all_category_numbers():
				self.confusion_matrix[(trueCategry, predictedCategory)] = 0
	#It returns a tuple (training, testing, flag), where flag becomes false when n = len(data) - 1 
	def next_training_testing_sets(self):
		testing = ([self.features[self.n]], [self.categories[self.n]])
		training = (self.features[:self.n] + self.features[self.n:], self.categories[:self.n] + self.categories[self.n:])
		flag = True
		if self.n == len(self.features) -1:
			self.n = 0
			flag = False
		else:
			self.n = self.n + 1 

		return (training, testing, flag)
		
	def restart_learning(self):
		self.n = 0

	def get_accuracy(self):
		return (sum(self.results)/(1.0*len(self.results)))

	def get_true_possitive(self, category_number):
		tp = 1.0*self.confusion_matrix[(category_number, category_number)]
		if tp == 0:
			tp = 0.000000000001
		return tp
	
	def get_false_possitive(self, predicted_category_number):
		category_numbers = get_all_category_numbers()
		category_numbers.remove(predicted_category_number)
		
		false_positive = 0.0
		
		for true_category_number in category_numbers:
				false_positive += self.confusion_matrix[(true_category_number, predicted_category_number)]
		
		if false_positive == 0:
			false_positive = 0.000000000001
		
		return false_positive
	
	def get_true_negative(self, category_number):
		category_numbers = get_all_category_numbers()
		category_numbers.remove(category_number)
		
		true_negative = 0.0
		
		for predicted_category in category_numbers:
			for true_category in category_numbers:
				true_negative += self.confusion_matrix[(true_category,predicted_category)]
		
		if true_negative == 0:
			true_negative = 0.000000000001
		
		return true_negative
	
	def get_false_negative(self, category_number):
		category_numbers = get_all_category_numbers()
		category_numbers.remove(category_number)
		
		false_negative = 0.0
		
		for predicted_category in category_numbers: 
			false_negative += self.confusion_matrix[(category_number, predicted_category)]
		
		if false_negative == 0:
			false_negative = 0.000000000001
		
		return false_negative
		
	def get_sensitivities(self):
		sensitivities = {}
		
		category_numbers = get_all_category_numbers()
		
		for category_number in category_numbers:
			category_name = map_numbers_to_categories(category_number)
			
			tp = self.get_true_possitive(category_number)
			fn = self.get_false_negative(category_number)
			
			sensitivities[category_name] = tp/(tp+fn)  
		
		return sensitivities
		
	def get_spesificities(self):
		spesificities = {}
		
		category_numbers = get_all_category_numbers()
		
		for category_number in category_numbers:
			category_name = map_numbers_to_categories(category_number)
			
			tn = self.get_true_negative(category_number)
			fp = self.get_false_possitive(category_number)
			
			spesificities[category_name] = tn/(tn+fp)
		
		return spesificities
	
	def get_presicions(self):
		presicions = {}
		
		category_numbers = get_all_category_numbers()
		
		for category_number in category_numbers:
			category_name = map_numbers_to_categories(category_number)
			
			tp = self.get_true_possitive(category_number)
			fp = self.get_false_possitive(category_number)
			
			presicions[category_name] = tp/(tp+fp)
		
		return presicions
	
	def get_confussion_matrix(self):
		return self.confusion_matrix
	
	def get_results(self):
		return self.results
	#this shpuld return the statistics
	#TODO: implement this
	def execute(self):
		next = self.next_training_testing_sets()

		flag = next[2]
		
		while flag:
			training_features = next[0][0]
			training_categories = next[0][1]
			testing_features = next[1][0]
			testing_categories = next[1][1]
	
			flag = next[2]
			self.train(training_features, training_categories)
	
			#Now try to classify the testig case and keep track
			prediction = self.predict(testing_features)
			
			self.confusion_matrix[(testing_categories[0], prediction)] += 1 
			
			if prediction == testing_categories[0]:
				self.results.append(1)
			else:
				self.results.append(0)
			
			
			#Get next group
			next = self.next_training_testing_sets()
	
	
	def train(self, training_features, training_categories): raise NotImplementedError('Override me')
	
	def predict(self, feature_vector): raise NotImplementedError('Override me')

	def get_parameters(self): raise NotImplementedError('Override me')

#This class uses sklearn Linear SVM CLassifier. It uses Classifier as Superclass
class Linear_SVM_Classifier(Classifier):
	def __init__(self, features, categories):
		super(Linear_SVM_Classifier, self).__init__(features, categories)
	
	#TODO: Keep implementing this
	def train(self, training_features, training_categories):
		self.classifier = svm.LinearSVC()
		self.classifier.fit(training_features, training_categories)

	def predict(self, feature_vector):
		return self.classifier.predict(feature_vector)[0]
	
	def get_parameters(self):
		return self.classifier.get_params()

#Uses the SVC class of sklearn to implement an SVM classifier
class SVM_Classifier(Classifier):
	def __init__(self, features, categories):
		super(SVM_Classifier, self).__init__(features, categories)
	
	#TODO: Keep implementing this
	def train(self, training_features, training_categories):
		self.classifier = svm.SVC(kernel='linear')
		self.classifier.fit(training_features, training_categories)

	def predict(self, feature_vector):
		return self.classifier.predict(feature_vector)[0]
	
	def get_parameters(self):
		return self.classifier.get_params()

#Uses sklearn Neural Network Classifier
class Neural_Nets_Classifier(Classifier):
	def __init__(self, features, categories):
		super(Neural_Nets_Classifier, self).__init__(features, categories)
	
	#TODO: Keep implementing this
	def train(self, training_features, training_categories):
		self.classifier = MLPClassifier(solver='lbgfs', alpha=1e-5, hidden_layer_sizes=(130, ), random_state=1)
		self.classifier.fit(training_features, training_categories)

	def predict(self, feature_vector):
		return self.classifier.predict(feature_vector)[0]
	
	def get_parameters(self):
		return self.classifier.get_params()
