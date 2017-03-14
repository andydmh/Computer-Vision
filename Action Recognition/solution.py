'''
Name: Andy D. Martinez
Programing Assigment 3
'''

from utils import *
from sys import *

#Names used to identify classifiers
nets = 'Neural Nets'
svm = 'SVM'
svm2 = 'SVC' 

#Path to UCF action database
ucf_action_dir_path = None

#-----------------------------------------------Experiments----------------------------------------------------

#This experiment provides data in which even clever classifiers will perform 10% accuracyin the best case scenario
def experiment_problematic_data(which_classifier, plot_data, f_extractor_description=None):
	
	classifier_str = ''
	categories = [1,2,3,4,5, 6,7,8,9,10]
	features = [[1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,1],[1,1],[1,1],[1,1]]
	
	if plot_data:
		plot_points(features, categories)
	
	classifier = get_classifier(which_classifier, features, categories)
	classifier.execute()
	
	results = {}
	results['Accuracy'] = str(classifier.get_accuracy())
	results['Specifisities'] = str(classifier.get_spesificities())
	results['Sensitivities'] = str(classifier.get_sensitivities())
	results['Presicions'] = str(classifier.get_presicions())
	parameters = str(classifier.get_parameters())
	objective = 'To test problematic data that guarantees a 10% accuracy'
	f_extractor_description='This is a toy example. Features were manually created'
	
	describe_experiment(which_classifier, f_extractor_description, parameters, results, objective)



#This is just an experiment in which the provided data has no structure. Although it is always possible for classifiers to find some hidden structure in randomness, it is spected that the performance in this experiment will not be great 
def experiment_random(which_classifier, plot_data, f_extractor_description=None):
	
	classifier_str = ''
	categories = [random.randint(1,10) for x in range(100)]
	features = []
	for i in range(100):
		features.append(random.sample(range(20), 2))
	
	if plot_data:
		#plot_points(features, categories)
		print 'plot'
	
	classifier = get_classifier(which_classifier, features, categories)
	classifier.execute()
	
	results = {}
	results['Accuracy'] = str(classifier.get_accuracy())
	results['Specifisities'] = str(classifier.get_spesificities())
	results['Sensitivities'] = str(classifier.get_sensitivities())
	results['Presicions'] = str(classifier.get_presicions())
	parameters = str(classifier.get_parameters())
	objective = 'To test random data'
	f_extractor_description='This is a toy example. Features were manually created'
	
	describe_experiment(which_classifier, f_extractor_description, parameters, results, objective)




#This is a toy example to test that my set up for the sklearn classifiers is fine
#which_classifier = 'nets' for neural nets and 'svm' for svm
#In this case data is extremely well structure and classifiers should perform great, as well as 100% accuracy
def experiment_to_test_classifier(which_classifier, plot_data, f_extractor_description=None):
	
	classifier_str = ''
	categories = [1,1,2,2,2, 3,3]
	features = [[1,1], [1,2], [5,5], [4,5], [4,10], [15,15], [15,17]]
	
	if plot_data:
		plot_points(features, categories)
	
	classifier = get_classifier(which_classifier, features, categories)
	classifier.execute()
	
	results = {}
	results['Accuracy'] = str(classifier.get_accuracy())
	results['Specifisities'] = str(classifier.get_spesificities())
	results['Sensitivities'] = str(classifier.get_sensitivities())
	results['Presicions'] = str(classifier.get_presicions())
	parameters = str(classifier.get_parameters())
	objective = 'To make sure that my setup for the learning algorithm is working fine'
	f_extractor_description='This is a toy example. Features were manually created'
	
	describe_experiment(which_classifier, f_extractor_description, parameters, results, objective)
	
#This is a generic experiment that can be used to test multiple scenarios
def generic_experiment(video_paths, which_classifier, feature_extractor, f_extractor_description, fix_feature_vector_length = False, plot_data = False):
	categories = []
	features = []
	
	for category in video_paths:
		category_videos = video_paths[category]
	
		for video in category_videos:
			categories.append(map_directory_categories_to_numbers(category))
			features.append(feature_extractor(video, 100))
	
	if fix_feature_vector_length:
		features = fix_different_numbers_of_features(features)
	
	if plot_data:
		plot_points(features, categories)
		
	classifier = get_classifier(which_classifier, features, categories)
	classifier.execute()
	
	results = {}
	results['Accuracy'] = str(classifier.get_accuracy())
	results['Specifisities'] = str(classifier.get_spesificities())
	#create_histogram(classifier.get_spesificities())
	results['Sensitivities'] = str(classifier.get_sensitivities())
	results['Presicions'] = str(classifier.get_presicions())
	#results['Confussion Matrix'] = str(classifier.get_confussion_matrix())
	
	parameters = classifier.get_parameters()
	
	describe_experiment(which_classifier, f_extractor_description, parameters, results)
	
#-----------------------------------------Helper Functions----------------------------------------------

#A helper function to print the results of all the experiments
def describe_experiment(which_classifier, f_extractor_description, parameters, results, objective = None):
	print '-----------------------------------------------------'
	
	if objective == None:
		print 'Experiment Objective: To test the feature extractor ' + \
		  'in combination with the learning algorithm ' + which_classifier
	else:
		print 'Experiment Objective: ' + objective
	
	print '\nClassifier: ' + which_classifier	  
	print '\nFeature Extractor: ' + f_extractor_description
	
	for key in results:
		print '\n'+key +': ' + str(results[key])
		
	print '\nParameters:' + str(parameters)
	print '------------------------------------------------------'

def get_classifier(which_classifier, features, categories):
	classifier = None	
	
	if which_classifier == svm:
		classifier = Linear_SVM_Classifier(features, categories)
	elif which_classifier == svm2:
		classifier = SVM_Classifier(features, categories)
	else:
		classifier = Neural_Nets_Classifier(features, categories)
	
	return classifier
	
	
#This function will deal with theuser input parameters
def handle_user_input():
	#pass as an argument the path to the ucf action database directory
	if len(argv) != 2:
		return None
		
	ucf_action_dir_path = argv[1]
	video_paths = get_all_video_paths(ucf_action_dir_path)
	
	return video_paths

#This function is the starting point for all the experiments 
def run_experiments(video_paths):
	print 'Experiment 1:'
	experiment_problematic_data(nets, False)
	print '\n\n\nExperiment 2:'
	experiment_random(nets, False)
	print '\n\n\nExperiment 3:'
	experiment_to_test_classifier(svm, False)
	print '\n\n\n Experiment 4:'
	generic_experiment(video_paths, svm, flow_feature_extractor_avg, flow_feature_extractor_avg_description, False, False)
	print '\n\n\n Experiment 5:'
	generic_experiment(video_paths, svm, flow_feature_extractor_avg_max_min, flow_feature_extractor_avg_max_min_description)
	print '\n\n\n Experiment 6:'
	generic_experiment(video_paths, svm, flow_feature_extractor, flow_feature_extractor_description, True)
	print '\n\n\n Experiment 7:'
	generic_experiment(video_paths, svm, flow_feature_extractor_max, flow_feature_extractor_max_description, False, True)
	print '\n\n\n Experiment 8:'
	generic_experiment(video_paths, svm2, flow_feature_extractor, flow_feature_extractor_description, True)
	print '\n\n\n Experiment 9:'
	generic_experiment(video_paths, svm2, flow_feature_extractor_avg_max_min, flow_feature_extractor_avg_max_min_description)
	print '\n\n\n Experiment 10'
	generic_experiment(video_paths, nets, flow_feature_extractor, flow_feature_extractor_description, True)
	print '\n\n\n Experiment 11:'
	generic_experiment(video_paths, nets, flow_feature_extractor_avg_max_min, flow_feature_extractor_avg_max_min_description)

#----------------------------------------------Main------------------------------------------
video_paths = handle_user_input()
if video_paths == None:
	print "Provide as an argument the path to ucf action database. python solution.py path"
else:
	run_experiments(video_paths)
