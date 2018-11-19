import numpy as np
import sys
from CNN_Classifier import Cnn_Classifier
from data_utils import Data_utils
from random import randint

action_type = sys.argv[1]


utils = Data_utils()

training_input,training_output,validation_input,validation_output,testing_input,testing_output = utils.load_cifar100_data(source = 'keras')
training_input,training_output = utils.apply_preprocessing(training_input,training_output,operations = ['flip_horizontal'])

if action_type == 'train':
	cnn = Cnn_Classifier(classifier_name = "Cifar100_Classifier",input_shape = training_input.shape,output_num = 100)
	output = cnn.build_cnn_model(input_images = training_input,output_classes = training_output)
	cnn.set_batch_size(1000)
	cnn.train_classifier(input_images = training_input,output_encoded= output,testing_input = testing_input,testing_output = testing_output ,iterations_num=1000)
	cnn.test_classifier(input_images = testing_input,output_classes = testing_output)
	cnn.sess.close()

if action_type == 'retrain':
	new_cnn = Cnn_Classifier(classifier_name = "Cifar100_Classifier",input_shape = training_input.shape,output_num = 100,define_weights = False)
	new_cnn.set_batch_size(1000)
	new_cnn.retrain_classifier(input_images = training_input,output_classes = training_output,testing_input = testing_input,testing_output = testing_output,iterations_num = 500,checkpoint='70')
	new_cnn.test_classifier(input_images = testing_input,output_classes = testing_output)
	new_cnn.sess.close()

if action_type == 'predict':
	new_cnn = Cnn_Classifier(classifier_name = "Cifar100_Classifier",input_shape = training_input.shape,output_num = 100,define_weights = False)
	rand_numbers = [randint(0, 5000) for p in range(0, 10)]
	images=[]
	for num in rand_numbers:
		curr_image = testing_input[num]
		curr_image = curr_image.reshape(32,32,3)
		true_label_index = testing_output[num]
		true_label = new_cnn.cifar100_labels[true_label_index]
		new_cnn.predict_labels(curr_image,i)
		print("true label of image "+str(i)+"is  "+true_label)
		









