import numpy as np
import keras.datasets.mnist as mst

import keras.datasets.cifar100 as cf100
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data 
class Data_utils:

	def load_mnist_data(self,source,path = ""):

		if source == "keras":
			"""
			load data from keras
			"""

			training,testing = mst.load_data()
			training_input,training_output = training
			testing_input,testing_output = testing
			training_input = training_input.reshape(-1,28,28,1)
			training_output = training_output.reshape(-1,1)
			testing_input = testing_input.reshape(-1,28,28,1)
			testing_output = testing_output.reshape(-1,1)


			validation_input = testing_input[5000:]
			testing_input = testing_input[0:5000]
			validation_output = testing_output[5000:]
			testing_output = testing_output[0:5000]


		elif source =="tensorflow":
			"""
			load data from tensorflow
			"""
			mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
			training,validation,testing = mnist 
			training_input = training.images
			training_output = training.labels
			testing_input = testing.images
			testing_output = testing.labels
			validation_input = validation.images
			validation_output = validation.labels

		elif source == "input file":
			"""
			used to talk the data from input file
			"""

		else:
			print("This data source is not supported")
			return 

		return training_input,training_output,validation_input,validation_output,testing_input,testing_output

	def load_cifar100_data(self,source,path = ""):


		if source == "keras":
			"""
			load data from keras
			"""

			training,testing = cf100.load_data()
			training_input,training_output = training
			testing_input,testing_output = testing
			training_input = training_input.reshape(-1,32,32,3)
			training_output = training_output.reshape(-1,1)
			testing_input = testing_input.reshape(-1,32,32,3)
			testing_output = testing_output.reshape(-1,1)

			validation_input = testing_input[5000:]
			testing_input = testing_input[0:5000]
			
			validation_output = testing_output[5000:]
			testing_output = testing_output[0:5000]
			
		elif source == "input file":
			"""
			used to talk the data from input file
			"""

		else:
			print("This data source is not supported")
			return 


		return training_input,training_output,validation_input,validation_output,testing_input,testing_output

		




		


	







