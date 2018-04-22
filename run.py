import numpy as np

from CNN_Classifier import Cnn_Classifier
from data_utils import Data_utils

utils = Data_utils()

training_input,training_output,validation_input,validation_output,test_input,test_output = utils.load_cifar100_data(source = 'keras')
#training_input_tf,training_output_tf,validation_input_tf,validation_output_tf,test_input_tf,test_output_tf = utils.load_mnist_data(source = 'tensorflow')


cnn = Cnn_Classifier(classifier_name = "Cifar100 Classifier",input_shape = training_input.shape,output_num = 100)


cnn.train_classifier(input_images = training_input,output_classes = training_output,iterations_num = 200)

cnn.test_classifier(input_images = test_input_tf,output_classes = test_output_tf)


