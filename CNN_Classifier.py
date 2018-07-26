import numpy as np
import tensorflow as tf
import os
import random


class Cnn_Classifier:

	def __init__(self,classifier_name,input_shape = [None,28,28,1],conv_layers = [(32,5),(64,5),(128,3),(256,2),(256,1)],pooling_layers = [1,1,1,0,0],dropout_layers=[1,1,1,1,1], batchnorm_layers = [1,1,1,1,1],activation_type = "relu",pool_type='avg',fc_size = 2048, learning_rate = 0.0003, output_num = 10,dropout=.75,define_weights = True):
		self.weights_list = []
		self.biases_list = []
		self.layers_num = len(conv_layers)

		current_channels = input_shape[3]

		self.pool_type = pool_type
		self.output_num = output_num
		self.classifier_name = classifier_name
		self.input_shape = input_shape
		self.batch_size=2000
		self.sess = tf.Session()
		self.dropout = dropout
		self.learning_rate = learning_rate
		if define_weights:
			w_name = 'kernel_weight'
			i=0
			for layer in conv_layers:
				current_weights = self.weights_variable(shape = [layer[1],layer[1],current_channels,layer[0]],name = w_name+str(i))
				current_bias = self.bias_variable([1,layer[0]])
				current_channels = layer[0]
				self.weights_list.append(current_weights)
				self.biases_list.append(current_bias)
				i+=1
			self.pooling_layers = pooling_layers
			self.batchnorm_layers = batchnorm_layers
			self.dropout_layers = dropout_layers
			self.activation_type = activation_type
			self.fc_size = fc_size
			

			num =1
			for i in pooling_layers:
				if i ==1:
					num*=2

			w_name = "fully_connected"
			self.fc_wights = self.weights_variable(shape = [int((input_shape[1]/num)*(input_shape[2]/num)*conv_layers[-1][0]),output_num],name = w_name)

			self.fc_bias = self.bias_variable([1,output_num])
			
			_,image_hight,image_width,channel_num = self.input_shape
			self.Input = tf.placeholder(tf.float32,[None,image_hight,image_width,channel_num],name = "Input")
			self.Output = tf.placeholder(tf.float32,[None,self.output_num],name = "Output")
			self.keep_prob = tf.placeholder(tf.float32,[],name = "keep_prob")
			tf.add_to_collection('Input', self.Input)
			tf.add_to_collection('Output',self.Output)
			tf.add_to_collection('keep_prob',self.keep_prob)



		self.output_num = output_num
		self.classifier_name = classifier_name
		self.input_shape = input_shape
		self.batch_size=500
		self.sess = tf.Session()

		self.cifar100_labels = [ 'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle','bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
		'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
		'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
		'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
		'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
		'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
		'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
		'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
		'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
		'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
		'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
		'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
		'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman','worm']

		if not os.path.exists(self.classifier_name):
			os.makedirs(self.classifier_name)


	def get_classifer_name(self):
		return self.classifier_name

	def set_batch_size(self,new_batch_size):
		self.batch_size = new_batch_size

	def weights_variable(self,shape,name):
		return tf.get_variable(name = name,shape = shape,initializer=  tf.contrib.layers.xavier_initializer()) 


	def bias_variable(self,shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)


	def encode_labels(self,labels,classes_num):
		"""
		generate one_hot encoding matrix for the labels with shape (len(labels),classes_num) 
		"""
		labels_encoded = np.zeros([len(labels),classes_num])


		for i in range(labels_encoded.shape[0]):
			labels_encoded[i][self.labels_hashed[labels[i]]]=1

		return labels_encoded



	def one_hot(self,output_list):

		output_encoded = tf.one_hot(output_list,self.output_num)
		s = tf.Session()

		results = s.run(output_encoded)
		s.close()
		return results.reshape(-1,self.output_num)


	def restore_model(self,checkpoint='75'):

		# saver = tf.train.Saver()
		saver = tf.train.import_meta_graph(self.classifier_name+"/best_model/model_"+str(checkpoint)+'.meta')

		saver.restore(self.sess, self.classifier_name+"/best_model/model_"+str(checkpoint))

		self.optimizer = tf.get_collection('optimizer')[0]
		self.accuracy = tf.get_collection('accuracy')[0]
		self.cross_entropy = tf.get_collection('cross_entropy')[0]
		self.Input = tf.get_collection('Input')[0]
		self.Output = tf.get_collection('Output')[0]
		self.keep_prob = tf.get_collection('keep_prob')[0]
		self.correct_labels = tf.get_collection('correct_labels')[0]


	def shuffle_data(self,features,output):

		size = features.shape[0]

		ind_list = [i for i in range(size)]
		random.shuffle(ind_list)
		shuffled_features  = features[ind_list, :,:,:]
		shuffled_output = output[ind_list,:]

		return shuffled_features,shuffled_output



	

	def build_cnn_model(self,input_images,output_classes,encode = True):
		#import ipdb;ipdb.set_trace()

		input_layer = self.Input
		curr_pool_layer = None
		curr_batchnorm_layer = None

		print('')
		print('')
		print('layers stacked dimensions is as following:')
		print('')
		for i in range(self.layers_num):

			curr_conv_layer = tf.nn.conv2d(input = input_layer,filter = self.weights_list[i], strides = [1,1,1,1],padding = 'SAME',name = 'conv_'+str(i)) + self.biases_list[i] 
			tf.add_to_collection('conv_'+str(i),curr_conv_layer)
			if self.activation_type == "relu":
				curr_activation_layer = tf.nn.relu(curr_conv_layer)
			elif self.activation_type == "sigmoid":
				curr_activation_layer = tf.nn.sigmoid(curr_conv_layer)
			elif self.activation_type == "tanh":
				curr_activation_layer = tf.nn.tanh(curr_conv_layer)
			else:
				curr_activation_layer = curr_conv_layer


			print(curr_conv_layer)


			if self.batchnorm_layers[i] ==1 :
				curr_batchnorm_layer = tf.contrib.layers.batch_norm(inputs = curr_activation_layer)
			else:
				curr_batchnorm_layer = curr_activation_layer
			tf.add_to_collection('batch_'+str(i),curr_batchnorm_layer)

			if self.pooling_layers[i] == 1:
				if self.pool_type == 'max':
					curr_pool_layer = tf.nn.max_pool(value = curr_batchnorm_layer, ksize = [1,2,2,1],strides = [1,2,2,1], padding = 'SAME', name = 'pool_'+str(i))
				elif self.pool_type == 'avg':
					curr_pool_layer = tf.nn.avg_pool(value = curr_batchnorm_layer, ksize = [1,2,2,1],strides = [1,2,2,1], padding = 'SAME', name = 'pool_'+str(i))


			else:
				curr_pool_layer = curr_batchnorm_layer
			tf.add_to_collection('pool_'+str(i),curr_pool_layer)

			if self.dropout_layers[i] ==1:
				curr_dropout_layer = tf.nn.dropout(curr_pool_layer,self.keep_prob)
			else:
				curr_dropout_layer = curr_pool_layer


			tf.add_to_collection('dropout_'+str(i),curr_dropout_layer)
			print(curr_dropout_layer)

			input_layer = curr_dropout_layer


		flatten_layer = tf.contrib.layers.flatten(curr_dropout_layer)

		self.Ylogits = tf.nn.softmax(tf.matmul(flatten_layer,self.fc_wights)+self.fc_bias)
		self.cross_entropy=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.Ylogits,labels=self.Output))
		self.optimizer= tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cross_entropy)


		tf.add_to_collection('cross_entropy', self.cross_entropy)
		tf.add_to_collection('optimizer',self.optimizer)

		if encode == True:
			output_encoded = self.one_hot(output_classes)
		else:
			output_encoded = output_classes

		self.correct_labels = tf.argmax(self.Ylogits,1)
		correct_prediction = tf.equal(tf.argmax(self.Ylogits,1), tf.argmax(self.Output,1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		tf.add_to_collection('accuracy',self.accuracy)
		tf.add_to_collection('correct_labels', self.correct_labels)

		self.sess.run(tf.global_variables_initializer())

		return output_encoded


	def train_classifier(self,input_images,output_encoded,testing_input,testing_output,iterations_num):


		images_num,image_width,image_hight,channel_num = input_images.shape
		acc_60 = False
		acc_70= False
		acc_80 = False
		acc_90 = False
		acc_95 = False
		acc_99 = False
		acc_996 = False
		acc_998 = False
		acc_999 = False
		acc_9992 = False
		acc_9994 = False
		acc_9996 = False
		acc_9998 = False
		acc_9999 = False


		testing_output = self.one_hot(testing_output)

		print('Start training '+self.classifier_name+'...')

		number_of_batches = images_num/self.batch_size

		if images_num % self.batch_size != 0:
			number_of_batches +=1



		for it in range(iterations_num):
			#import ipdb;ipdb.set_trace()
			avg_accuracy = 0.0
			avg_loss = 0.0

			for i in range(int(number_of_batches)):
				

				start = i*self.batch_size
				end = (i+1)*self.batch_size 

				if end > input_images.shape[0]:
					end = input_images.shape[0]
				# import ipdb;ipdb.set_trace()
				input_batch = input_images[start:end]
				output_batch  = output_encoded[start:end].reshape(-1,self.output_num)
				results= self.sess.run([self.optimizer,self.accuracy,self.cross_entropy],feed_dict={self.Input:input_batch,self.Output: output_batch,self.keep_prob :self.dropout})

				avg_accuracy += results[1]
				avg_loss += results[2]

				# print('the accuracy of current batch '+str(i)+'/'+str(number_of_batches)+' with accuacy: %'+str(results[1]*100))
			

			avg_accuracy /=number_of_batches
			avg_loss /= number_of_batches

			saver = tf.train.Saver()


			if avg_accuracy > 0.60 and acc_60 == False:
				save_path = saver.save(self.sess, self.classifier_name+"/model_"+str(60))
				self.learning_rate = self.learning_rate /10
				acc_60 = True
				print("Model with accuracy higher than 0.60 are saved in path: %s" % save_path)			
			if avg_accuracy > 0.70 and acc_70 == False:
				save_path = saver.save(self.sess, self.classifier_name+"/model_"+str(70))
				self.learning_rate = self.learning_rate /10
				acc_70 = True
				print("Model with accuracy higher than 0.70 are saved in path: %s" % save_path)

			elif avg_accuracy > 0.80 and acc_80 == False:
				save_path = saver.save(self.sess, self.classifier_name+"/model_"+str(80))
				self.learning_rate = self.learning_rate /10
				acc_80 = True
				print("Model with accuracy higher than 0.80 are saved in path: %s" % save_path)
			elif avg_accuracy > 0.90 and acc_90 == False:
				save_path = saver.save(self.sess, self.classifier_name+"/model_"+str(90))
				self.learning_rate = self.learning_rate /10
				acc_90 = True
				print("Model with accuracy higher than 0.9 are saved in path: %s" % save_path)
			elif avg_accuracy > 0.95 and acc_95 == False:
				save_path = saver.save(self.sess, self.classifier_name+"/model_"+str(95))
				self.learning_rate = self.learning_rate /10
				acc_95 = True
				print("Model with accuracy higher than 0.95 are saved in path: %s" % save_path)
			elif avg_accuracy > 0.99 and acc_99==False:
				save_path = saver.save(self.sess, self.classifier_name+"/model_"+str(99))
				self.learning_rate = self.learning_rate/10
				acc_99 = True
				print("Model with accuracy higher than 0.99 are saved in path: %s" % save_path)
			elif avg_accuracy > 0.996 and acc_996 ==False:
				save_path = saver.save(self.sess, self.classifier_name+"/model_"+str(996))
				self.learning_rate = self.learning_rate/10
				acc_996 = True
				print("Model with accuracy higher than 0.996 are saved in path: %s" % save_path)
			elif avg_accuracy > 0.998 and acc_998 ==False:
				save_path = saver.save(self.sess, self.classifier_name+"/model_"+str(998))
				self.learning_rate = self.learning_rate/10
				acc_998 = True
				print("Model with accuracy higher than 0.998 are saved in path: %s" % save_path)
			elif avg_accuracy > 0.999 and acc_999 == False:
				save_path = saver.save(self.sess, self.classifier_name+"/model_"+str(999))
				self.learning_rate = self.learning_rate/10
				acc_999 = True
				print("Model with accuracy higher than 0.999 are saved in path: %s" % save_path)
			elif avg_accuracy > 0.9992 and acc_9992 == False:
				save_path = saver.save(self.sess, self.classifier_name+"/model_"+str(9992))
				self.learning_rate = self.learning_rate/10
				acc_9992 = True
				print("Model with accuracy higher than 0.9992 are saved in path: %s" % save_path)
			elif avg_accuracy > 0.9994 and acc_9994 == False:
				save_path = saver.save(self.sess, self.classifier_name+"/model_"+str(9994))
				self.learning_rate = self.learning_rate/10
				acc_9994 = True
				print("Model with accuracy higher than 0.9994 are saved in path: %s" % save_path)
			elif avg_accuracy > 0.9996 and acc_9996 == False:
				save_path = saver.save(self.sess, self.classifier_name+"/model_"+str(9996))
				self.learning_rate = self.learning_rate/10
				acc_9996 = True
				print("Model with accuracy higher than 0.9996 are saved in path: %s" % save_path)
			elif avg_accuracy > 0.9998 and acc_9998 == False:
				save_path = saver.save(self.sess, self.classifier_name+"/model_"+str(9998))
				self.learning_rate = self.learning_rate/10
				acc_9998 = True
				print("Model with accuracy higher than 0.9998 are saved in path: %s" % save_path)
			elif avg_accuracy > 0.9999 and acc_9999 == False:
				save_path = saver.save(self.sess, self.classifier_name+"/model_"+str(9999))
				self.learning_rate = self.learning_rate/10
				acc_9999 = True
				print("Model with accuracy higher than 0.9999 are saved in path: %s" % save_path)


			print('the accuracy of the epoch '+str(it+1)+' is : %'+str(avg_accuracy*100.0)+' with loss = '+str(avg_loss))
			self.test_classifier(input_images = testing_input,output_classes = testing_output,encode = False)
			print("------>>>>>"+str(it+1))
			input_images,output_encoded = self.shuffle_data(input_images,output_encoded)

		print('End Training '+self.classifier_name+'.')


	def retrain_classifier(self,input_images,output_classes,testing_input,testing_output,iterations_num,checkpoint='80',encode = True):

		self.restore_model(checkpoint= checkpoint)

		if encode == True:
			output_encoded = self.one_hot(output_classes)
		else:
			output_encoded = output_classes

		self.train_classifier(input_images=input_images,output_encoded=output_encoded,testing_input = testing_input,testing_output = testing_output,iterations_num=iterations_num)



	def predict_labels(self,input_image,index):

		results= self.sess.run([self.correct_labels],feed_dict={self.Input:input_image,self.keep_prob:1})
		label = self.cifar100_labels[results[0][0]]

		cv2.imshow("image "+str(index)+" predict label is "+label,input_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		
	def test_classifier(self,input_images,output_classes,encode = True):


		if self.batch_size > input_images.shape[0]:
			self.batch_size = input_images.shape[0]

		if encode == True:
			output_encoded = self.one_hot(output_classes)
		else:
			output_encoded = output_classes

		images_len = input_images.shape[0]
		number_of_batches = images_len/self.batch_size
		avg_loss = 0.0
		avg_accuracy = 0.0

		if images_len%self.batch_size != 0:
			number_of_batches +=1


		for i in range(int(number_of_batches)):

			start = i*self.batch_size
			end = (i+1)*self.batch_size 

			if (i+1)*self.batch_size > input_images.shape[0]:
				end = input_images.shape[0]

			input_batch =  input_images[start:end].reshape(-1,self.input_shape[1],self.input_shape[2],self.input_shape[3])
			output_batch = output_encoded[start:end].reshape(-1,self.output_num)

			results= self.sess.run([self.accuracy,self.cross_entropy],feed_dict={self.Input:input_batch,self.Output: output_batch,self.keep_prob:1})
			avg_accuracy += results[0]
			avg_loss += results[1]

			print('the accuracy of current batch: %'+str(results[0]*100))

		avg_accuracy /=number_of_batches
		avg_loss /= number_of_batches
		print('the accuracy of the test data is : %'+str(avg_accuracy*100.0)+' with loss = '+str(avg_loss))






		























