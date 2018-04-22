import numpy as np
import tensorflow as tf



class Cnn_Classifier:

	def __init__(self,classifier_name,input_shape = [None,28,28,1],conv_layers = [(64,7),(128,5),(128,3)],pooling_layers = [1,0,1], batchnorm_layers = [1,1,1],activation_type = "relu",fc_size = 2048, learning_rate = 0.0003, output_num = 10):


		self.weights_list = []
		self.biases_list = []
		self.layers_num = len(conv_layers)

		current_channels = input_shape[3]

		for layer in conv_layers:
			current_weights = self.weights_variable([layer[1],layer[1],current_channels,layer[0]])
			current_bias = self.bias_variable([1,layer[0]])
			current_channels = layer[0]
			self.weights_list.append(current_weights)
			self.biases_list.append(current_bias)

		self.classifier_name = classifier_name
		self.pooling_layers = pooling_layers
		self.batchnorm_layers = batchnorm_layers
		self.activation_type = activation_type
		self.fc_size = fc_size
		self.learning_rate = learning_rate
		self.output_num = output_num

		num =1
		for i in pooling_layers:
			if i ==1:
				num*=2


		self.fc_wights = self.weights_variable([int((input_shape[1]/num)*(input_shape[2]/num)*conv_layers[-1][0]),output_num])

		self.fc_bias = self.bias_variable([1,output_num])
		self.batch_size=1000

		self.sess = None

	def get_classifer_name(self):
		return self.classifier_name

	def set_batch_size(self,new_batch_size):
		self.batch_size = new_batch_size

	def weights_variable(self,shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial) 


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
		sess = tf.Session()

		results = sess.run(output_encoded)
		sess.close()
		return results


	def train_classifier(self,input_images,output_classes,iterations_num,encode = True):
		#import ipdb;ipdb.set_trace()
		images_num,image_width,image_hight,channel_num = input_images.shape

		self.Input = tf.placeholder(tf.float32,[None,image_width,image_hight,channel_num])
		self.Output = tf.placeholder(tf.float32,[None,self.output_num])
		input_layer = self.Input
		curr_pool_layer = None
		curr_batchnorm_layer = None

		acc_75= False
		acc_90 = False
		acc_99 = False
		acc_996 = False
		acc_998 = False
		acc_999 = False
		acc_9992 = False
		acc_9994 = False
		acc_9996 = False
		acc_9998 = False
		acc_9999 = False
		print('')
		print('')
		print('layers stacked dimensions is as following:')
		print('')
		for i in range(self.layers_num):

			curr_conv_layer = tf.nn.conv2d(input = input_layer,filter = self.weights_list[i], strides = [1,1,1,1],padding = 'SAME',name = 'conv_'+str(i)) + self.biases_list[i] 
			if self.activation_type == "relu":
				curr_activation_layer = tf.nn.relu(curr_conv_layer)
			elif self.activation_type == "sigmoid":
				curr_activation_layer = tf.nn.sigmoid(curr_conv_layer)
			elif self.activation_type == "tanh":
				curr_activation_layer = tf.nn.tanh(curr_conv_layer)
			else:
				curr_activation_layer = curr_conv_layer


			print(curr_conv_layer)
			if self.pooling_layers[i] == 1:

				curr_pool_layer = tf.nn.max_pool(value = curr_activation_layer, ksize = [1,2,2,1],strides = [1,2,2,1], padding = 'SAME', name = 'pool_'+str(i))
			else:
				curr_pool_layer = curr_activation_layer


			if self.batchnorm_layers[i] ==1 :
				curr_batchnorm_layer = tf.contrib.layers.batch_norm(inputs = curr_pool_layer)
			else:
				curr_batchnorm_layer = curr_pool_layer

			print(curr_batchnorm_layer)

			input_layer = curr_batchnorm_layer


		flatten_layer = tf.contrib.layers.flatten(curr_batchnorm_layer)

		self.Ylogits = tf.nn.softmax(tf.matmul(flatten_layer,self.fc_wights)+self.fc_bias)
		self.cross_entropy=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.Ylogits,labels=self.Output))
		self.optimizer= tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cross_entropy)

		if encode == True:
			output_encoded = self.one_hot(output_classes)
		else:
			output_encoded = output_classes

		correct_prediction = tf.equal(tf.argmax(self.Ylogits,1), tf.argmax(self.Output,1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

		print('Start training'+self.classifier_name+'...')

		number_of_batches = images_num/self.batch_size

		for it in range(iterations_num):
			#import ipdb;ipdb.set_trace()
			avg_accuracy = 0.0
			avg_loss = 0.0

			for i in range(int(number_of_batches)):

				input_batch = input_images[i*self.batch_size:(i+1)*self.batch_size]
				output_batch  = output_encoded[i*self.batch_size:(i+1)*self.batch_size].reshape(-1,self.output_num)
				results= self.sess.run([self.optimizer,self.accuracy,self.cross_entropy],feed_dict={self.Input:input_batch,self.Output: output_batch})

				avg_accuracy += results[1]
				avg_loss += results[2]

				#print('the accuracy of current batch '+str(i)+'/'+str(number_of_batches)+' with accuacy: %'+str(results[1]*100))


			avg_accuracy /=number_of_batches
			avg_loss /= number_of_batches

			if avg_accuracy > 0.75 and acc_75 == False:
				self.learning_rate = self.learning_rate /10
				acc_75 = True
			elif avg_accuracy > 0.90 and acc_90 == False:
				self.learning_rate = self.learning_rate /10
				acc_90 = True
			elif avg_accuracy > 0.99 and acc_99==False:
				self.learning_rate = self.learning_rate/10
				acc_99 = True
			elif avg_accuracy > 0.996 and acc_996 ==False:
				self.learning_rate = self.learning_rate/10
				acc_996 = True
			elif avg_accuracy > 0.998 and acc_998 ==False:
				self.learning_rate = self.learning_rate/10
				acc_998 = True
			elif avg_accuracy > 0.999 and acc_999 == False:
				self.learning_rate = self.learning_rate/10
				acc_999 = True
			elif avg_accuracy > 0.9992 and acc_9992 == False:
				self.learning_rate = self.learning_rate/10
				acc_9992 = True
			elif avg_accuracy > 0.9994 and acc_9994 == False:
				self.learning_rate = self.learning_rate/10
				acc_9994 = True
			elif avg_accuracy > 0.9996 and acc_9996 == False:
				self.learning_rate = self.learning_rate/10
				acc_9996 = True
			elif avg_accuracy > 0.9998 and acc_9998 == False:
				self.learning_rate = self.learning_rate/10
				acc_9998 = True
			elif avg_accuracy > 0.9999 and acc_9999 == False:
				self.learning_rate = self.learning_rate/10
				acc_9999 = True



			print('the accuracy of the epoch '+str(it)+' is : %'+str(avg_accuracy*100.0)+' with loss = '+str(avg_loss))

		print('End Training'+self.classifier_name+'.')


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


		print("Start Testing"+self.classifier_name+"...")
		for i in range(int(number_of_batches)):

			input_batch =  input_images[i*self.batch_size:(i+1)*self.batch_size].reshape(-1,28,28,1)
			output_batch = output_encoded[i*self.batch_size:(i+1)*self.batch_size].reshape(-1,10)

			results= self.sess.run([self.accuracy,self.cross_entropy],feed_dict={self.Input:input_batch,self.Output: output_batch})
			avg_accuracy += results[0]
			avg_loss += results[1]

			print('the accuracy of current batch: %'+str(results[0]*100))

		avg_accuracy /=number_of_batches
		avg_loss /= number_of_batches
		print('the accuracy of the test data is : %'+str(avg_accuracy*100.0)+' with loss = '+str(avg_loss))

		print('End Testing'+self.classifier_name+'.')




		























