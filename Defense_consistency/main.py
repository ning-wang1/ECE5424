import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import imblearn
from sklearn.semi_supervised import LabelSpreading

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
# Settings
pd.set_option('display.max_columns', None)
#np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=3)
sns.set(style="darkgrid")
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

print("pandas : {0}".format(pd.__version__))
print("numpy : {0}".format(np.__version__))
print("matplotlib : {0}".format(matplotlib.__version__))
print("seaborn : {0}".format(sns.__version__))
print("sklearn : {0}".format(sklearn.__version__))
print("imblearn : {0}".format(imblearn.__version__))

# user define library
from NSL_setup import NSL_KDD
import classifier as clf
import adver
from svc_adv import svm_adv

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def generate_data(test_data, test_labels, original = 1, target = 0, samples=3):
	inputs = []
	targets = []
	original_labels =[]

	for i in range(samples):
		idx = np.random.randint(0,test_labels.shape[0]-1)
		while np.argmax(test_labels[idx]) != original:
			idx += 1
		inputs.append(test_data[idx])
		targets.append(np.eye(data.test_labels.shape[1])[target])
		original_labels.append(test_labels[idx])

	inputs = np.array(inputs)
	targets = np.array(targets)
	original_labels = np.array(original_labels)
	return inputs, targets, original_labels


if __name__ == "__main__":
	# Possible seletion
	classifiers = {'KNN', 'LGR', 'BNB', 'DTC'} # ML models
	attack_list = [('DoS', 0.0), ('Probe', 2.0), ('R2L', 3.0), ('U2R', 4.0)] # attack classes


	# experiment setting (from possible selection)
	attackclass = [('DoS', 0.0)]

	# data preprocessing and data partition
	data = NSL_KDD(attackclass)

	# Model training and model evaluation
	#model = clf.classifier(classifier_name, train_data, train_labels)
	#clf.evaluate(classifier_name, model, test_data, test_labels)

	# to deal with memory error (use small subset of data)
	train_data = data.train_data[0:10000,:]
	test_data = data.test_data[0:300,:]
	train_labels_one_hot = data.train_labels[0:10000]
	test_labels_one_hot = data.test_labels[0:300]
	train_labels = np.argmax(train_labels_one_hot,1)
	test_labels = np.argmax(test_labels_one_hot,1)

	x_all = np.concatenate((train_data, test_data)) # concatenate the train and test data (for structure exploitation)
	test_labels_none = -1*np.ones([test_labels.shape[0],]) # the label of the test_data is set to -1
	y_all = np.concatenate((train_labels,test_labels_none)) # concatenate the train labels and -1 test labels

	consist_model = LabelSpreading(gamma=4, max_iter=60) 
	consist_model.fit(x_all, y_all)
	clf.evaluate_sub('consistency model', test_labels, consist_model.predict(test_data))

	lgr_model = clf.classifier('LGR', train_data, train_labels)
	clf.evaluate('LGR', lgr_model, test_data, test_labels)

	knn_model = clf.classifier('KNN', train_data, train_labels)
	clf.evaluate('KNN', knn_model, test_data, test_labels)

	bnb_model = clf.classifier('BNB', train_data, train_labels)
	clf.evaluate('BNB', bnb_model, test_data, test_labels)

	svm_model = clf.classifier('SVM', train_data, train_labels)
	clf.evaluate('SVM', svm_model, test_data, test_labels)

	dtc_model = clf.classifier('DTC', train_data, train_labels)
	clf.evaluate('DTC', dtc_model, test_data, test_labels)

	model_to_attack = clf.classifier('MLP', train_data, train_labels)
	# the number of adversarial examples that models can resist to
	consist_num = 0
	lgr_num = 0
	knn_num = 0
	svm_num = 0
	dtc_num = 0
	bnb_num = 0
	mlp_num = 0

	# adversarial examples crafting target MLP
	# for i in range(300):
	# 	# adversarial examples targeting MLP using GD
	# 	adv, idx= adver.sneaky_generate(0,1,model_to_attack, test_data, test_labels) # 0 is the target, 1 is the origin label
	# 	print(bnb_model.predict(test_data[idx].reshape(1,-1)))
	# 	consist_num += consist_model.predict(adv)
	# 	lgr_num += lgr_model.predict(adv)
	# 	knn_num += knn_model.predict(adv)
	# 	svm_num += svm_model.predict(adv)
	# 	dtc_num += dtc_model.predict(adv)
	# 	bnb_num += bnb_model.predict(adv)
	# 	mlp_num += model_to_attack.predict(adv)

	# print('the number of adversarial example the consistency model resist to : ', consist_num)
	# print('the number of adversarial example the LGR model resist to : ', lgr_num)
	# print('the number of adversarial example the KNN model resist to : ', knn_num)
	# print('the number of adversarial example the SVM model resist to : ', svm_num)
	# print('the number of adversarial example the DTC model resist to : ', dtc_num)
	# print('the number of adversarial example the BNB model resist to : ', bnb_num)
	# print('the number of adversarial example the MLP model resist to : ', mlp_num)

	#adversarial examples crafting target SVM
	for i in range(10):
		# Adversarial exaples targeting SVM using L-BFGS
		adv = svm_adv(1,0, knn_model, test_data, test_labels ) # 1 is the  target, 0 is the original label
		consist_num += consist_model.predict(adv)
		lgr_num += lgr_model.predict(adv)
		knn_num += knn_model.predict(adv)
		svm_num += svm_model.predict(adv)
		dtc_num += dtc_model.predict(adv)
		bnb_num += bnb_model.predict(adv)
		mlp_num += model_to_attack.predict(adv)


	print('the number of adversarial example misclassified by the consistency model : ', consist_num)
	print('the number of adversarial example misclassified by the LGR model : ', lgr_num)
	print('the number of adversarial example misclassified by the KNN model : ', knn_num)
	print('the number of adversarial example misclassified by the SVM model : ', svm_num)
	print('the number of adversarial example misclassified by the DTC model : ', dtc_num)
	print('the number of adversarial example misclassified by the BNB model : ', bnb_num)
	print('the number of adversarial example misclassified by the MLP model : ', mlp_num)



















	# from attack_l2 import CarliniL2
	# import tensorflow as tf
	# from NSL_setup import NNModel
	# import time

	# with tf.Session() as sess:
	# 	features_num = train_data.shape[1]
	# 	model_to_attack = NNModel("models/nsl_kdd",features_num, sess)
	# 	attack = CarliniL2(sess, model_to_attack,data.min_v,data.max_v, batch_size=1, max_iterations=1000, confidence=0)
	# 	# #attack = CarliniL0(sess, model, max_iterations=1000, initial_const=10,
	# 	# #				   largest_const=15)
	# 	inputs, targets, original_labels = generate_data(test_data,test_labels_one_hot,samples = 1)

	# 	print('shape>>>>>>>>>>>>>>>>>>')
	# 	print('input>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',inputs)
	# 	print(targets.shape)
	# 	print(original_labels.shape)

	# 	timestart = time.time()
	# 	adv = attack.attack(inputs, targets)
	# 	timeend = time.time()

	# 	# print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")

	# 	# for i in range(len(adv)):		

	# 	# 	print("Valid:")
	# 	# 	print('Original Labels: ', original_labels[i], ' Class: ', np.argmax(original_labels))
	# 	# 	#show(inputs[i])			

	# 	# 	print("Adversarial:")
	# 	# 	#show(adv[i])
	# 	# 	outputs = softmax_exp(model.model.predict(adv[i:i+1])[0])
	# 	# 	print("Classification before softmax: ", outputs)
	# 	# 	print("After softmax: ", outputs, ' Class: ', np.argmax(outputs))

	# 	# 	print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)



	# 