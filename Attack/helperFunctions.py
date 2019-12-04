# Helper functions

import numpy as np
import pickle
import os
import struct
from array import array
from matplotlib import pyplot as plt
import urllib.request
import gzip

# A function that loads data from a data set
def load(path_img, path_lbl):
	with open(path_lbl, 'rb') as file:
		magic, size = struct.unpack(">II", file.read(8))
		if magic != 2049:
			raise ValueError('Magic number mismatch, expected 2049,'
			'got %d' % magic)

		labels = array("B", file.read())

	with open(path_img, 'rb') as file:
		magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
		if magic != 2051:
			raise ValueError('Magic number mismatch, expected 2051,'
			'got %d' % magic)

		image_data = array("B", file.read())

	images = []
	for i in range(size):
		images.append([0]*rows*cols)

	for i in range(size):
		images[i][:] = image_data[i*rows*cols : (i+1)*rows*cols]

	return images, labels



# Make MNIST data set easy to use
'''
trData:   60000 by 784 array of unsigned int
trLabels: 60000 by 1   array of 0 or 1
teData:   10000 by 784 array of unsigned int
teLabels: 10000 by 1   array of 0 or 1
'''

def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*28*28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images,28*28)
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
        print(labels.shape)

    #return (np.arange(10) == labels[:, None]).astype(np.float32)
    return labels

def getMnistDataSet():
	if not os.path.exists("data"):
		os.mkdir("data")
		files = ["train-images-idx3-ubyte.gz",
				 "t10k-images-idx3-ubyte.gz",
				 "train-labels-idx1-ubyte.gz",
				 "t10k-labels-idx1-ubyte.gz"]
		for name in files:

			urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/"+name)

	trData = extract_data("data/train-images-idx3-ubyte.gz", 60000)
	trLabels = extract_labels("data/train-labels-idx1-ubyte.gz", 60000)
	teData = extract_data("data/t10k-images-idx3-ubyte.gz", 10000)
	teLabels = extract_labels("data/t10k-labels-idx1-ubyte.gz", 10000)
	return trData, trLabels, teData, teLabels



# Show the image
'''
imgArray: 1 by 784 array of unsigned int items
Example: fun.imgDisp(trData[1][:])
'''
def imgDisp(imgArray):
	image = np.reshape(imgArray,(28,28))
	img = plt.imshow(image, cmap='gray')
	plt.show(img)


# Activation function for a neuron
'''
sumInputs: A numpy array of summed inputs for all neurons in one layer
actfun:   Activation function
'''
def act(sumInputs, actfun):
	if actfun == 'logistic':
		return 1/(1+np.exp(-sumInputs))
	elif actfun == 'relu':
		return (sumInputs+np.abs(sumInputs))/2
	elif actfun == 'identity':
		return sumInput
	elif actfun == 'tanh':
		return np.tanh(sumInputs)



# Softmax function
def softmax(x):
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()



# Calculate the (y+1)th column of the Jacobian w.r.t. x. (ie. the corresponding gradient)
'''
x: 1 x nInp array
y: scalar
W1: nInp x nHid array
W2: nHid x nCls array
'''
def compGradient_h1(x, c, W1, W2, b1, b2):
	nInp = len(x)
	nHid, nCls = np.shape(W2)
	X = np.array(x)
	W1 = np.array(W1)
	W2 = np.array(W2)
	W2_trans = np.transpose(W2)
	
	Z = np.dot(X,W1) + b1	  # weighted sum of layer 1
	Y = 1/(1+np.exp(-Z))	   # sigmoid output of the hidden layer
	S = np.dot(Y,W2) + b2	  # weighted sum of layer 2
	L = softmax(S)			 # Likelihood of every class

	D = np.zeros(nCls)
	A = np.dot(np.multiply(np.multiply(Y,1-Y),W1) , np.transpose(W2_trans))

	for k in range(nCls):
		if k == c:
			D[k] = L[c]*(1-L[c])
		else:
			D[k] = -L[k]*L[c]

	out = np.dot(D,np.transpose(A))
	'''
	# The following is the non-matrix version which runs much slower
	for i in range(nInp):
		tmp = 0
		for k in range(nCls):
			if k == c:
				D = L[c]*(1-L[c])
			else:
				D = -L[k]*L[c]
			for j in range(nHid)
				tmp = tmp + D * W2[j][k] * Y[j] * (1 - Y[j]) * W1[i][j]
		out.append(tmp)
	'''
	return out


# Get the the first n data points for each class label
def sbst_init_tr(data,labels,n):
	out = []
	count = np.zeros(10)
	count1 = 0
	for i in range(len(labels)):
		k = labels[i]
		if count[k] < n:
			out.append(data[i])
			count[k] = count[k] + 1
			count1 = count1 + 1
		if count1 >= 10*n:
			return out
	return out


# Estimate labels for given test data using forward-propagation for NNs with 1 hidden layer
def NN_pred_h1(X_test, W1, W2, b1, b2):
	Z = np.dot(X_test,W1) + b1   	# weighted sum vector of layer 1
	Y = 1/(1+np.exp(-Z))	   		# sigmoid output vector of the hidden layer
	T = np.dot(Y,W2) + b2	  		# weighted sum vector of layer 2
	estLabels = np.argmax(T, axis=1)
	return estLabels


# Estimate labels for given test data using forward-propagation
'''
For NNs with arbitrary number of hidden layers
model: Example: dict([(0,Oracle['W1']),(1,Oracle['b1']),(2,Oracle['W2']),(3,Oracle['b2']),...])
'''
def NN_pred(X_test, model, actfun):
	nHid = int(len(model) / 2 - 1)		 # number of hidden layers
	Z = np.dot(X_test,model[0]) + model[1]	# weighted sum vector before the 1st hidden layer
	Y = act(Z, actfun)	   			# activated output of the first hidden layer
	
	for l in range(nHid-1):
		Z = np.dot(Y,model[2*l+2]) + model[2*l+3]   # weighted sum vector before this hidden layer
		Y = act(Z, actfun)		 			  # sigmoid output of this hidden layer
	
	T = np.dot(Y,model[2*nHid]) + model[2*nHid+1]	  # weighted sum vector of layer 2
	estLabels = np.argmax(T,axis=1)
	return estLabels

# Compute accuracy score
def NN_score(estLabels, Y_test):
	score = 1 - sum(np.sign(np.abs(estLabels-Y_test))) / len(Y_test)
	return score



