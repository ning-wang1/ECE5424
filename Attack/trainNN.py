# Train a Neural Network and Save its Model Coefficients
#

import numpy as np
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt
import helperFunctions as fun


# Load MNIST data - training size = 50000, test size = 10000
trData, trLabels, teData, teLabels = fun.getMnistDataSet()

# Convert the feature range into [0,1]
trData = trData / 256;
teData = teData / 256;

print("Data loading completed.")

###################################
##   Neural Network Parameters   ##
###################################
solver = 'adam'
act = 'relu'


# Training the oracle
Oracle = MLPClassifier(solver=solver, alpha=1e-4, hidden_layer_sizes=(60,40,20), 
					activation=act, random_state=1)


print("Oracle training started...")
Oracle.fit(trData,trLabels)  # This takes time

print(Oracle)
print("Oracle training completed.")
print("Score: ", Oracle.score(teData,teLabels))

# Save the MNIST data set we don't have to call fun.getMnistDataSet() again.
np.savez('mnist', trData=trData, trLabels=trLabels, teData=teData, teLabels=teLabels)
print("MNIST saved.")

# Save the oracle's model coefficients
W1 = Oracle.coefs_[0]
b1 = Oracle.intercepts_ [0]
W2 = Oracle.coefs_[1]
b2 = Oracle.intercepts_ [1]

W3 = Oracle.coefs_[2]
b3 = Oracle.intercepts_ [2]

W4 = Oracle.coefs_[3]
b4 = Oracle.intercepts_ [3]


#np.savez('Oracle_h1', solver=solver, act=act, W1=W1, b1=b1, W2=W2, b2=b2)

#np.savez('Oracle_h2', solver=solver, act=act, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)

np.savez('Oracle_h3', solver=solver, act=act, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, W4=W4, b4=b4)

print("Oracle model saved.")
