# Training the Substitute Model - Neural Network based
#
import numpy as np
from sklearn.neural_network import MLPClassifier
import helperFunctions as fun
import warnings

warnings.filterwarnings("ignore")

# Substitute Model Parameters
sbst_initSamples = 20				  # number of samples for each class in the initial substitute training set
sbst_maxEpoch = 6     		  		  # max number of augmentation rounds
sbst_maxIter = 10     		          # max number of iterations in each augmentation epoch
sbst_lambda = 0.1     		  		  # lambda the augmentation factor
sbst_lrate = 0.01     		  		  # learning rate
sbst_mmt = 0.9      				  # momentum
sbst_solver = 'adam'
sbst_act = 'logistic'

# Load MNIST data
mnist = np.load('mnist.npz')
X_test = mnist['teData']        	  # All 10,000 test data points
Y_test = mnist['teLabels']      	  # All 10,000 test labels
sbst_X = fun.sbst_init_tr(X_test, Y_test, sbst_initSamples)  # initial training set for the substitute model
print("Data loading completed.")

#########################
##     Load Oracle     ##
################################################################################################
'''
Oracle = np.load('Oracle_h1.npz')   # 1 hidden layer
Oracle_model = dict([(0,Oracle['W1']),(1,Oracle['b1']),(2,Oracle['W2']),(3,Oracle['b2'])])
''
Oracle = np.load('Oracle_h2.npz')   # 2 hidden layers
Oracle_model = dict([(0,Oracle['W1']),(1,Oracle['b1']),(2,Oracle['W2']),(3,Oracle['b2']),
					 (4,Oracle['W3']),(5,Oracle['b3'])])
'''
Oracle = np.load('Oracle_h3.npz')   # 3 hidden layers
Oracle_model = dict([(0,Oracle['W1']),(1,Oracle['b1']),(2,Oracle['W2']),(3,Oracle['b2']),
					 (4,Oracle['W3']),(5,Oracle['b3']),(6,Oracle['W4']),(7,Oracle['b4'])])
################################################################################################
Oracle_act = Oracle['act']          # oracle's activation function

Oracle_score = fun.NN_score(fun.NN_pred(X_test, Oracle_model, actfun=Oracle_act),Y_test)
print("Oracle Score: ", Oracle_score)

# Training the substitute model with the algorithm in [1]
sbstClf = MLPClassifier(solver=sbst_solver, alpha=1e-4, hidden_layer_sizes=(80), 
	activation=sbst_act, random_state=1, learning_rate_init=sbst_lrate,
	momentum=sbst_mmt, max_iter=sbst_maxIter)

sbst_Y_pred = []

for epoch in range(sbst_maxEpoch+1):
	
	# Label the current training set using the oracle
	sbst_Y = fun.NN_pred(sbst_X, Oracle_model, actfun=Oracle_act)
	# Train the substitute with the oracle labels
	sbstClf.fit(sbst_X,sbst_Y)
	# Test the current substitute model
	tmp_score = sbstClf.score(X_test,Y_test)
	print('Epoch ', epoch,' Sbst Score: ', round(tmp_score,4))
	if epoch == sbst_maxEpoch:
		continue
	# Perform Jacobian-based dataset augmentation
	W1 = sbstClf.coefs_[0]
	W2 = sbstClf.coefs_[1]
	b1 = sbstClf.intercepts_ [0]
	b2 = sbstClf.intercepts_ [1]
	for idx in range(len(sbst_Y)):
		Jacobian_theCol = fun.compGradient_h1(sbst_X[idx], sbst_Y[idx], W1, W2, b1, b2)
		sbst_X.append(sbst_X[idx] + sbst_lambda*np.sign(Jacobian_theCol))


# Save the substitute model
np.savez('Sbst_NN_h1', solver=sbst_solver, act=sbst_act, W1=W1, b1=b1, W2=W2, b2=b2)
