# Whilte-box Attack Algorithms

import numpy as np
from scipy.optimize import minimize
import helperFunctions as fun
from matplotlib import pyplot as plt


# Fast Gradient Sign to Generate Adversarial Samples - Goodfellow et al.
'''
x: Original data
sbst: substitute model dict
'''
def FastGradSign_h1(x, epsilon, grad_th, sbst):
	W1 = sbst['W1']
	b1 = sbst['b1']
	W2 = sbst['W2']
	b2 = sbst['b2']
	actfun = sbst['act']

	nInp = len(x)
	nHid, nCls = np.shape(W2)
	X = np.array(x)
	W1 = np.array(W1)
	W2 = np.array(W2)
	W2_trans = np.transpose(W2)
	
	Z = np.dot(X,W1) + b1      # weighted sum of layer 1
	Y = fun.act(Z, actfun)         # sigmoid output of the hidden layer
	S = np.dot(Y,W2) + b2      # weighted sum of layer 2
	L = fun.softmax(S)             # Likelihood of every class		 # softmax output vector

	A = np.dot(np.multiply(np.multiply(Y,1-Y),W1) , np.transpose(W2_trans))

	out = np.zeros((nCls,nInp))

	for c in range(nCls):      # Craft samples for all target classes
		T = np.zeros(nCls)
		T[c] = 1

		G = (1-T)/(1-L) - T/L
		
		D = np.zeros(nCls)	
		for k in range(nCls):
			if k == c:
				D[k] = L[k]*(1-L[c])
			else:
				D[k] = -L[k]*L[c]
		D = np.repeat(np.array([D]), nCls, axis=0)
		Gradient = np.dot(G, np.dot(D, np.transpose(A)))

		for i in range(nInp):
			if Gradient[i] <= grad_th and Gradient[i] >= -grad_th:
				Gradient[i] = 0
		out[c] = x + epsilon * np.sign(Gradient)

	out = np.clip(out, 0, 256)
	return out



# Iterative Gradient Sign to Generate Adversarial Samples - Kurakin et al.
'''
x: Original data
sbst: substitute model dict
'''
def IterativeGradSign_h1(x, epsilon, grad_th, sbst):
	W1 = sbst['W1']
	b1 = sbst['b1']
	W2 = sbst['W2']
	b2 = sbst['b2']
	actfun = sbst['act']

	nInp = len(x)
	nHid, nCls = np.shape(W2)
	X = np.array(x)
	W1 = np.array(W1)
	W2 = np.array(W2)
	W2_trans = np.transpose(W2)
	
	Z = np.dot(X,W1) + b1      # weighted sum of layer 1
	Y = fun.act(Z, actfun)         # sigmoid output of the hidden layer
	S = np.dot(Y,W2) + b2      # weighted sum of layer 2
	L = fun.softmax(S)             # Likelihood of every class		 # softmax output vector

	A = np.dot(np.multiply(np.multiply(Y,1-Y),W1) , np.transpose(W2_trans))

	out = np.zeros((nCls,nInp))

	for c in range(nCls):      # Craft samples for all target classes
		T = np.zeros(nCls)
		T[c] = 1

		G = (1-T)/(1-L) - T/L
		
		D = np.zeros(nCls)	
		for k in range(nCls):
			if k == c:
				D[k] = L[k]*(1-L[c])
			else:
				D[k] = -L[k]*L[c]
		D = np.repeat(np.array([D]), nCls, axis=0)
		Gradient = np.dot(G, np.dot(D, np.transpose(A)))

		for i in range(nInp):
			if Gradient[i] <= grad_th and Gradient[i] >= -grad_th:
				Gradient[i] = 0
		out[c] = x + epsilon * np.sign(Gradient)

	out = np.clip(out, 0, 256)
	return out



# Optimization with L-BFGS - Szegedy et al.
'''
x: Original data
sbst: substitute model dict
'''
def Opt_L_BFGS_h1(x, sbst, coeff):
	W1 = sbst['W1']
	b1 = sbst['b1']
	W2 = sbst['W2']
	b2 = sbst['b2']
	actfun = sbst['act']

	nInp = len(x)
	nHid, nCls = np.shape(W2)
	W1 = np.array(W1)
	W2 = np.array(W2)

	out = np.zeros((nCls,nInp))

	for c in range(nCls):      # Craft samples for all target classes
		objfun = lambda x_adv: coeff * np.linalg.norm(x_adv-x) - np.log(fun.softmax(np.dot(fun.act(np.dot(x_adv,W1) + b1, actfun),W2) + b2)[c]) 
		bnds = np.concatenate((np.zeros((nInp,1)),np.ones((nInp,1))), axis=1)
		optRes = minimize(objfun, x, method='L-BFGS-B', bounds=bnds)
		out[c] = optRes.x
	return out






# Optimization with L2 distortion metric to Generate Adversarial Samples - Carlini et al.
'''
x: Original data
sbst: substitute model dict
'''
'''
def OptL2_h1(x, c, sbst):
'''



# Test a Specific data sample with Fast Gradient Sign as the crafting method
'''
x: a data record, 1 by 784
y: x's true label
'''
def testOneSample(x, y, epsilon, grad_th, coeff, sbst, sbst_model, Oracle, Oracle_model, sbst_pred, Oracle_pred):
	print('Test on a sample with true label ',y)
	X_adv = FastGradSign_h1(x, epsilon, grad_th, sbst)
	#X_adv = Opt_L_BFGS_h1(x, sbst, coeff)

	Pred_Adv_sbst   = fun.NN_pred(X_adv,  sbst_model,   sbst['act'])
	Pred_Adv_Oracle = fun.NN_pred(X_adv,  Oracle_model, Oracle['act'])

	print('Sbst   - Prediction of the Original Sample:  ', sbst_pred)
	print('Oracle - Prediction of the Original Sample:  ', Oracle_pred)
	print('Sbst   - Prediction of the Adversarial Samples: ', Pred_Adv_sbst)
	print('Oracle - Prediction of the Adversarial Samples: ', Pred_Adv_Oracle)

	fig = plt.figure(figsize=(14, 1.2))
	for idx in range(10):	
		fig.add_subplot(1,10,idx+1)
		plt.imshow(np.reshape(X_adv[idx],(28,28)), cmap='gray')
		plt.axis('off')
	plt.show()