import numpy as np 
def sigmoid(x):
	return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
	return sigmoid(x)*(1-sigmoid(x))
learnrate=0.5
x=np.array([2,3,4,5])
y=np.array(0.5)
weights=np.array([0.5,-0.5,0.3,0.1])
h=np.dot(weights,x)
output=sigmoid(h)
error = y-output
output_grad=sigmoid_derivative(h)
error_term = error * output_grad
del_w=error_term* learnrate* x
print('Neural Network output:')
print(output)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)