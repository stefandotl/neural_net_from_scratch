
import numpy as np
import random

# X = inputs!
X =[[1, 2, 3, 2.5],
	[2.0, 5.0, -1.0, 2.0],
	[-1.5, 2.7, 3.3, -0.8]]


class Layer_Dense(object):
	"""docstring for Layer"""
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.1*np.random.randn(n_inputs, n_neurons)	
		self.biases = np.zeros((1, n_neurons))
		# self.biases = np.random.rand(n_inputs, n_neurons)
		# print('inputs:', self.weights)

	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases


layer_1 = Layer_Dense(4, 5)		# 4 len of X, 3 is egal, can be any number!
layer_2 = Layer_Dense(5, 2)		# has to be same input the as output number in previousd layer

layer_1.forward(X)
# print(layer_1.output)
layer_2.forward(layer_1.output)
print(layer_2.output)


