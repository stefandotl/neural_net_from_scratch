
import numpy as np
import random
import nnfs
from nnfs.datasets import spiral_data  # See for code: https://gist.github.com/Sentdex/454cb20ec5acf0e76ee8ab8448e6266c

# for replication
nnfs.init()		# like: np.random.seed(0) ?

X, y = spiral_data(100, 3)

# X = inputs!
X = [[1, 2, 3, 2.5],
	[2.0, 5.0, -1.0, 2.0],
	[-1.5, 2.7, 3.3, -0.8]]


class Layer_Dense(object):
	"""docstring for Layer"""
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))
		# self.biases = np.random.rand(n_inputs, n_neurons)
		# print('#############################')
		# print('inputs:', self.weights)

	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
	def forward(self, inputs):
		self.output = np.maximum(0, inputs)


# layer_1 = Layer_Dense(4, 5)		# 4 len of X, 3 is egal, can be any number!
# layer_2 = Layer_Dense(5, 1)		# has to be same input the as output number in previousd layer
# #
# layer_1.forward(X)
# print(layer_1.output)
# layer_2.forward(layer_1.output)
# print(layer_2.output)

# def relu_own_version(np_array):
# 	a = np_array
# 	a[a < 0] = 0  	# all elements smaller than 0
# 	return a
#
#
# inputs = np.ones([1, 3])
# weights = np.ones([3, 2])		# second number: number of neurons
# biases = np.array([1, 2])		# second number: number of neurons
# result = np.dot(inputs, weights) + biases

# print(weights)
# print(biases.shape)
# print(result)
# print(relu_own_version(result))