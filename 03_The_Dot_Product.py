
import numpy as np

inputs = [[1, 2, 3, 2.5],
			[2.0, 5.0, -1.0, 2.0],
			[-1.5, 2.7, 3.3, -0.8]]


weights = [[0.2, 0.8, -0.5, 1.0],
			[0.5, -0.91, 0.26, -0.5],
			[-0.26, -0.27, 0.17, 0.87]]

weights_2 = [[0.2, 0.8, -0.5],
			[0.5, -0.91, 0.26],
			[-0.26, -0.27, 0.17]]

biases = [2, 3, 0.5]

biases_2 = [2, 3, 0.5]


output = np.dot(inputs, np.array(weights).T) + biases

inputs_2 = output

output_2 = np.dot(inputs_2, np.array(weights_2).T) + biases_2

# out_0 = output[0]
# out_1 = output[1]
# out_2 = output[2]

# print(output)
print(output_2)