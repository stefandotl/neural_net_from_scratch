
import numpy as np

inputs = [1, 2, 3]
weights = [1, 2, 3]

bias = [5]

goal = 17

lr = 0.0005
epochs=10

output = 0
endsum =0

for epoch in range(10):
	endsum = 0
	for j, num in enumerate(inputs):
		output_j = inputs[j]*weights[j]
		endsum += output_j

	# print(endsum)

	loss = round((goal - endsum), 5)

	for k in range(len(weights)):
		# print("old Weight: ", weights[k])
		weights[k] = lr*loss*inputs[k]
		# print("New Weight: ", weights[k])

	print("epoch: ", epoch, " loss: ", loss, "output: ", endsum)




