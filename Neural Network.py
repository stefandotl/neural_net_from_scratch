neurons = [10, -5, 7]
weights = [15.6, -7.3, 225.2]

ziel = 1
lr = 0.005
endsum = 0
epoche = 100

for a in range(epoche):
	for i in range(len(neurons)):
		result = neurons[i] * weights[i]
		endsum += result


	delta_i = round((ziel - endsum), 9)
	print("Epoche:", a+1, "loss:", abs(delta_i), "Output:", round(endsum, 1))

	# print(endsum)	

	for i in range(len(neurons)):
		weights[i] = lr * delta_i * neurons[i]

	if abs(delta_i) < 0.01:
		break

print(weights)
print('hello')