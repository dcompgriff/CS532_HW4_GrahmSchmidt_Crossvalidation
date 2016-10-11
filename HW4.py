import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')


fisherData = []

def main():
	#part1()
	part2()

'''
Answers to HW4 Q4.
'''
def part2():
	global fisherData
	global X, y, what
	print('Running HW4.py main...')
	fisherData = np.loadtxt('./fisher.csv', delimiter=',')
	# print(fisherData)
	X = np.hstack((np.ones((fisherData.shape[0], 1)), fisherData[:, :-1]))
	y = fisherData[:, -1]
	#w = linalg.lstsq(X, y)
	what = linalg.inv(np.dot(X.T,X)).dot(X.T).dot(y).reshape((X.shape[1],1))
	yPredicted = distancePrediction(X.dot(what), [-1, 0, 1])
	print(yPredicted)

'''
Compare the distance of a prediction to each one of the labels, and
assign the label that is the closest.
'''
def distancePrediction(y, labels):
	yPred = []
	for yVal in y:
		minDist = linalg.norm(yVal - labels[0])
		currentPred = labels[0]
		for label in labels:
			if abs(yVal - label) < minDist:
				currentPred = label
		yPred.append(currentPred)
	return np.array(yPred)



'''
Answers to HW4 Q2 and Q3.
'''
def part1():
	A = np.array([[3, 1],
	     [0, 3],
	     [0, 4]])
	A2 = np.array([[3, 1, 2],
	      [0, 3, 3],
	      [0, 4, 4],
	      [6, 1, 4]])
	A3 = np.array([[1, 1, 2],
	      [0, 3, 3],
	      [0, 4, 4],
	      [3, 1, 4]])
	print('Problem 2 Ans:')
	print(gramSchmidt(A))
	print('Problem 3 Ans:')
	print('Rank of \n' + str(A2) + '\nis ' + str(gramSchmidt(A2).shape[1]))
	print(gramSchmidt(A2))
	print('Numpy gives: ' + str(linalg.matrix_rank(A2)))
	print('Rank of \n' + str(A3) + '\nis ' + str(gramSchmidt(A3).shape[1]))
	print(gramSchmidt(A3))
	print('Numpy gives: ' + str(linalg.matrix_rank(A3)))


def gramSchmidt(A, epsilon=1.0e-10):
	if type(A) == list:
		A = np.array(A)
	#Set v1
	U = (1/linalg.norm(A[:, 0]))*A[:, 0].reshape((A.shape[0], 1))
	for col in range(1, A.shape[1]):
		v = A[:, col]
		for uCol in range(0, U.shape[1]):
			projCoeff = np.dot(U[:, uCol], v)/(linalg.norm(U[:, uCol])**2)
			v = v - (projCoeff*U[:, uCol])

		isZeroVector = True
		for elem in v.ravel():
			if abs(elem) > epsilon:
				isZeroVector = False
				break

		if isZeroVector:
			continue
		else:
			v = (1 / linalg.norm(v)) * v.reshape((v.size, 1))
			U = np.hstack((U, v))

	return U



















if __name__ == '__main__':
	main()







