import numpy as np
import numpy.linalg as linalg
import random as rand
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():
	#part1()
	#part2()
	part2_def()

'''
Answers to HW4 Q4.
'''
def part2():
	global fisherData
	global X, y, what
	print('Running HW4.py main...')
	fisherData = np.loadtxt('./fisher.csv', delimiter=',')

	#HW4, Q4, Part a.
	X = np.hstack((np.ones((fisherData.shape[0], 1)), fisherData[:, :-1]))
	y = fisherData[:, -1]
	#w = linalg.lstsq(X, y)
	what = linalg.inv(np.dot(X.T,X)).dot(X.T).dot(y).reshape((X.shape[1],1))
	yPredicted = distancePrediction(X.dot(what), [-1, 0, 1])
	#Calculate average error. (Total - CorrectlyClassified)/ Total
	#Note, y[i]-yPredicted[i] == 0 only when y[i] == yPredicted[i]
	averageError = (X.shape[0] - list(y - yPredicted).count(0))/float(X.shape[0])
	print('Q4a, Average Error: ' + str(averageError))

	#HW4, Q4, Part b.
	# totalErrors, avgError = crossValidateError(X, y, k=40, trials=10000)
	# print('Q4b, Total Average of Errors: ' + str(avgError))

	#HW4, Q4, Part c.
	plotErrorVsTrainSetSize(X, y)

def part2_def():
	global fisherData
	global X, y, what
	print('Running HW4.py main...')
	fisherData = np.loadtxt('./fisher.csv', delimiter=',')

	# HW4, Q4, Part d.
	X = fisherData[:, :3]
	y = fisherData[:, -1]
	what = linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y).reshape((X.shape[1], 1))
	yPredicted = distancePrediction(X.dot(what), [-1, 0, 1])
	# Calculate average error. (Total - CorrectlyClassified)/ Total
	# Note, y[i]-yPredicted[i] == 0 only when y[i] == yPredicted[i]
	averageError = (X.shape[0] - list(y - yPredicted).count(0)) / float(X.shape[0])
	print('Q4c, Average Error: ' + str(averageError))

	# HW4, Q4, Part e.
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('x')
	ax.set_xlim((0, 10))
	ax.set_ylabel('y')
	ax.set_ylim((0, 10))
	ax.set_zlabel('z')
	ax.set_zlim((0, 10))
	ax.scatter(list(X[:50, 0]), list(X[:50, 1]), zs=list(X[:50, 2]), c='r')
	ax.scatter(list(X[50:100, 0]), list(X[50:100, 1]), zs=list(X[50:100, 2]), c='b')
	ax.scatter(list(X[100:150, 0]), list(X[100:150, 1]), zs=list(X[100:150, 2]), c='g')

	c1 = np.linspace(-1, 1.5, num=100)
	c2 = np.linspace(-1, 9, num=10)
	xyzList = []
	for cVal in c1:
		for c2Val in c2:
			mx = 10*cVal
			my = c2Val
			mz = 7*cVal
			xyzList.append((mx, my, mz))
	#ax.scatter(list(map(lambda item: item[0], xyzList)), list(map(lambda item: item[1], xyzList)), zs=list(map(lambda item: item[2], xyzList)), c='k')
	#plt.show()

	'''
	C represents the subspace vector span choosen by inspection from part e.
	The solution for part f is as follows:
	1) Project each (x, y, z) point onto the span by solving the equation C*<ai, bi> = <xi, yi, zi>
	using least squares for each point.
	2) Now each point is of the form <ai,bi> with associated yi. The model form is w0 + w1*ai + w2*bi = yi.
	3) Solve for the weights w0, w1, and w2 in this least squares problem.
	4) Classify the points and calculate the error as in early problem parts.
	'''
	C = [[0, 10],
	     [1, 0],
	     [0, 7]]
	projectedX = []
	for xVector in X[:, :3]:
		coordinates = linalg.lstsq(C, xVector)
		projectedX.append(list(coordinates[0]))
	projectedX = np.array(projectedX)
	Xsubspace = np.hstack((np.ones((projectedX.shape[0], 1)), projectedX))
	what = linalg.lstsq(Xsubspace, y)[0]
	what = what.reshape((what.size, 1))
	yPredicted = distancePrediction(Xsubspace.dot(what), [-1, 0, 1])
	# Calculate average error. (Total - CorrectlyClassified)/ Total
	# Note, y[i]-yPredicted[i] == 0 only when y[i] == yPredicted[i]
	averageError = (Xsubspace.shape[0] - list(y - yPredicted).count(0)) / float(Xsubspace.shape[0])
	print('Q4f, Average Error: ' + str(averageError))


def plotErrorVsTrainSetSize(X, y):
	errorAverages = []
	for k in range(1, 50):
		print('Itteration: ' + str(k))
		try:
			totalErrors, avgError = crossValidateError(X, y, k=k, trials=1000)
		except:
			print('Encountered singular matrix on iteration: ' + str(k))
			continue
		errorAverages.append((k, avgError))

	plt.plot(list(map(lambda item: item[0], errorAverages)), list(map(lambda item: item[1], errorAverages)))
	plt.scatter(list(map(lambda item: item[0], errorAverages)), list(map(lambda item: item[1], errorAverages)), color='r')
	plt.title('Classification error vs training set size.')
	plt.xlabel('Training set size.')
	plt.ylabel('Test set error.')
	plt.xticks((1, 50))
	plt.show()

def crossValidateError(X, y, k, trials):
	#HW4, Q4, Part b.
	totalErrors = 0
	for i in range(0, trials):
		xtr1, ytr1, xte1, yte1 = testTrainSplit(X[:50], y[:50], k=k)
		xtr2, ytr2, xte2, yte2 = testTrainSplit(X[50:100], y[50:100], k=k)
		xtr3, ytr3, xte3, yte3 = testTrainSplit(X[100:150], y[100:150], k=k)
		Xtrain = np.vstack((xtr1, xtr2, xtr3))
		Ytrain = np.vstack((ytr1, ytr2, ytr3))
		Xtest = np.vstack((xte1, xte2, xte3))
		Ytest = np.vstack((yte1, yte2, yte3))

		what = linalg.inv(np.dot(Xtrain.T, Xtrain)).dot(Xtrain.T).dot(Ytrain).reshape((Xtrain.shape[1], 1))
		yPredicted = distancePrediction(Xtest.dot(what), [-1, 0, 1])
		yPredicted = yPredicted.reshape((yPredicted.size, 1))
		# Calculate average error. (Total - CorrectlyClassified)/ Total
		# Note, y[i]-yPredicted[i] == 0 only when y[i] == yPredicted[i]
		error = Xtest.shape[0] - list(Ytest - yPredicted).count(0)
		totalErrors += error
	return totalErrors, totalErrors/float(trials*(50-k)*3.0)




'''
Break the data into test and train sets, with train sets of size k.
'''
def testTrainSplit(X, y, k):
	testIndexList = list(range(0, X.shape[0]))
	trainIndexList = []

	#Build train set index randomly.
	for i in range(0, k):
		randIndex = rand.randint(0, len(testIndexList) - 1)
		trainIndexList.append(testIndexList[randIndex])
		testIndexList.remove(testIndexList[randIndex])

	Xtrain = X[trainIndexList]
	Ytrain = y[trainIndexList].reshape((y[trainIndexList].size, 1))
	Xtest = X[testIndexList]
	Ytest = y[testIndexList].reshape((y[testIndexList].size, 1))

	return Xtrain, Ytrain, Xtest, Ytest




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







