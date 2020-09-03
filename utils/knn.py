#original source code take from: https://docs.opencv.org/master/d5/d26/tutorial_py_knn_understanding.html
#documentation of cv.ml.knn: https://docs.opencv.org/master/dd/de1/classcv_1_1ml_1_1KNearest.html
#documentation of sklearn KNN: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
import cv2 as cv
import matplotlib.pyplot as plt
from scipy import spatial
import heapq

import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN

class MyKNN:
	def __init__(self):
		self.knnPoints = []
		self.knnLabels = []

	def trainWithOne(self, point, label) -> None:
		""" Add a point with his label to the KNN space. """
		if point is not None and label is not None:
			self.knnPoints.append(point)
			self.knnLabels.append(label)

	def train(self, points, labels) -> None:
		""" Add a list of points with a list of their labels to the KNN space. """
		if len(points) != len(labels):
			print("Error: len(points) != len(labels)")
		else:
			[ self.knnPoints.append(pt) for pt in points ]
			[ self.knnLabels.append(lb) for lb in labels ]


	def neighbours(self, query, k: int=5) -> "list of neighbours":
		""" Return the list of neighbour points (with len=k) that are the most similar ones to the query. """
		distances = []
		for point in self.knnPoints:
			dist = spatial.distance.sqeuclidean(query, point)
			distances.append(dist)

		topK = heapq.nsmallest(k, zip(distances, self.knnLabels, self.knnPoints))
		#print("Matches:")
		#for (i, (score, label, point)) in enumerate(topK):
		#	print("The {} match has score {:.3f} and label {}".format(i+1, score, str(label)))
		
		return topK

	def predict(self, query, k: int=5) -> "label":
		""" Return the most probable label according to the k neighbour points. """
		topK = self.neighbours(query, k)
		#print("topK:", topK)

		# list of labels of neighbour points
		lst = [el[1] for el in topK]
		#print("neighbours:", lst)
		
		# compute the max according to the occurrencies into the list
		bestMatch = max(lst, key=lst.count)
		#print("bestMatch:", bestMatch)
		return bestMatch

	def elementsInfo(self) -> None:
		print("len(knn):", len(self.knnLabels), "->", self.knnLabels)






################################################################################################
####################################   Test only Code   ########################################
################################################################################################

def addPoints(cvKnn, skKnn, myKnn, n):
	# Feature set containing (x,y) values of n known/training data
	trainData = np.random.randint(0,100,(n,2)).astype(np.float32)
	#trainData = np.array([[15., 27.], [20., 46.], [58., 99.], [38., 27.], [74., 95.], [16., 78.], [72.,  7.], [61., 33.], [ 8., 52.], [52., 36.]])

	# Label each one either Red or Blue with numbers 0 and 1
	labels = np.random.randint(0,3,(n,1)).astype(np.float32)
	#labels = np.array([[0.], [1.], [0.], [1.], [0.], [1.], [0.], [2.], [1.], [2.]])

	# Take Red neighbours and plot them
	red = trainData[labels.ravel()==0]
	plt.scatter(red[:,0],red[:,1],80,'r','^')

	# Take Blue neighbours and plot them
	blue = trainData[labels.ravel()==1]
	plt.scatter(blue[:,0],blue[:,1],80,'b','s')

	# Take Blue neighbours and plot them
	green = trainData[labels.ravel()==2]
	plt.scatter(green[:,0],green[:,1],80,'g','+')

	# Train Knn on the new points
	print("trainData:", trainData)
	print("labels:", labels)

	cvKnn.train(samples=trainData, layout=cv.ml.ROW_SAMPLE, responses=labels)#, updateBase=True) #openCV KNN
	skKnn.fit(trainData, np.ravel(labels))		#sklearn KNN
	myKnn.train(points=trainData, labels=labels)	#MyKNN
	return((cvKnn, skKnn, myKnn))

def createQuery():
	# Create a new Entry
	newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
	plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o')

	return newcomer

def classifyQuery(cvKnn, skKnn, myKnn, query, k):
	# Predict the label of the newcomer
	print("\n\n0=Red, 1=Blue, 2=green")
	print("Query:", query)

	print("________________________________\nOpenCV KNN")
	ret, results, neighbours, dist = cvKnn.findNearest(query, k) # openCV  KNN
	neighbours = [ "red" if int(l)==0 else "blue" if int(l)==1 else "green" for l in neighbours[0] ]
	print( "result:  {}".format(results) )
	print( "neighbours:  {}".format(neighbours) )
	print( "distance:  {}".format(dist) )

	print("________________________________\nSKlearn KNN")
	pred = skKnn.predict(query)
	# print probability for each class according to the k chosen near points
	print("prob for classes:", skKnn.predict_proba(query))	
	print(pred)

	print("________________________________\nMyKNN classification:")
	label = myKnn.predict(query[0], k)
	print("predicted label:", label)


def main():
	# Create the classifiers
	cvKnn = cv.ml.KNearest_create()	# openCV  KNN
	skKnn = KNN(n_neighbors=5)		# sklearn KNN
	myKnn = MyKNN()					# my KNN

	#first round of adding points
	cvKnn, skKnn, myKnn = addPoints(cvKnn, skKnn, myKnn, 10)

	query = createQuery()
	classifyQuery(cvKnn, skKnn, myKnn, query, 5)
	plt.show()

if __name__=="__main__":
	main()
