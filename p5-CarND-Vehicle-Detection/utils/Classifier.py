import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import time


#Standardization, or mean removal and variance scaling
class Classifier(object):
	def __init__(self):
		car_features = []
		notcar_features = []
		X_train = 0
		X_test = 0
		y_train = 0
		y_test = 0
		svc = 0

	def standardize(self,car_features,notcar_features):
		self.car_features = car_features
		self.notcar_features = notcar_features

		X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
		# Fit a per-column scaler
		X_scaler = StandardScaler().fit(X)
		# Apply the scaler to X
		scaled_X = X_scaler.transform(X)

		# Define the labels vector
		y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
		
		return X_scaler,scaled_X,y 

	def train_test_split(self,scaled_X, labels_vector):
		rand_state = np.random.randint(0, 100)
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(scaled_X, labels_vector, test_size=0.2, random_state=rand_state)
		return self.X_train, self.X_test, self.y_train, self.y_test

	def SVC_Normalize(self,X_train, y_train, X_test,y_test):
		t=time.time()
		svc = LinearSVC()
		svc.fit(X_train, y_train)
		t2 = time.time()
		print(round(t2-t, 2), 'Seconds to train SVC...')
		# Check the score of the SVC
		svc_Score = round(svc.score(X_test, y_test), 4)
		print('Test Accuracy of SVC = ', svc_Score)
		# Check the prediction time for a single sample
		t=time.time()
		return svc, svc_Score





