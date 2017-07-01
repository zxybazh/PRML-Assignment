from dataloader import *
import numpy as np
from numpy.linalg import pinv, norm
from scipy.linalg.blas import dgemm


class DiscriminativeClassifier(object):
	def __init__(self,preprocessing="", bias=False):
		self.x_train, self.y_train, self.x_test, self.y_test, self.feature_size = \
			load_data(t=preprocessing, bias=bias)
		self.train_size = len(self.y_train)
		self.test_size = len(self.x_test)

def sigmoid(x):
	return 1.0 / (1.0 + np.exp((-1)*x))


# calc gradient (and/or Hessian matrix)
# TODO: you should implement gradient descent or other optimizer to this
#		so please calculate the gradient (and/or) hessian in this function
#		to help TA undertand your code
#		notice that you are only allow to use numpy here
def calc_grad(W, X, Y):
	# W: weight matrix
	# X: input
	# Y: ground true
	
	grad, Hess = np.zeros_like(W), None
	for i in xrange(len(X)):
		grad += np.insert(X[i], 0, 1) * np.asscalar(sigmoid(np.insert(X[i], 0, 1).dot(W)) - Y[i])
	return grad, Hess

class LogisticRegression(DiscriminativeClassifier):

	def __init__(self, l2norm=1, preprocessing="z", eta=1e-4, max_epoch=30, l2_on=True):
		"""
		:param l2norm: l2 norm penalty
		:param preprocessing: preprocessing method
		:param eta: learning rate (step size)
		:param max_epoch: how many epochs are you going to train the regression (optional)
		:param l2_on: use l2 or not
		"""
		super(LogisticRegression, self).__init__(preprocessing=preprocessing, bias=True)
		self.weight = np.ones(self.feature_size + 1)
		self.L2norm = l2norm
		self.eta = eta
		self.max_epoch = max_epoch
		self.l2_on = l2_on
		# mask is to filter out the weight and discard the bias
		self.mask = np.ones_like(self.weight)
		self.mask[0] = 0

	def train(self, eps=1e-4):
		epoch = 0
		# Error rate
		ratio = -1

		while True:
			# Count Right & Wrong Number
			self.count  = [0, 0]
			
			# update weight
			update = calc_grad(self.weight, self.x_train, self.y_train)
			self.weight -= self.eta * update[0]
			if self.l2_on:
				# L2 regularization
				self.weight -= self.eta * self.L2norm * np.insert(self.weight[1:], 0, 0);


			if norm(update[0]) < self.eta * 0.1: # TODO: you should think about some early stopping scheme here
				break

			err = 0
			for i in xrange(len(self.x_train)):
				y_1 = sigmoid(self.weight.dot(np.insert(self.x_train[i], 0, 1)))
				y_0 = 1-y_1
				if (y_0 > y_1): y = 0
				else: y = 1
				if y == self.y_train[i]: self.count[1]  += 1
				else: self.count[0]  += 1
				err -= self.y_train[i]*math.log(y_1+eps) + (1-self.y_train[i])*math.log(y_0+eps)
			if self.l2_on:
				for para in self.weight[1:]:
					err += self.L2norm/2.0*para*para

			epoch += 1
			# print "epoch\t", epoch, "\ttraining loss:", err
			ratio = 100 * self.count[0] / float(len(self.y_train))
			# print "Training Error Ratio: ", ratio, "%"
			# print "-" * 59

			if epoch == self.max_epoch: break
		return (ratio, err)

	def test(self, eps=1e-4):
		# Count Right & Wrong Number
		self.count  = [0, 0]
		# Count Error Type & Number
		self.ecount = [0, 0]
		# Error rate
		ratio = -1

		err = 0
		for i in xrange(len(self.x_test)):
			# P(y = 1) = \sigmoid ( W ^ T * X)
			# P(y = 0) = 1 - P(y = 1)
			y_1 = sigmoid(self.weight.dot(np.insert(self.x_test[i], 0, 1)))
			y_0 = 1-y_1
			# J(\theta) = - \sum{ y_i * \log(h_theta{x_i}) + (1-y_i) * \log(1-h_theta{x_i}) }
			err -= self.y_test[i]*math.log(y_1+eps) + (1-self.y_test[i])*math.log(y_0+eps)
			# Count Error
			if (y_0 > y_1): y = 0
			else: y = 1
			if y == self.y_test[i]:
				self.count[1]  += 1
			else:
				self.count[0]  += 1
				self.ecount[y] += 1

		# Regularization = lambda / 2 * \sum{ W_j ^ 2 }
		if self.l2_on:
			for para in self.weight[1:]:
				err += self.L2norm/2.0*para*para

		print
		# print "-" * 10, "Logistic Regression Classifier", "-" * 18
		# print "Correct Classcification:", self.count[1], ", Wrong Classcification:", self.count[0]
		# print "Spam => Normal:", self.ecount[0], ", Normal => Spam:", self.ecount[1]
		ratio = 100 * self.count[0] / float(len(self.y_test))
		# print "Error Ratio: ", ratio, "%"
		# print "-" * 59
		# print "Loss:", err
		return (ratio, err)

class LinearRegression(DiscriminativeClassifier):

	def __init__(self, l2norm=1, preprocessing="z", eta=1e-4, max_epoch=30, l2_on=True, method="lsq", gamma = 0.5, miniBatch=32):
		"""
		:param l2norm: l2 norm penalty
		:param preprocessing: preprocessing method
		:param eta: learning rate (step size)
		:param max_epoch: how many epochs are you going to train the regression (optional)
		:param l2_on: use l2 or not
		"""
		super(LinearRegression, self).__init__(preprocessing=preprocessing, bias=True)
		self.weight = np.ones(self.feature_size + 1)
		self.L2norm = l2norm
		self.eta = eta
		self.max_epoch = max_epoch
		self.l2_on = l2_on
		self.method = method
		self.gamma = gamma
		self.miniBatch = miniBatch

	def train(self, eps=1e-4):
		if (self.method == "lsq"):
			if self.l2_on:
				self.weight, err = np.linalg.lstsq(np.r_[np.c_[np.ones(len(self.x_train)), self.x_train],\
					np.mat(np.insert(np.ones(self.feature_size) * np.sqrt(self.L2norm), 0, 0))],\
					np.append(self.y_train, 0))[:2]
			else:
				self.weight, err = np.linalg.lstsq(np.c_[np.ones(len(self.x_train)), self.x_train], self.y_train)[:2]
			if err != []:
				err = np.asscalar(err) / len(self.x_train)
			else:
				err = 0
			self.count = [0, 0]
			for i in xrange(len(self.x_train)):
				y_1 = self.weight.dot(np.insert(self.x_train[i], 0, 1))
				y_0 = 1 - y_1
				if (y_0 > y_1): y = 0
				else: y = 1
				if y == self.y_train[i]: self.count[1]  += 1
				else: self.count[0]  += 1
			ratio = 100 * self.count[0] / float(len(self.y_train))
			if self.l2_on:
				for para in self.weight[1:]:
					err += self.L2norm/2.0*para*para
		else:

			epoch = 0

			Recall_training    = []
			Precision_training = []
			Recall_test        = []
			Precision_test     = []

			while True:
				momentum = 0
				for i in xrange(0, len(self.x_train), self.miniBatch):
					length = min(self.miniBatch, len(self.x_train) - i)
					update = np.zeros_like(self.weight)
					for j in xrange(length):
						update += np.insert(self.x_train[i+j], 0, 1) \
									* ( self.weight.dot(np.insert(self.x_train[i+j], 0, 1))\
										- self.y_train[i+j] )
					momentum = momentum * self.gamma + self.eta * update
					if self.l2_on:
						momentum += self.eta * self.L2norm * np.insert(self.weight[1:], 0, 0);

					self.weight -= momentum

				recall = 0
				precision = 0
				for i in xrange(len(self.x_train)):
					y_1 = self.weight.dot(np.insert(self.x_train[i], 0, 1))
					y_0 = 1-y_1
					if (y_0 > y_1): y = 0
					else: y = 1

					if (y == 1 and self.y_train[i] == 1):
						recall += 1
					if (y == self.y_train[i]):
						precision += 1
				precision /= float(len(self.y_train))
				recall /= float(sum(self.y_train))
				Recall_training.append(recall)
				Precision_training.append(precision)

				recall = 0
				precision = 0
				for i in xrange(len(self.x_test)):
					y_1 = self.weight.dot(np.insert(self.x_test[i], 0, 1))
					y_0 = 1-y_1
					if (y_0 > y_1): y = 0
					else: y = 1

					if (y == 1 and self.y_test[i] == 1):
						recall += 1
					if (y == self.y_test[i]):
						precision += 1
				precision /= float(len(self.y_test))
				recall /= float(sum(self.y_test))
				Recall_test.append(recall)
				Precision_test.append(precision)

				epoch += 1
				if epoch == self.max_epoch:
					return (Precision_training, Recall_training, Precision_test, Recall_test)

		return (ratio, err)

	def test(self, eps=1e-4):
		# Count Right & Wrong Number
		self.count  = [0, 0]
		# Count Error Type & Number
		self.ecount = [0, 0]
		# Error rate
		ratio = -1
		if self.l2_on:
			err = norm(np.mat(self.weight) * np.r_[np.c_[np.ones(len(self.x_test)), self.x_test],\
					np.mat(np.insert(np.ones(self.feature_size) * np.sqrt(self.L2norm), 0, 0))].T -\
					np.append(self.y_test, 0)) / len(self.x_test)
		else:
			err = norm(np.mat(self.weight) * np.c_[np.ones(len(self.x_test)), self.x_test].T - self.y_test) / len(self.x_test)
		for i in xrange(len(self.x_test)):
			y_1 = self.weight.dot(np.insert(self.x_test[i], 0, 1))
			y_0 = 1 - y_1
			# Count Error
			if (y_0 > y_1): y = 0
			else: y = 1
			if y == self.y_test[i]:
				self.count[1]  += 1
			else:
				self.count[0]  += 1
				self.ecount[y] += 1

		# Regularization = lambda / 2 * \sum{ W_j ^ 2 }
		if self.l2_on:
			for para in self.weight[1:]:
				err += self.L2norm/2.0*para*para

		ratio = 100 * self.count[0] / float(len(self.y_test))
		return (ratio, err)


class KNNClassifier(DiscriminativeClassifier):

	def __init__(self, preprocessing="", K=4):
		"""

		:param preprocessing: preprocessing method
		:param K: how much neighbours you want
		"""
		super(KNNClassifier, self).__init__(preprocessing=preprocessing)
		self.binary = (preprocessing == "binary")
		self.K = K

	# TODO: implement distance between sample here
	def __calc_distance(self, a, b):
		if self.binary:
			return np.sum(abs(a-b))
		else:
			return norm(a-b)

	def train(self):
		self.count = [0, 0]
		for i in xrange(len(self.x_train)):
			temp = sorted(range(len(self.x_train)), key = lambda x:self.__calc_distance(self.x_train[i], self.x_train[x]))[:self.K]
			y_1 = sum([self.y_train[w] for w in temp])
			y_0 = self.K - y_1
			if (y_0 > y_1): y = 0
			else: y = 1
			if y == self.y_train[i]: self.count[1]  += 1
			else: self.count[0]  += 1
		ratio = 100 * self.count[0] / float(len(self.y_train))
		return ratio

	def test(self):
		self.count = [0, 0]
		for i in xrange(len(self.x_test)):
			temp = sorted(range(len(self.x_train)), key = lambda x:self.__calc_distance(self.x_test[i], self.x_train[x]))[:self.K]
			y_1 = sum([self.y_test[w] for w in temp])
			y_0 = self.K - y_1
			if (y_0 > y_1): y = 0
			else: y = 1
			if y == self.y_test[i]: self.count[1]  += 1
			else: self.count[0]  += 1
		ratio = 100 * self.count[0] / float(len(self.y_test))
		return ratio

if __name__ == '__main__':
	# logistic = LogisticRegression()
	# logistic.train()
	# logistic.test()
	# linear = LinearRegression(method = "sgd", miniBatch = 256)
	# file = open("../output/sgd.out", "w")
	# print >> file, linear.train()
	knn = KNNClassifier()
	print knn.train()
	print knn.test()