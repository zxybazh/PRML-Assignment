from dataloader import *
import numpy as np
from numpy.linalg import pinv, norm


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

	def __init__(self, l2norm=1, preprocessing="z", eta=1e-4, max_epoch=30, l2_on=False):
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
			y_1 = self.weight.dot(np.insert(self.x_test[i], 0, 1))
			y_0 = 1-y_1
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
			return
		else:
			return

	def train(self):
		# TODO
		return

	def test(self):
		# TODO
		err = 0
		return err

if __name__ == '__main__':
	logistic = LogisticRegression();
	logistic.train()
	logistic.test()