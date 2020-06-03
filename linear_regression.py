import numpy as np


class LinearRegression:
	def __init__(self, fit_intercept=True):
		self._coef = []
		self._intercept = 0
		self.fit_intercept = fit_intercept
	
	@property
	def coef(self):
		return self._coef
	
	@property
	def intercept(self):
		return self._intercept
	
	def fit(self, X, y):
		X = np.array(X)
		if len(X.shape) == 1:
			X = X.reshape(-1, 1)
		if self.fit_intercept:
			ones = np.ones((X.shape[0], 1))
			X = np.concatenate((ones, X), 1)
		
		X_t = X.T
		beta = np.dot(np.dot(np.linalg.inv(np.dot(X_t, X)), X_t), y)
		
		if self.fit_intercept:
			self._intercept = beta[0]
			self._coef = beta[1:]
		else:
			self._coef = beta
	
	def predict(self, X):
		return np.dot(X, self._coef) + self._intercept
	
	def score(self, X, y):
		y_pred = self.predict(X)
		sse = ((y - y_pred) ** 2).sum()
		s_yy = ((y - y.mean()) ** 2).sum()
		return 1 - sse / s_yy
