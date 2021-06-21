import numpy as np


def mse_score(y_true, y_pred):
	return np.mean(np.square(y_true - y_pred))

def mae_score(y_true, y_pred):
	return np.mean(np.abs(y_true - y_pred))

def poisson_score(y_true, y_pred):
	return np.mean(y_true * np.log(y_true / y_pred) - y_true + y_pred)


class DecisionTreeRegressorNode:
	def __init__(self, score, num_samples, prediction):
		self.score = score
		self.num_samples = num_samples
		self.prediction = prediction
		self.feature_idx = 0
		self.threshold = 0
		self.left = None
		self.right = None


class DecisionTreeRegressor:
	def __init__(self, criterion='mse', max_depth=None):
		self.criterion = criterion.lower()
		self.depth = 0
		self.n_features = 0
		self.n_classes = 0
		self.n_leaves = 0
		self.tree = None
		self.max_depth = max_depth
		
		if self.criterion == 'mse':
			self.prediction_fn = lambda x: np.mean(x)
			self.score_fn = mse_score
		elif self.criterion == 'mae':
			self.prediction_fn = lambda x: np.median(x)
			self.score_fn = mae_score
		elif self.criterion == 'poisson':
			self.prediction_fn = lambda x: np.mean(x)
			self.score_fn = poisson_score
		else:
			raise ValueError(f'invalid criterion: {criterion}')
	
	def fit(self, X, y):
		X = np.array(X)
		y = np.array(y)
		n, k = X.shape
		
		self.n_classes = len(np.unique(y))
		self.n_features = k
		self.tree = self._build_tree(X, y)
		
		return self.evaluate(X, y)
	
	def _build_tree(self, X, y, depth=0):
		n = len(y)
		if n == 0:
			return None
		
		# the prediction is the mean/median of the data
		prediction = self.prediction_fn(y)
		# score = error(ground truth, prediction)
		score = self.score_fn(y)
		
		node = DecisionTreeRegressorNode(
			score=score,
			num_samples=n,
			prediction=prediction
		)
		
		# keep splitting recursively until max depth
		if not self.max_depth or depth < self.max_depth:
			self.depth = max(depth, self.depth)
			
			# find optimal split for node
			feature_idx, threshold = self._split(X, y)
			
			if feature_idx is not None:
				# calculate left and right indices
				idx_left = X[:, feature_idx] < threshold
				idx_right = ~idx_left
				# split to left and right nodes
				X_left, y_left = X[idx_left], y[idx_left]
				X_right, y_right = X[idx_right], y[idx_right]
				
				# update node values
				node.feature_idx = feature_idx
				node.threshold = threshold
				node.left = self._build_tree(X_left, y_left, depth + 1)
				node.right = self._build_tree(X_right, y_right, depth + 1)
			else:
				# leaf node
				self.n_leaves += 1
		else:
			# leaf node
			self.n_leaves += 1
		
		return node
	
	def _split(self, X, y):
		n = len(y)
		if n <= 1:
			return None, None
		
		# the prediction is the mean/median of the data
		prediction = self.prediction_fn(y)
		# score = error(ground truth, prediction)
		best_score = self.score_fn(y, prediction)
		best_feature_idx, best_threshold = None, None
		
		# loop through all features
		for feature_idx in range(self.n_features):
			# get current feature vector of all entries
			feature = X[:, feature_idx]
			# sorted feature vector is the thresholds
			thresholds = np.sort(feature)
			
			# loop through all possible split thresholds
			for threshold in thresholds:
				indices = feature < threshold
				y_left = y[indices]
				y_right = y[~indices]
				
				prediction_left = self.prediction_fn(y_left)
				prediction_right = self.prediction_fn(y_right)
				
				# calculate score sum of left and right node scores
				score_left = self.score_fn(y_left, prediction_left)
				score_right = self.score_fn(y_right, prediction_right)
				score = score_left + score_right
				
				if score < best_score:
					best_score = score
					best_feature_idx = feature_idx
					best_threshold = threshold
					
		return best_feature_idx, best_threshold
	
	def predict(self, X):
		return np.array([self._predict_one(x) for x in X])
	
	def _predict_one(self, x):
		return self._predict_node(x).prediction
	
	def _predict_node(self, x):
		node = self.tree
		while node.left:
			if x[node.feature_idx] < node.threshold:
				node = node.left
			else:
				node = node.right
		return node
	
	def evaluate(self, X, y):
		y = np.array(y)
		y_pred = self.predict(X)
		
		sse = np.sum(np.square(y - y_pred))
		s_yy = np.sum(np.square(y - np.mean(y)))
		
		# average score across all predictions
		score = np.mean([self._predict_node(x).score] for x in X)
		# r2 score
		r2 = 1 - sse / s_yy
		
		return score, r2
