import numpy as np


def gini(p):
	return 1 - np.sum(np.square(p))

def entropy(p):
	return -np.sum(p * np.log(p))


class DecisionTreeNode:
	def __init__(self, score, num_samples, num_samples_per_class, predicted_class):
		self.score = score
		self.num_samples = num_samples
		self.num_samples_per_class = num_samples_per_class
		self.predicted_class = predicted_class
		self.feature_index = 0
		self.threshold = 0
		self.left = None
		self.right = None


class DecisionTree:
	def __init__(self, criterion='gini', max_depth=None):
		self.criterion = criterion
		self.depth = 0
		self.n_features = 0
		self.n_classes = 0
		self.tree = None
		self.max_depth = max_depth
		
		if criterion == 'entropy':
			self.score_fn = entropy
		else:
			self.score_fn = gini
	
	def fit(self, X, y):
		X = np.array(X)
		y = np.array(y)
		n, k = X.shape
		
		self.n_classes = len(np.unique(y))
		self.n_features = k
		self.tree = self._build_tree(X, y)
		
		return self.evaluate(X, y)
	
	def _build_tree(self, X, y, depth=0):
		n = y.size
		if n == 0:
			return None
		
		# count the number of samples per class
		num_samples_per_class = np.bincount(y, minlength=self.n_classes)
		# the predicted class is the highest count
		predicted_class = np.argmax(num_samples_per_class)
		# p_k = count of class k / total
		score = self.score_fn(num_samples_per_class / n)
		
		node = DecisionTreeNode(
			score=score,
			num_samples=n,
			num_samples_per_class=num_samples_per_class,
			predicted_class=predicted_class
		)
		
		# keep splitting recursively until max depth
		if not self.max_depth or depth < self.max_depth:
			self.depth = max(depth, self.depth)
			
			# find optimal split for node
			idx, threshold = self._split(X, y)
			
			if idx is not None:
				# calculate left and right indices
				idx_left = X[:, idx] < threshold
				idx_right = ~idx_left
				# split to left and right nodes
				X_left, y_left = X[idx_left], y[idx_left]
				X_right, y_right = X[idx_right], y[idx_right]
				
				# update node values
				node.feature_index = idx
				node.threshold = threshold
				node.left = self._build_tree(X_left, y_left, depth + 1)
				node.right = self._build_tree(X_right, y_right, depth + 1)
		
		return node
	
	def _split(self, X, y):
		n = y.size
		if n <= 1:
			return None, None
		
		# count the number of samples per class
		num_samples_per_class = np.bincount(y, minlength=self.n_classes)
		# p_k = count of class k / total
		best_score = self.score_fn(num_samples_per_class / n)
		best_idx, best_threshold = None, None
		
		# loop through all features
		for idx in range(self.n_features):
			# get current feature vector of all entries
			X_j = X[:, idx]
			# sorted vector is the thresholds
			sorted_indices = np.argsort(X_j)
			thresholds, classes = X_j[sorted_indices], y[sorted_indices]
			
			# class counts for left and right
			num_left = np.zeros(self.n_classes)
			num_right = np.copy(num_samples_per_class)
			
			# loop through all possible split position
			for i in range(1, n):
				c = classes[i - 1]
				num_left[c] += 1
				num_right[c] -= 1
				
				# calculate score using weighted average of left and right nodes
				score_left = self.score_fn(num_left / i)
				score_right = self.score_fn(num_right / (n - i))
				score = (i * score_left + (n - i) * score_right) / n
				
				# ensure no split on identical values of feature
				if thresholds[i] == thresholds[i - 1]:
					continue
				
				if score < best_score:
					best_score = score
					best_idx = idx
					best_threshold = (thresholds[i] + thresholds[i - 1]) / 2
					
		return best_idx, best_threshold
	
	def predict(self, X):
		return np.array([self._predict_one(x) for x in X])
	
	def predict_proba(self, X):
		return np.array([self._predict_proba_one(x) for x in X])
	
	def predict_log_proba(self, X):
		return np.log(self.predict_proba(X))
	
	def _predict_one(self, x):
		return self._predict_node(x).predicted_class
	
	def _predict_proba_one(self, x):
		node = self._predict_node(x)
		return np.mean(node.num_samples_per_class == node.predicted_class)
	
	def _predict_node(self, x):
		node = self.tree
		while node.left:
			if x[node.feature_index] < node.threshold:
				node = node.left
			else:
				node = node.right
		return node
	
	def evaluate(self, X, y):
		y = np.array(y)
		y_pred = self.predict(X)
		
		# gini/entropy score
		score = np.array([self._predict_node(x).score] for x in X)
		# categorical accuracy
		acc = np.mean(y == y_pred)
		
		return score, acc
