import numpy as np


class DecisionTreeNode:
	def __init__(self, predicted_class):
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
	
	def fit(self, X, y):
		X = np.array(X)
		y = np.array(y)
		n, k = X.shape
		
		self.n_classes = len(np.unique(y))
		self.n_features = k
		self.tree = self._build_tree(X, y)
	
	def _build_tree(self, X, y, depth=0):
		if y.size == 0:
			return None
		
		num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
		predicted_class = np.argmax(num_samples_per_class)
		node = DecisionTreeNode(predicted_class)
		
		self.depth = max(depth, self.depth)
		
		if not self.max_depth or depth < self.max_depth:
			if self.criterion == 'gini':
				idx, threshold = self._gini_split(X, y)
			elif self.criterion == 'entropy':
				idx, threshold = self._entropy_split(X, y)
			else:
				raise ValueError(f'invalid criterion: {self.criterion}')
			
			if idx is not None:
				idx_left = X[:, idx] < threshold
				X_left, y_left = X[idx_left], y[idx_left]
				X_right, y_right = X[~idx_left], y[~idx_left]
				node.feature_index = idx
				node.threshold = threshold
				node.left = self._build_tree(X_left, y_left, depth + 1)
				node.right = self._build_tree(X_right, y_right, depth + 1)
		return node
	
	def _gini_split(self, X, y):
		m = y.size
		if m <= 1:
			return None, None
		
		num_parent = [np.sum(y == c) for c in range(self.n_classes)]
		best_gini = 1 - sum((n / m) ** 2 for n in num_parent)
		best_idx, best_threshold = None, None
		
		for idx in range(self.n_classes):
			thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
			num_left = np.zeros(self.n_classes)
			num_right = np.copy(num_parent)
			
			for i in range(1, m):
				c = classes[i - 1]
				num_left[c] += 1
				num_right[c] -= 1
				
				gini_left = 1 - sum(num_left[x] for x in range(self.n_classes)) / i
				gini_right = 1 - sum(num_right[x] for x in range(self.n_classes)) / i
				gini = (i * gini_left + (m - i) * gini_right) / m
				
				if thresholds[i] == thresholds[i - 1]:
					continue
				
				if gini < best_gini:
					best_gini = gini
					best_idx = idx
					best_threshold = (thresholds[i] + thresholds[i - 1]) / 2
					
		return best_idx, best_threshold
	
	def _entropy_split(self, X, y):
		m = y.size
		if m <= 1:
			return None, None
		
		num_parent = [np.sum(y == c) for c in range(self.n_classes)]
		best_entropy = -sum((n / m) * np.log2(n / m) for n in num_parent)
		best_idx, best_threshold = None, None
		
		for idx in range(self.n_classes):
			thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
			num_left = np.zeros(self.n_classes)
			num_right = np.copy(num_parent)
			
			for i in range(1, m):
				c = classes[i - 1]
				num_left[c] += 1
				num_right[c] -= 1
				
				entropy_left = -sum(num_left[x] for x in range(self.n_classes)) / i
				entropy_right = -sum(num_right[x] for x in range(self.n_classes)) / i
				entropy = (i * entropy_left + (m - i) * entropy_right) / m
				
				if thresholds[i] == thresholds[i - 1]:
					continue
				
				if entropy < best_entropy:
					best_gini = entropy
					best_idx = idx
					best_threshold = (thresholds[i] + thresholds[i - 1]) / 2
					
		return best_idx, best_threshold
	
	def predict(self, X):
		return np.array([self._predict_one(x) for x in X])
	
	def _predict_one(self, x):
		node = self.tree
		while node.left:
			if x[node.feature_index] < node.threshold:
				node = node.left
			else:
				node = node.right
		return node.predicted_class
	
	def evaluate(self, X, y):
		y = np.array(y)
		y_pred = self.predict(X)
		
		# categorical accuracy
		accs = np.mean(y == y_pred)
		
		return accs