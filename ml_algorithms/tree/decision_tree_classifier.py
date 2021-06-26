import numpy as np


def gini_score(p):
    return 1 - np.sum(np.square(p))


def entropy_score(p):
    return -np.sum(p * np.log(p))


def misclassification_score(p):
    return 1 - np.max(p)


class DecisionTreeClassifierNode:
    def __init__(self, score, num_samples, num_samples_per_class, predicted_class):
        self.score = score
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_idx = 0
        self.threshold = 0
        self.left = None
        self.right = None


class DecisionTreeClassifier:
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion.lower()
        self.depth = 0
        self.n_features = 0
        self.n_classes = 0
        self.n_leaves = 0
        self.tree = None
        self.max_depth = max_depth

        if self.criterion == 'gini':
            self.score_fn = gini_score
        elif self.criterion == 'entropy':
            self.score_fn = entropy_score
        elif self.criterion == 'misclassification':
            self.score_fn = misclassification_score
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

        # count the number of samples per class
        num_samples_per_class = np.bincount(y, minlength=self.n_classes)
        # the predicted class is the highest count
        predicted_class = np.argmax(num_samples_per_class)
        # p_k = count of class k / total
        score = self.score_fn(num_samples_per_class / n)

        node = DecisionTreeClassifierNode(
            score=score,
            num_samples=n,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
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

        # count the number of samples per class
        num_samples_per_class = np.bincount(y, minlength=self.n_classes)
        # p_k = count of class k / total
        best_score = self.score_fn(num_samples_per_class / n)
        best_feature_idx, best_threshold = None, None

        # loop through all features
        for feature_idx in range(self.n_features):
            # get current feature vector of all entries
            feature = X[:, feature_idx]
            # sorted vector is the thresholds
            sorted_indices = np.argsort(feature)
            thresholds, classes = feature[sorted_indices], y[sorted_indices]

            # class counts for left and right
            num_left = np.zeros(self.n_classes)
            num_right = np.copy(num_samples_per_class)

            # optimized loop through all possible split positions
            # linear search rather than quadratic through all classes and splits
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
                    best_feature_idx = feature_idx
                    best_threshold = (thresholds[i] + thresholds[i - 1]) / 2

        return best_feature_idx, best_threshold

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
        return node.num_samples_per_class[node.predicted_class] / node.num_samples

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

        # average score across all predictions
        score = np.mean([self._predict_node(x).score] for x in X)
        # categorical accuracy
        acc = np.mean(y == y_pred)

        return score, acc
