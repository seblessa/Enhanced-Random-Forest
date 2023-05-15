import random

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.left = None
        self.right = None
        self.feature_idx = None
        self.threshold = None
        self.leaf_value = None

    def fit(self, X, y):
        if self.random_state is not None:
            random.seed(self.random_state)

        if self.max_depth is None or self.max_depth > 0:
            best_feature, best_threshold = self._choose_split(X, y)
            if best_feature is not None:
                self.feature_idx = best_feature
                self.threshold = best_threshold
                left_idx, right_idx = self._split(X[:, best_feature], best_threshold)
                self.left = DecisionTree(max_depth=None if self.max_depth is None else self.max_depth - 1,
                                         min_samples_split=self.min_samples_split, random_state=self.random_state)
                self.right = DecisionTree(max_depth=None if self.max_depth is None else self.max_depth - 1,
                                          min_samples_split=self.min_samples_split, random_state=self.random_state)
                self.left.fit(X[left_idx, :], y[left_idx])
                self.right.fit(X[right_idx, :], y[right_idx])
            else:
                self.leaf_value = self._calculate_leaf_value(y)
                self.left = None
                self.right = None
        else:
            self.leaf_value = self._calculate_leaf_value(y)
            self.left = None
            self.right = None

    def predict(self, X):
        if self.leaf_value is not None:
            return self.leaf_value

        if X[self.feature_idx] <= self.threshold:
            return self.left.predict(X)
        else:
            return self.right.predict(X)

    def _choose_split(self, X, y):
        best_feature, best_threshold = None, None
        best_score = float('-inf')

        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None

        for feature_idx in range(n_features):
            thresholds = list(set(X[:, feature_idx]))
            for threshold in thresholds:
                left_idx, right_idx = self._split(X[:, feature_idx], threshold)
                if len(left_idx) < self.min_samples_split or len(right_idx) < self.min_samples_split:
                    continue
                score = self._gini_index(y[left_idx]) * len(left_idx) + self._gini_index(y[right_idx]) * len(right_idx)
                if score > best_score:
                    best_score = score
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _split(self, feature, threshold):
        left_idx = [i for i in range(len(feature)) if feature[i] <= threshold]
        right_idx = [i for i in range(len(feature)) if feature[i] > threshold]
        return left_idx, right_idx

    def _gini_index(self, y):
        n_samples = len(y)
        _, counts = zip(*list(set(zip(y, [1] * n_samples))))
        gini = 1.0
        for count in counts:
            p = count / n_samples
        gini -= p ** 2
        return gini

    def _calculate_leaf_value(self, y):
        _, counts = zip(*list(set(zip(y, [1] * len(y)))))
        return max(counts) / sum(counts)


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []
        for _ in range(n_estimators):
            self.trees.append(DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split,
                                           random_state=random_state))

    def fit(self, X, y):
        for tree in self.trees:
            sample_idx = [random.randint(0, len(X) - 1) for _ in range(len(X))]
            tree.fit(X[sample_idx, :], y[sample_idx])

    def predict(self, X):
        predictions = [tree.predict(X) for tree in self.trees]
        return max(set(predictions), key=predictions.count)
