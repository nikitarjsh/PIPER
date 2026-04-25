# Decision Tree implementation based on the CART algorithm

import numpy as np
import pandas as pd

from treenode import TreeNode


# Compute Gini impurity
def gini(y):
    if len(y) == 0:
        return 0.0

    counts = np.bincount(y.astype(int))
    probs = counts / len(y)

    return 1.0 - np.sum(probs ** 2)


# Compute Gini-based information gain
def information_gain(y_parent, y_left, y_right):
    n = len(y_parent)

    weighted_child = (
        (len(y_left) / n) * gini(y_left)
        + (len(y_right) / n) * gini(y_right)
    )

    return gini(y_parent) - weighted_child


# Find the best feature and threshold
def best_split(X, y):
    best_gain = -1
    best_feat = None
    best_thresh = None

    for feat in range(X.shape[1]):
        thresholds = np.unique(X[:, feat])

        for thresh in thresholds:
            left_mask = X[:, feat] <= thresh
            right_mask = ~left_mask

            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            gain = information_gain(y, y[left_mask], y[right_mask])

            if gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_thresh = thresh

    return best_feat, best_thresh, best_gain


# Create a leaf node
def make_leaf(y, n_samples):
    unique, counts = np.unique(y, return_counts=True)

    label_counts = dict(zip(unique.astype(int).tolist(), counts.tolist()))
    probs = {k: v / n_samples for k, v in label_counts.items()}

    return TreeNode(
        n_samples=n_samples,
        prediction_probs=probs,
        label_counts=label_counts
    )


# Recursively build the CART tree
def build_tree(X, y, depth=0, max_depth=10, min_samples_split=2, min_samples_leaf=1):
    n_samples = len(y)

    if (
        depth >= max_depth
        or n_samples < min_samples_split
        or len(set(y)) == 1
    ):
        return make_leaf(y, n_samples)

    feat, thresh, gain = best_split(X, y)

    if feat is None or gain <= 0:
        return make_leaf(y, n_samples)

    left_mask = X[:, feat] <= thresh
    right_mask = ~left_mask

    if left_mask.sum() < min_samples_leaf or right_mask.sum() < min_samples_leaf:
        return make_leaf(y, n_samples)

    node = TreeNode(
        feature_idx=feat,
        feature_val=thresh,
        information_gain=gain,
        n_samples=n_samples
    )

    node.left = build_tree(
        X[left_mask],
        y[left_mask],
        depth + 1,
        max_depth,
        min_samples_split,
        min_samples_leaf
    )

    node.right = build_tree(
        X[right_mask],
        y[right_mask],
        depth + 1,
        max_depth,
        min_samples_split,
        min_samples_leaf
    )

    return node


# Predict one sample
def predict_one(node, x):
    if node.is_leaf:
        return max(node.prediction_probs, key=node.prediction_probs.get)

    if x[node.feature_idx] <= node.feature_val:
        return predict_one(node.left, x)

    return predict_one(node.right, x)


# Predict many samples
def predict(root, X):
    return np.array([predict_one(root, x) for x in X])


# Calculate feature importance using information gain
def get_feature_importances(node, n_features):
    importances = np.zeros(n_features)

    def walk(n):
        if n is None or n.is_leaf:
            return

        importances[n.feature_idx] += n.information_gain

        walk(n.left)
        walk(n.right)

    walk(node)

    total = importances.sum()

    if total > 0:
        return importances / total

    return importances


class CARTClassifier:
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.feature_names_ = None

    def fit(self, X, y, feature_names=None):
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = np.asarray(y).astype(int)

        self.root = build_tree(
            X,
            y,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf
        )

        self.feature_names_ = feature_names

        return self

    def predict(self, X):
        X = X.values if isinstance(X, pd.DataFrame) else X

        return predict(self.root, X)

    def predict_proba(self, X):
        X = X.values if isinstance(X, pd.DataFrame) else X

        probs = []

        for x in X:
            node = self.root

            while not node.is_leaf:
                if x[node.feature_idx] <= node.feature_val:
                    node = node.left
                else:
                    node = node.right

            prob_0 = node.prediction_probs.get(0, 0)
            prob_1 = node.prediction_probs.get(1, 0)

            probs.append([prob_0, prob_1])

        return np.array(probs)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def feature_importances(self):
        n_features = len(self.feature_names_ or [])

        imps = get_feature_importances(self.root, n_features)

        if self.feature_names_:
            return pd.Series(imps, index=self.feature_names_).sort_values(ascending=False)

        return imps