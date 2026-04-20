#Decision Tree implementation is based on the CART Algorithm
#Compute Gini impurity, which measures how mixed the labels are (0 = pure, higher = more mixed).

import pandas as pd
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import GroupShuffleSplit
from Bio.Align import substitution_matrices
from collections import Counter
from treenode import TreeNode
from sklearn.model_selection import train_test_split

def gini(y):
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y)
    probs  = counts / len(y)
    return 1.0 - np.sum(probs ** 2)

#Compute Gini-based information gain, which measures how much a split improves purity. 
def information_gain(y_parent, y_left, y_right):
    n = len(y_parent)
    weighted_child = (len(y_left) / n)  * gini(y_left) + \
                     (len(y_right) / n) * gini(y_right)
    return gini(y_parent) - weighted_child

# Find the best feature and threshold to split on to maximize information gain 
def best_split(X, y, max_features=None):
    best_gain = -1
    best_feat, best_thresh = None, None

    n_features = X.shape[1]
    if max_features is not None:
        feat_indices = np.random.choice(n_features, max_features, replace=False)
    else:
        feat_indices = range(n_features)

    for feat in feat_indices:  # ← was: range(X.shape[1])
        thresholds = np.unique(X[:, feat])
        for thresh in thresholds:
            left_mask  = X[:, feat] <= thresh
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue
            gain = information_gain(y, y[left_mask], y[right_mask])
            if gain > best_gain:
                best_gain  = gain
                best_feat  = feat
                best_thresh = thresh

    return best_feat, best_thresh, best_gain

#Create a leaf node containing Create a leaf node containing class counts and class probabilities
def make_leaf(y, n_samples):
    unique, counts = np.unique(y, return_counts=True)
    label_counts = dict(zip(unique.tolist(), counts.tolist()))
    probs = {k: v / n_samples for k, v in label_counts.items()}

    return TreeNode(
        n_samples=n_samples,
        prediction_probs=probs,
        label_counts=label_counts
    )


#Recursively build the decision tree using CART algorithm. Stops when: max depth is reached, node is pure, or not enough samples
def build_tree(X, y, depth=0, max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features=None):
    n_samples = len(y)
    if (depth >= max_depth or n_samples < min_samples_split or len(set(y)) == 1):
        return make_leaf(y, n_samples)

    feat, thresh, gain = best_split(X, y, max_features=max_features)  # ← pass it here

    if feat is None or gain <= 0:
        return make_leaf(y, n_samples)

    left_mask  = X[:, feat] <= thresh
    right_mask = ~left_mask

    if left_mask.sum() < min_samples_leaf or right_mask.sum() < min_samples_leaf:
        return make_leaf(y, n_samples)

    node = TreeNode(feature_idx=feat, feature_val=thresh, information_gain=gain, n_samples=n_samples)
    node.left  = build_tree(X[left_mask],  y[left_mask],  depth+1, max_depth, min_samples_split, min_samples_leaf, max_features)
    node.right = build_tree(X[right_mask], y[right_mask], depth+1, max_depth, min_samples_split, min_samples_leaf, max_features)
    return node

#Predict label for a single sample by traversing the tree
def predict_one(node, x):
    if node.is_leaf:
        return max(node.prediction_probs, key=node.prediction_probs.get)

    if x[node.feature_idx] <= node.feature_val:
        return predict_one(node.left, x)
    else:
        return predict_one(node.right, x)
    
def predict_proba_one(node, x, n_classes=2):
    if node.is_leaf:
        probs = [node.prediction_probs.get(c, 0.0) for c in range(n_classes)]
        return probs
    if x[node.feature_idx] <= node.feature_val:
        return predict_proba_one(node.left, x, n_classes)
    else:
        return predict_proba_one(node.right, x, n_classes)

#Predict labels for multiple samples
def predict(root, X):
    return np.array([predict_one(root, x) for x in X])


#Aggregate feature importance based on information gain
def get_feature_importances(node, n_features):

    importances = np.zeros(n_features)

    def walk(n):
        if n is None or n.is_leaf:
            return

        importances[n.feature_idx] += n.feature_importance

        walk(n.left)
        walk(n.right)

    walk(node)

    total = importances.sum()
    return importances / total if total > 0 else importances

#CART decision tree classifier implementation
class CARTClassifier:
    #Store hyperparameters
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features=None):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.max_features      = max_features
        self.root              = None
        self.feature_names_    = None

    #Train the decision tree
    def fit(self, X, y, feature_names=None):
        self.root = build_tree(X, y,
                           max_depth=self.max_depth,
                           min_samples_split=self.min_samples_split,
                           min_samples_leaf=self.min_samples_leaf,
                           max_features=self.max_features)  # ← pass it through
        self.feature_names_ = feature_names
        return self

    #Predict labels
    def predict(self, X):
        return predict(self.root, X)
    
    def predict_proba(self, X):
        return np.array([predict_proba_one(self.root, x) for x in X])

    #Estimate accuracy (preliminary)
    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    #Return sorted feature importance values
    def feature_importances(self):
        imps = get_feature_importances(self.root, len(self.feature_names_ or []))
        if self.feature_names_:
            return pd.Series(imps, index=self.feature_names_).sort_values(ascending=False)
        return imps
    