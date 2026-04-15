import numpy as np

# TreeNode represents a single node in the decision tree.
# Each node stores the feature index and threshold used for splitting,
# the predicted class probabilities, label counts at that node, and  the information gain from the split.
# Feature importance is computed as n_samples * information_gain (weighted contribution of this split).
# Leaf nodes have no feature_idx, feature_val, or information_gain, only prediction_probs and label_counts.
# Left and right pointers link to child nodes, and are None for leaf nodes.
class TreeNode:
    def __init__(self, feature_idx=None, feature_val=None,
                 information_gain=None,
                 prediction_probs=None,
                 n_samples=None,
                 label_counts=None):

        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.information_gain = information_gain
        self.prediction_probs = prediction_probs
        self.n_samples = n_samples
        self.label_counts = label_counts
        self.feature_importance = (n_samples or 0) * (information_gain or 0)
        self.left = None
        self.right = None

    @property
    def is_leaf(self):
        return self.left is None and self.right is None

    def node_def(self) -> str:
        if not self.is_leaf:
            return (
                f"NODE | Gain={self.information_gain:.4f} | "
                f"Split: X[{self.feature_idx}] <= {self.feature_val:.4f}"
            )
        else:
            output = ", ".join(
                [f"{k}->{v}" for k, v in self.label_counts.items()]
            )
            return (
                f"LEAF | Counts = {output} | "
                f"Probs = {self.prediction_probs}"
            )
