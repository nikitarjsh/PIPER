import numpy as np
import pandas as pd

from CartClassifier import CARTClassifier


class RandomForestFromScratch:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X.iloc[indices].values, y[indices]

    def fit(self, X, y):
        self.trees = []

        for i in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)

            tree = CARTClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )

            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

        return self

    def predict_proba(self, X):
        X_vals = X.values if isinstance(X, pd.DataFrame) else X
        all_tree_preds = np.array([tree.predict(X_vals) for tree in self.trees])

        prob_pos = np.mean(all_tree_preds, axis=0)

        return np.vstack([1 - prob_pos, prob_pos]).T

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)