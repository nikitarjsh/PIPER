import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_pca(X):
    X = X.to_numpy(dtype=float)

    # center
    X = X - np.mean(X, axis=0)

    # scale
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    X = X / std

    # covariance
    cov = np.cov(X, rowvar=False)

    # eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # sort from largest to smallest
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # project to first 2 PCs
    X_pca = X @ eigenvectors[:, :2]

    var_ratio = eigenvalues / eigenvalues.sum()

    return X_pca, var_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input csv file")
    parser.add_argument("--output", required=True, help="output figure path")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    label_col = "Label"
    drop_cols = ["peptide", "HLA", "hla_sequence", label_col]

    y = df[label_col]
    X = df.drop(columns=drop_cols, errors="ignore")

    # keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    X_pca, var_ratio = run_pca(X)

    plt.figure(figsize=(8, 6))

    classes = sorted(y.unique())
    for c in classes:
        mask = y == c
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            label=f"Class {c}",
            alpha=0.7,
            s=20
        )

    plt.xlabel(f"PC1 ({var_ratio[0] * 100:.2f}%)")
    plt.ylabel(f"PC2 ({var_ratio[1] * 100:.2f}%)")
    plt.title("PCA of dataset")
    plt.legend()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.close()

    print("saved to:", args.output)


if __name__ == "__main__":
    main()