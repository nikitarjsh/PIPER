import argparse
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


ROOT_DIR = os.path.dirname(__file__)
SRC_PATH = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_PATH)

import __main__
from RandomForestFromScratch import RandomForestFromScratch

__main__.RandomForestFromScratch = RandomForestFromScratch

from utils import (
    standardize_hla_alleles,
    map_alleles,
    build_peptidy_feature_df,
    feature_processing,
    soft_voting_ensemble,
)


DROP_COLS = ["index", "peptide", "HLA", "hla_sequence"]

MODEL_DIR = os.path.join(ROOT_DIR, "models", "best_models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")


def load_input_data(args):
    """
    Load input data from a CSV file or from one peptide-HLA pair.
    """

    if args.input_csv is not None:
        df = pd.read_csv(args.input_csv)

        if "peptide" not in df.columns or "HLA" not in df.columns:
            raise ValueError("Input CSV must contain 'peptide' and 'HLA' columns.")

        df = df[["peptide", "HLA"]].copy()

    else:
        if args.peptide is None or args.hla is None:
            raise ValueError("Please provide either --input_csv or both --peptide and --hla.")

        df = pd.DataFrame({
            "peptide": [args.peptide],
            "HLA": [args.hla]
        })

    return df


def preprocess_input(df):
    """
    Apply the same preprocessing used during model training.
    """

    df = df.copy()

    # standardize HLA names and map them to pseudosequences
    df = standardize_hla_alleles(df)
    df = map_alleles(df)

    # remove peptides with invalid amino acids or lowercase letters
    valid_mask = df["peptide"].apply(
        lambda x: isinstance(x, str) and ("X" not in x) and x.isupper()
    )

    df = df[valid_mask].reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid peptides left after preprocessing.")

    # check if any HLA could not be mapped
    if df["hla_sequence"].isna().any():
        missing_hlas = df.loc[df["hla_sequence"].isna(), "HLA"].unique()
        raise ValueError(f"These HLA alleles could not be mapped: {missing_hlas}")

    # extract peptide position-specific features
    peptide_features = build_peptidy_feature_df(
        df,
        peptide_col="peptide",
        padding_len=10
    )

    # extract global HLA pseudosequence features
    hla_features = feature_processing(
        df["hla_sequence"].tolist(),
        seq_type="HLA_"
    )

    # combine original input with extracted features
    final_df = pd.concat(
        [
            df.reset_index(drop=True),
            peptide_features.reset_index(drop=True),
            hla_features.reset_index(drop=True)
        ],
        axis=1
    )

    return final_df


def make_model_input(processed_df):
    """
    Remove non-feature columns and prepare the final model input.
    """

    X = processed_df.drop(
        columns=[c for c in DROP_COLS if c in processed_df.columns]
    )

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols)

    return X


def load_models():
    """
    Load all best models except the decision tree model.
    """

    models = []

    for file in os.listdir(MODEL_DIR):
        if not file.endswith(".pkl"):
            continue

        name = file.replace(".pkl", "")

        # decision tree is excluded from the final consensus
        if "decision_tree" in name:
            continue

        path = os.path.join(MODEL_DIR, file)
        model = joblib.load(path)

        models.append((name, model))

    if len(models) == 0:
        raise ValueError("No models were loaded. Check models/best_models/.")

    return models


def align_features_to_model(X, models):
    """
    Match the input columns to the columns used during training.
    """

    first_model = models[0][1]

    if hasattr(first_model, "feature_names_in_"):
        expected_cols = list(first_model.feature_names_in_)
        X = X.reindex(columns=expected_cols, fill_value=0)

    return X


def label_prediction(y):
    """
    Convert 0/1 model output into a readable label.
    """

    if int(y) == 1:
        return "immunogenic"

    return "non-immunogenic"


def predict_all_models(models, X):
    """
    Run each model separately.
    """

    output = {}

    for name, model in models:
        y_pred = model.predict(X)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 1]
        else:
            y_prob = np.nan

        output[f"{name}_probability"] = y_prob
        output[f"{name}_prediction"] = [label_prediction(y) for y in y_pred]

    return output

def run_evaluation_plots():

    test_df = pd.read_csv("data/dataset_test.csv")

    DROP_COLS = ["index", "peptide", "HLA", "hla_sequence"]
    TARGET_COL = "Label"

    y_test = test_df[TARGET_COL].values
    X_test = test_df.drop(columns=[TARGET_COL] + [c for c in DROP_COLS if c in test_df.columns])

    cat_cols = X_test.select_dtypes(include=["object", "category"]).columns.tolist()
    X_test = pd.get_dummies(X_test, columns=cat_cols)

    models = load_models()

    plt.figure(figsize=(7,6))

    for name, model in models:
        if not hasattr(model, "predict_proba"):
            continue

        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    plt.plot([0,1],[0,1],"--",color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curves - Test Set")
    plt.legend()

    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved ROC curve to figures/")

def save_predictions(processed_df, model_outputs, consensus_pred, consensus_prob):
    """
    Save the final prediction table.
    """

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results_df = processed_df[["peptide", "HLA"]].copy()

    for col_name, values in model_outputs.items():
        results_df[col_name] = values

    results_df["consensus_probability"] = consensus_prob
    results_df["consensus_prediction"] = [
        label_prediction(y) for y in consensus_pred
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(RESULTS_DIR, f"piper_predictions_{timestamp}.csv")

    results_df.to_csv(output_path, index=False)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="PIPER: Protein Immunogenicity PredictoR"
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        help="Path to input CSV with peptide and HLA columns."
    )

    parser.add_argument(
        "--peptide",
        type=str,
        help="Peptide sequence for one prediction."
    )

    parser.add_argument(
        "--hla",
        type=str,
        help="HLA allele for one prediction."
    )

    parser.add_argument(
    "--plot_results",
    action="store_true",
    help="Generate ROC / evaluation plots on test set"
   )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used to convert probability to immunogenic/non-immunogenic."
    )

    args = parser.parse_args()

    if args.plot_results:
        print("Generating evaluation plots...")
        run_evaluation_plots()
        return

    if args.input_csv is not None and (args.peptide is not None or args.hla is not None):
        raise ValueError("Use either --input_csv or --peptide with --hla, not both.")

    print("\nRunning PIPER")
    print("Protein Immunogenicity PredictoR")
    print("--------------------------------")

    print("Reading input data...")
    input_df = load_input_data(args)

    print("Predicting immunogenicity...")
    processed_df = preprocess_input(input_df)
    X = make_model_input(processed_df)

    print("Loading trained models...")
    models = load_models()
    X = align_features_to_model(X, models)

    print("Running individual models...")
    model_outputs = predict_all_models(models, X)

    print("Running consensus prediction...")
    consensus_pred, consensus_prob = soft_voting_ensemble(
        models,
        X,
        threshold=args.threshold
    )

    print("Saving results...")
    output_path = save_predictions(
        processed_df,
        model_outputs,
        consensus_pred,
        consensus_prob
    )

    print("--------------------------------")
    print(f"Results saved to: {output_path}\n")


if __name__ == "__main__":
    main()