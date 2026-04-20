import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    matthews_corrcoef,
    classification_report
)


def evaluate_model(model, X_test, y_test, X, feature_names=None, model_name=None):
    """
    Evaluates a binary classifier and plots:
    - Accuracy, ROC-AUC, MCC
    - Classification report
    - Confusion matrix
    - Top 20 feature importances
    - Staged ROC-AUC curve (AdaBoost and gradient boosting models only)

    Compatible with: DecisionTreeClassifier, RandomForestClassifier,
                     AdaBoostClassifier, XGBClassifier, and any sklearn-style model.

    Args:
        model:          trained classifier
        X_test:         test features
        y_test:         test labels
        X:              full feature matrix (used for feature names)
        feature_names:  list of feature names (optional, inferred from X if DataFrame)
        model_name:     string name for plot titles (optional, inferred from model class name)
    """

    if model_name is None:
        model_name = type(model).__name__

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f"\n{'=' * 40}")
    print(f"Model    : {model_name}")
    print(f"{'=' * 40}")
    print(f"Accuracy : {acc:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print(f"MCC      : {mcc:.4f}")
    print(f"{'=' * 40}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ── Feature names ─────────────────────────────────────────────────────────
    if feature_names is None:
        try:
            feature_names = X.columns.tolist()
        except AttributeError:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # ── Check for staged predict (AdaBoost / gradient boosting) ──────────────
    has_staged = hasattr(model, "staged_predict_proba")

    # ── Plot layout ───────────────────────────────────────────────────────────
    n_plots = 3 if has_staged else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    fig.suptitle(f"{model_name} — Evaluation", fontsize=14, fontweight="bold")

    # Confusion matrix
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(
        ax=axes[0], colorbar=False
    )
    axes[0].set_title("Confusion Matrix")

    # Top 20 feature importances
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = np.array(model.feature_importances_)
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])

    if importances is not None:
        top_idx = np.argsort(importances)[::-1][:20]
        top_names = [feature_names[i] for i in top_idx]
        top_vals = importances[top_idx]

        axes[1].barh(top_names[::-1], top_vals[::-1], color="steelblue")
        axes[1].set_title("Top 20 Feature Importances")
        axes[1].set_xlabel("Importance")
    else:
        axes[1].text(
            0.5, 0.5,
            "Feature importances\nnot available for this model",
            ha="center", va="center", transform=axes[1].transAxes
        )
        axes[1].set_title("Top 20 Feature Importances")

    # Staged ROC-AUC (AdaBoost / gradient boosting only)
    if has_staged:
        staged_auc = [
            roc_auc_score(y_test, p[:, 1])
            for p in model.staged_predict_proba(X_test)
        ]
        axes[2].plot(range(1, len(staged_auc) + 1), staged_auc, color="steelblue")
        axes[2].set_xlabel("Number of Estimators")
        axes[2].set_ylabel("ROC-AUC")
        axes[2].set_title("Staged ROC-AUC")
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"{model_name.lower().replace(' ', '_')}_metrics.png"
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Plot saved to {filename}")

    return {"model": model_name, "accuracy": acc, "roc_auc": auc, "mcc": mcc}