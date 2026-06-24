import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)

try:
    from . import config
except ImportError:
    import config


def _scores_from_model(name, model, X_test):
    y_pred = model.predict(X_test)
    proba  = model.predict_proba(X_test)[:, 1]
    return y_pred, proba


def evaluate_all(models_dict, X_test, y_test):
    print("\n[8] Evaluating models on the test set")
    results = {}

    for name, model in models_dict.items():
        y_pred, proba = _scores_from_model(name, model, X_test)
        results[name] = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall":    recall_score(y_test, y_pred, zero_division=0),
            "f1":        f1_score(y_test, y_pred, zero_division=0),
            "roc_auc":   roc_auc_score(y_test, proba),
            "y_pred":    y_pred,
            "proba":     proba,
        }
        print(f"\n    === {name} ===")
        print(f"    Accuracy : {results[name]['accuracy']:.4f}")
        print(f"    Precision: {results[name]['precision']:.4f}")
        print(f"    Recall   : {results[name]['recall']:.4f}")
        print(f"    F1-score : {results[name]['f1']:.4f}")
        print(f"    ROC-AUC  : {results[name]['roc_auc']:.4f}")

    return results


def comparison_table(results):
    print("\n[9] Performance comparison table")
    rows = []
    for name, r in results.items():
        rows.append({
            "Model":     name,
            "Accuracy":  round(r["accuracy"],  4),
            "Precision": round(r["precision"], 4),
            "Recall":    round(r["recall"],    4),
            "F1-score":  round(r["f1"],        4),
            "ROC-AUC":   round(r["roc_auc"],   4),
        })
    table = pd.DataFrame(rows).sort_values("F1-score", ascending=False).reset_index(drop=True)
    print("\n" + table.to_string(index=False))

    config.ensure_directories()
    csv_path = os.path.join(config.OUTPUTS_DIR, "model_comparison.csv")
    table.to_csv(csv_path, index=False)
    print(f"\n    Saved table to: {csv_path}")
    return table


def plot_confusion_matrices(results, y_test):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, r) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, r["y_pred"])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Fraud"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Fraud"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    path = os.path.join(config.OUTPUTS_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved confusion matrices to: {path}")


def plot_roc_curves(results, y_test):
    plt.figure(figsize=(7, 6))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["proba"])
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {r['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(config.OUTPUTS_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved ROC curves to: {path}")


def plot_f1_comparison(table):
    plt.figure(figsize=(7, 4))
    plt.bar(table["Model"], table["F1-score"], color="#4C72B0")
    plt.ylabel("F1-score")
    plt.title("Model comparison (F1-score)")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    for i, v in enumerate(table["F1-score"]):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.tight_layout()
    path = os.path.join(config.OUTPUTS_DIR, "f1_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved F1 comparison chart to: {path}")


def run_evaluation(models_dict, X_test, y_test):
    results = evaluate_all(models_dict, X_test, y_test)
    table   = comparison_table(results)
    print("\n[10] Saving figures")
    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)
    plot_f1_comparison(table)
    return results, table
