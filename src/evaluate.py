import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support
from . import config

def save_curves(history):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for axis, metric, title in [(ax[0], "loss", "Loss"), (ax[1], "accuracy", "Accuracy")]:
        axis.plot(history.history.get(metric, []), label="Training")
        axis.plot(history.history.get("val_" + metric, []), label="Validation")
        axis.set_title(title); axis.set_xlabel("Epoch"); axis.legend()
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "training_curves.png", dpi=200); plt.close(fig)

def evaluate(y_true, y_pred, class_ids):
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True); config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = np.arange(len(class_ids))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, target_names=[str(x) for x in class_ids], output_dict=True, zero_division=0)
    metrics = {"overall_accuracy": float(accuracy_score(y_true, y_pred)), "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f), "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)), "class_ids": [int(x) for x in class_ids], "classification_report": report}
    (config.METRICS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(10, 8)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_ids, yticklabels=class_ids, ax=ax)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.set_title("Validation Confusion Matrix")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "confusion_matrix.png", dpi=200); plt.close(fig)
    return metrics

def save_map(predicted_map):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6)); im = ax.imshow(predicted_map, interpolation="nearest")
    ax.set_title("Salinas 2D-CNN Classification Map"); ax.axis("off"); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "classification_map.png", dpi=200); plt.close(fig)
