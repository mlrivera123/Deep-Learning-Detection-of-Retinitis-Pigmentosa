#!/usr/bin/env python3
"""
Combined script that generates all metrics, figures, and hyperparameter summaries
from a single experiment directory containing 5-fold CV results.

Usage:
    python generate_paper_metrics.py /path/to/experiment_dir

Expected directory structure:
    experiment_dir/
        0/ 1/ 2/ 3/ 4/   (fold subdirectories)
            probabilities.npy
            labels.npy
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t as student_t
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_fold_metrics(y_true, y_prob):
    """Compute all binary classification metrics for a single fold."""
    y_pred = np.argmax(y_prob, axis=1)
    y_score = y_prob[:, 1]

    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average="macro")
    sensitivity = recall_score(y_true, y_pred, pos_label=1)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    auc_val = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    mcc = matthews_corrcoef(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        "AUC": auc_val,
        "AUPRC": ap,
        "Accuracy": acc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision (PPV)": ppv,
        "NPV": npv,
        "Recall (Macro)": recall,
        "F1": f1,
        "F1 (Macro)": f1_macro,
        "MCC": mcc,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def fmt(x):
    """Format a float to 2 decimal places."""
    return f"{x:.2f}"


def fmt4(x):
    """Format a float to 4 decimal places."""
    return f"{x:.4f}"


def format_mean_std(vals):
    """Format as mean (lower, upper) using mean +/- std, clipped to [0,1]."""
    vals = np.asarray(vals, dtype=float)
    m, s = float(np.mean(vals)), float(np.std(vals))
    lower = max(0.0, m - s)
    upper = min(1.0, m + s)
    return f"{fmt(m)} ({fmt(lower)}, {fmt(upper)})"



# Bounds for clipping CI values to their valid range.
# All proportion/rate metrics live in [0, 1]; MCC lives in [-1, 1].
_METRIC_BOUNDS = {
    "MCC": (-1.0, 1.0),
}
_DEFAULT_BOUNDS = (0.0, 1.0)


def format_ci(vals, confidence=0.95, lo=0.0, hi=1.0):
    """Format as mean (CI_lower, CI_upper) using 95% Student's t CI, clipped to [lo, hi]."""
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    if n < 2:
        return f"{fmt(vals[0])} (n/a)"
    mean = float(np.mean(vals))
    sem = float(np.std(vals, ddof=1)) / np.sqrt(n)
    t_score = student_t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_score * sem
    lower = max(lo, mean - margin)
    upper = min(hi, mean + margin)
    return f"{fmt(mean)} ({fmt(lower)}, {fmt(upper)})"


def plot_roc_curve(y_true_folds, y_score_folds, output_path):
    """Plot per-fold and pooled ROC curves."""
    plt.figure(figsize=(8, 6), dpi=150)

    # Per-fold curves
    for i, (yt, ys) in enumerate(zip(y_true_folds, y_score_folds)):
        fpr, tpr, _ = roc_curve(yt, ys)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, alpha=0.4, label=f"Fold {i} (AUC = {fmt(roc_auc)})")

    # Pooled curve
    yt_all = np.concatenate(y_true_folds)
    ys_all = np.concatenate(y_score_folds)
    fpr, tpr, _ = roc_curve(yt_all, ys_all)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"Pooled (AUC = {fmt(roc_auc)})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance (AUC = 0.50)")
    plt.xlabel("False Positive Rate", fontsize=16)
    plt.ylabel("True Positive Rate", fontsize=16)
    plt.legend(loc="lower right", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"ROC curve saved to {output_path}")


def plot_precision_recall_curve(y_true_folds, y_score_folds, output_path):
    """Plot per-fold and pooled precision-recall curves."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    for i, (yt, ys) in enumerate(zip(y_true_folds, y_score_folds)):
        prec, rec, _ = precision_recall_curve(yt, ys)
        ap = average_precision_score(yt, ys)
        ax.plot(rec, prec, lw=1, alpha=0.4, label=f"Fold {i} (AP = {fmt(ap)})")

    yt_all = np.concatenate(y_true_folds)
    ys_all = np.concatenate(y_score_folds)
    prec, rec, _ = precision_recall_curve(yt_all, ys_all)
    ap = average_precision_score(yt_all, ys_all)
    ax.plot(rec, prec, lw=2, color="blue", label=f"Pooled (AP = {fmt(ap)})")

    prevalence = np.mean(yt_all)
    ax.axhline(y=prevalence, linestyle="--", color="gray", label=f"Baseline ({fmt(prevalence)})")

    ax.set_xlabel("Recall", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)
    ax.legend(loc="lower left", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Precision-Recall curve saved to {output_path}")


def plot_confusion_matrix(cm, output_path, class_names=None):
    """Plot a single confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names or ["0", "1"],
        yticklabels=class_names or ["0", "1"],
        annot_kws={"fontsize": 16},
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=14)
    ax.set_ylabel("True Label", fontsize=14)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Confusion matrix saved to {output_path}")


def plot_per_fold_confusion_matrices(y_true_folds, y_prob_folds, output_path, class_names=None):
    """Plot all per-fold confusion matrices in a single figure."""
    n = len(y_true_folds)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), dpi=300)
    if n == 1:
        axes = [axes]
    for i, (yt, yp) in enumerate(zip(y_true_folds, y_prob_folds)):
        y_pred = np.argmax(yp, axis=1)
        cm = confusion_matrix(yt, y_pred, labels=[0, 1])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names or ["0", "1"],
            yticklabels=class_names or ["0", "1"],
            annot_kws={"fontsize": 14},
            ax=axes[i],
        )
        axes[i].set_title(f"Fold {i}", fontsize=13)
        axes[i].set_xlabel("Predicted", fontsize=11)
        axes[i].set_ylabel("True", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Per-fold confusion matrices saved to {output_path}")


def plot_metrics_summary_table(summary_df, output_path):
    """Render the summary DataFrame as a publication-quality table image."""
    fig, ax = plt.subplots(figsize=(14, 3), dpi=300)
    ax.axis("off")
    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(summary_df.columns))))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Summary table image saved to {output_path}")


def extract_hyperparameters(base_path):
    """Extract hyperparameters from config.json and training_args.bin in fold 0."""
    lines = []
    fold0 = os.path.join(base_path, "0")

    # Model config
    config_path = os.path.join(fold0, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        lines.append("=" * 60)
        lines.append("MODEL CONFIGURATION (config.json)")
        lines.append("=" * 60)
        key_fields = [
            "architectures", "model_type", "hidden_size", "num_hidden_layers",
            "num_attention_heads", "intermediate_size", "hidden_act",
            "hidden_dropout_prob", "attention_probs_dropout_prob",
            "image_size", "patch_size", "num_channels", "id2label",
            "problem_type",
        ]
        for k in key_fields:
            if k in config:
                val = config[k]
                if isinstance(val, list) and len(val) == 1:
                    val = val[0]
                lines.append(f"  {k}: {val}")
        lines.append("")

    # Training args
    training_args_path = os.path.join(fold0, "training_args.bin")
    if os.path.exists(training_args_path):
        try:
            import torch
            args = torch.load(training_args_path, map_location="cpu", weights_only=False)
            d = vars(args)
            lines.append("=" * 60)
            lines.append("TRAINING HYPERPARAMETERS (training_args.bin)")
            lines.append("=" * 60)
            key_args = [
                "learning_rate", "num_train_epochs", "per_device_train_batch_size",
                "per_device_eval_batch_size", "gradient_accumulation_steps",
                "warmup_ratio", "warmup_steps", "weight_decay",
                "adam_beta1", "adam_beta2", "adam_epsilon",
                "max_grad_norm", "lr_scheduler_type", "optim",
                "seed", "fp16", "bf16", "label_smoothing_factor",
                "metric_for_best_model", "load_best_model_at_end",
                "eval_strategy", "save_strategy",
            ]
            for k in key_args:
                if k in d:
                    lines.append(f"  {k}: {d[k]}")
            lines.append("")
        except ImportError:
            lines.append("  (torch not available - cannot read training_args.bin)")
            lines.append("")

    # Training results from fold 0
    for fname in ["train_results.json", "all_metrics.json"]:
        fpath = os.path.join(fold0, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
            lines.append("=" * 60)
            lines.append(f"TRAINING RESULTS ({fname})")
            lines.append("=" * 60)
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, dict):
                        lines.append(f"  {k}:")
                        for kk, vv in v.items():
                            lines.append(f"    {kk}: {vv}")
                    else:
                        lines.append(f"  {k}: {v}")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate all paper-ready metrics, figures, and hyperparameter summary "
                    "from a 5-fold CV experiment directory."
    )
    parser.add_argument(
        "experiment_path",
        type=str,
        help="Path to the experiment directory containing fold subdirectories 0-4.",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=None,
        help="Optional class names for confusion matrix labels (e.g. --class-names Negative Positive).",
    )
    args = parser.parse_args()

    base_path = os.path.normpath(args.experiment_path)
    experiment_name = os.path.basename(base_path)
    output_dir = os.path.join(base_path, "paper_outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Try to read class names from config.json if not provided
    class_names = args.class_names
    if class_names is None:
        config_path = os.path.join(base_path, "0", "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            if "id2label" in config:
                id2label = config["id2label"]
                class_names = [id2label[str(i)] for i in sorted(int(k) for k in id2label)]

    print(f"Experiment: {experiment_name}")
    print(f"Output directory: {output_dir}")
    if class_names:
        print(f"Class names: {class_names}")
    print()


    y_true_folds = []
    y_prob_folds = []
    y_score_folds = []
    fold_metrics = []

    # Metric keys excluding raw counts
    metric_keys = [
        "AUC", "AUPRC", "Accuracy", "Sensitivity", "Specificity",
        "Precision (PPV)", "NPV", "Recall (Macro)", "F1", "F1 (Macro)", "MCC",
    ]

    for i in range(5):
        prob_file = os.path.join(base_path, str(i), "probabilities.npy")
        label_file = os.path.join(base_path, str(i), "labels.npy")
        if not os.path.exists(prob_file) or not os.path.exists(label_file):
            print(f"  Skipping fold {i} (missing files)")
            continue

        y_prob = np.load(prob_file, allow_pickle=True)
        y_true = np.load(label_file, allow_pickle=True)

        if y_prob.ndim != 2 or y_prob.shape[1] != 2:
            print(f"  Skipping fold {i}: expected binary probabilities, got shape {y_prob.shape}")
            continue

        y_true_folds.append(y_true)
        y_prob_folds.append(y_prob)
        y_score_folds.append(y_prob[:, 1])

        metrics = compute_fold_metrics(y_true, y_prob)
        fold_metrics.append(metrics)
        print(f"  Fold {i}: AUC={fmt(metrics['AUC'])}  Acc={fmt(metrics['Accuracy'])}  "
              f"Sens={fmt(metrics['Sensitivity'])}  Spec={fmt(metrics['Specificity'])}  "
              f"F1={fmt(metrics['F1'])}  MCC={fmt(metrics['MCC'])}")

    if not fold_metrics:
        print("ERROR: No valid folds found. Check the experiment path.")
        sys.exit(1)

    n_folds = len(fold_metrics)
    print(f"\n  {n_folds} folds loaded.\n")


    yt_all = np.concatenate(y_true_folds)
    yp_all = np.concatenate(y_prob_folds)
    pooled = compute_fold_metrics(yt_all, yp_all)

    print(f"  Pooled AUC:  {fmt(pooled['AUC'])}")
    print(f"  Pooled AUPRC: {fmt(pooled['AUPRC'])}")
    print(f"  Pooled Acc:  {fmt(pooled['Accuracy'])}")
    print(f"  Pooled F1:   {fmt(pooled['F1'])}")
    print()


    per_fold_rows = []
    all_metric_keys = metric_keys + ["TP", "TN", "FP", "FN"]
    for i, m in enumerate(fold_metrics):
        row = {"Fold": i}
        for k in all_metric_keys:
            row[k] = m[k]
        per_fold_rows.append(row)

    per_fold_df = pd.DataFrame(per_fold_rows)

    agg_row_std = {"Metric": "Mean ± Std"}
    agg_row_ci = {"Metric": "Mean (95% CI)"}
    pooled_row = {"Metric": "Pooled"}

    for k in metric_keys:
        vals = [m[k] for m in fold_metrics]
        lo, hi = _METRIC_BOUNDS.get(k, _DEFAULT_BOUNDS)
        agg_row_std[k] = format_mean_std(vals)
        agg_row_ci[k] = format_ci(vals, lo=lo, hi=hi)
        pooled_row[k] = fmt(pooled[k])

    summary_df = pd.DataFrame([agg_row_std, agg_row_ci, pooled_row])


    per_fold_csv = os.path.join(output_dir, "per_fold_metrics.csv")
    per_fold_df.to_csv(per_fold_csv, index=False, float_format="%.4f")
    print(f"Per-fold metrics saved to {per_fold_csv}")

    summary_csv = os.path.join(output_dir, "summary_metrics.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary metrics saved to {summary_csv}")


    plot_roc_curve(
        y_true_folds, y_score_folds,
        os.path.join(output_dir, "roc_curve.png"),
    )

    plot_precision_recall_curve(
        y_true_folds, y_score_folds,
        os.path.join(output_dir, "precision_recall_curve.png"),
    )

    # Aggregated confusion matrix
    cm_agg = confusion_matrix(yt_all, np.argmax(yp_all, axis=1), labels=[0, 1])
    plot_confusion_matrix(
        cm_agg,
        os.path.join(output_dir, "confusion_matrix_aggregated.png"),
        class_names=class_names,
    )

    # Per-fold confusion matrices
    plot_per_fold_confusion_matrices(
        y_true_folds, y_prob_folds,
        os.path.join(output_dir, "confusion_matrices_per_fold.png"),
        class_names=class_names,
    )

    # Summary table as image
    plot_metrics_summary_table(
        summary_df,
        os.path.join(output_dir, "summary_table.png"),
    )


    hp_text = extract_hyperparameters(base_path)
    if hp_text.strip():
        hp_path = os.path.join(output_dir, "hyperparameters.txt")
        with open(hp_path, "w") as f:
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Path: {base_path}\n")
            f.write(f"Folds used: {n_folds}\n\n")
            f.write(hp_text)
        print(f"Hyperparameters saved to {hp_path}")
    else:
        print("No hyperparameter files found (config.json / training_args.bin). Skipping.")


    report_path = os.path.join(output_dir, "metrics_report.txt")
    with open(report_path, "w") as f:
        f.write(f"{'=' * 60}\n")
        f.write(f"METRICS REPORT: {experiment_name}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Experiment path: {base_path}\n")
        f.write(f"Number of folds: {n_folds}\n")
        f.write(f"Total samples (pooled): {len(yt_all)}\n")
        f.write(f"Class distribution: {dict(zip(*np.unique(yt_all, return_counts=True)))}\n\n")

        f.write(f"{'-' * 60}\n")
        f.write("AGGREGATED METRICS (Mean ± Std)\n")
        f.write(f"{'-' * 60}\n")
        for k in metric_keys:
            vals = [m[k] for m in fold_metrics]
            f.write(f"  {k:20s}: {format_mean_std(vals)}\n")
        f.write("\n")

        f.write(f"{'-' * 60}\n")
        f.write("AGGREGATED METRICS (Mean, 95% CI)\n")
        f.write(f"{'-' * 60}\n")
        for k in metric_keys:
            vals = [m[k] for m in fold_metrics]
            lo, hi = _METRIC_BOUNDS.get(k, _DEFAULT_BOUNDS)
            f.write(f"  {k:20s}: {format_ci(vals, lo=lo, hi=hi)}\n")
        f.write("\n")

        f.write(f"{'-' * 60}\n")
        f.write("POOLED METRICS (all folds concatenated)\n")
        f.write(f"{'-' * 60}\n")
        for k in metric_keys:
            f.write(f"  {k:20s}: {fmt4(pooled[k])}\n")
        f.write(f"\n  Confusion Matrix (Pooled):\n")
        f.write(f"    TN={pooled['TN']}  FP={pooled['FP']}\n")
        f.write(f"    FN={pooled['FN']}  TP={pooled['TP']}\n\n")

        f.write(f"{'-' * 60}\n")
        f.write("PER-FOLD METRICS\n")
        f.write(f"{'-' * 60}\n")
        for i, m in enumerate(fold_metrics):
            f.write(f"\n  Fold {i}:\n")
            for k in all_metric_keys:
                if k in ("TP", "TN", "FP", "FN"):
                    f.write(f"    {k:20s}: {m[k]}\n")
                else:
                    f.write(f"    {k:20s}: {fmt4(m[k])}\n")

    print(f"Full metrics report saved to {report_path}")
    print(f"\nAll outputs written to: {output_dir}/")


if __name__ == "__main__":
    main()
