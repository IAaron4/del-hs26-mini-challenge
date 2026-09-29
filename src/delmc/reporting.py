"""Tables and plots built from cached histories.

Labels are German to match the notebook; established English terms are kept.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .engine import History


def summary_table(histories: dict[str, History], use_best: bool = False) -> pd.DataFrame:
    """One row per run, with the train/val gap that drives the regularisation story."""
    rows = []
    for name, history in histories.items():
        record = history.best("val_accuracy") if use_best else history.records[-1]
        rows.append(
            {
                "Run": name,
                "Parameter": history.config.get("params"),
                "Conv-Tiefe": history.config.get("conv_depth"),
                "Epoche": record.epoch,
                "Train-Accuracy": record.train_accuracy,
                "Val-Accuracy": record.val_accuracy,
                "Gap": record.train_accuracy - record.val_accuracy,
                "Val-Macro-F1": record.val_macro_f1,
                "Train-Loss": record.train_loss,
                "Val-Loss": record.val_loss,
                "s/Epoche": record.seconds,
            }
        )
    return pd.DataFrame(rows).sort_values("Val-Accuracy", ascending=False).reset_index(drop=True)


def plot_learning_curves(histories: dict[str, History], title: str | None = None, figsize=(12, 4.5)):
    """Kostenfunktion and Accuracy side by side; train solid, validation dashed."""
    figure, axes = plt.subplots(1, 2, figsize=figsize)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for index, (name, history) in enumerate(histories.items()):
        color = colors[index % len(colors)]
        epochs = history.series("epoch")
        axes[0].plot(epochs, history.series("train_loss"), "-", color=color, label=f"{name} (Train)")
        axes[0].plot(epochs, history.series("val_loss"), "--", color=color, label=f"{name} (Val)")
        axes[1].plot(epochs, history.series("train_accuracy"), "-", color=color, label=f"{name} (Train)")
        axes[1].plot(epochs, history.series("val_accuracy"), "--", color=color, label=f"{name} (Val)")

    axes[0].set_xlabel("Epoche"), axes[0].set_ylabel("Cross-Entropy"), axes[0].set_title("Kostenfunktion")
    axes[1].set_xlabel("Epoche"), axes[1].set_ylabel("Accuracy"), axes[1].set_title("Accuracy")
    for axis in axes:
        axis.grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower right")
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_overfitting_gap(histories: dict[str, History], figsize=(7, 4.5)):
    """Train minus validation accuracy per epoch -- the overfitting signal itself."""
    figure, axis = plt.subplots(figsize=figsize)
    for name, history in histories.items():
        gap = [t - v for t, v in zip(history.series("train_accuracy"), history.series("val_accuracy"))]
        axis.plot(history.series("epoch"), gap, "-o", markersize=3, label=name)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Epoche"), axis.set_ylabel("Train - Val Accuracy"), axis.set_title("Overfitting-Gap")
    axis.grid(alpha=0.3), axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


def plot_gradient_flow(history: History, epochs: list[int] | None = None, figsize=(8, 4.5)):
    """Gradient norm per layer; exponential decay with depth is the phenomenon."""
    records = [r for r in history.records if r.gradient_norms]
    if not records:
        raise ValueError("history carries no gradient snapshots -- run with probe=True")
    chosen = records if epochs is None else [r for r in records if r.epoch in epochs]

    figure, axis = plt.subplots(figsize=figsize)
    for record in chosen:
        names = list(record.gradient_norms)
        axis.plot(range(len(names)), list(record.gradient_norms.values()), "-o", markersize=4, label=f"Epoche {record.epoch}")
    axis.set_yscale("log")
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    axis.set_xlabel("Layer (Eingang → Ausgang)"), axis.set_ylabel(r"$\|\partial L/\partial W_l\|$")
    axis.set_title("Gradientenfluss pro Layer")
    axis.grid(alpha=0.3), axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


def plot_gradient_ratio(histories: dict[str, History], figsize=(7, 4.5)):
    """First-layer over last-layer gradient norm across training."""
    figure, axis = plt.subplots(figsize=figsize)
    for name, history in histories.items():
        points = [(r.epoch, r.gradient_ratio) for r in history.records if r.gradient_ratio is not None]
        if points:
            axis.plot(*zip(*points), "-o", markersize=3, label=name)
    axis.set_yscale("log")
    axis.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    axis.set_xlabel("Epoche"), axis.set_ylabel("Norm erster / letzter Layer")
    axis.set_title("Gradienten-Verhältnis über das Training")
    axis.grid(alpha=0.3), axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


def per_class_table(confusion, class_names: list[str]) -> pd.DataFrame:
    """Precision, Recall, F1 und Support pro Klasse aus der Confusion Matrix."""
    matrix = np.asarray(confusion, dtype=float)
    true_positive = matrix.diagonal()
    predicted = matrix.sum(axis=0)
    actual = matrix.sum(axis=1)

    # A class absent from predictions or from the split scores 0, not NaN.
    def ratio(numerator, denominator):
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)

    precision = ratio(true_positive, predicted)
    recall = ratio(true_positive, actual)
    f1 = ratio(2 * precision * recall, precision + recall)

    frame = pd.DataFrame(
        {"Precision": precision, "Recall": recall, "F1": f1, "Support": actual.astype(int)},
        index=class_names,
    )
    frame.index.name = "Klasse"
    return frame.sort_values("F1", ascending=False)


def plot_confusion_matrix(confusion, class_names: list[str], normalize: bool = True, figsize=(7.5, 6.5)):
    """Zeilen sind die wahre Klasse, Spalten die Vorhersage."""
    matrix = np.asarray(confusion, dtype=float)
    if normalize:
        totals = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)

    figure, axis = plt.subplots(figsize=figsize)
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
    axis.set_xticks(range(len(class_names))), axis.set_yticks(range(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    axis.set_yticklabels(class_names, fontsize=8)
    axis.set_xlabel("Vorhersage"), axis.set_ylabel("Wahre Klasse")
    axis.set_title("Confusion Matrix" + (" (zeilennormiert)" if normalize else ""))

    threshold = matrix.max() / 2
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            value = matrix[row, column]
            if value > 0.005:
                axis.text(column, row, f"{value:.2f}" if normalize else f"{int(value)}",
                          ha="center", va="center", fontsize=7,
                          color="white" if value > threshold else "black")
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    return figure


def plot_class_performance(confusion, class_names: list[str], figsize=(8, 4.5)):
    """F1 pro Klasse, sortiert - beantwortet direkt was gut und was schlecht laeuft."""
    table = per_class_table(confusion, class_names)
    figure, axis = plt.subplots(figsize=figsize)
    colors = ["seagreen" if v >= table["F1"].median() else "indianred" for v in table["F1"]]
    axis.barh(table.index[::-1], table["F1"][::-1], color=colors[::-1])
    axis.axvline(table["F1"].mean(), color="black", linestyle=":", linewidth=1,
                 label=f"Macro-F1 = {table['F1'].mean():.3f}")
    axis.set_xlabel("F1"), axis.set_xlim(0, 1), axis.set_title("Performance pro Klasse")
    axis.legend(fontsize=8), axis.grid(alpha=0.3, axis="x")
    figure.tight_layout()
    return figure


def plot_metric_vs_hyperparameter(histories: dict[str, History], key: str, xlabel: str,
                                  log_x: bool = True, figsize=(7, 4.5)):
    """Val-Accuracy (final und bestes Epoch) gegen einen skalaren Hyperparameter."""
    points = []
    for history in histories.values():
        value = history.config.get(key)
        if value is None:
            continue
        points.append((value, history.records[-1].val_accuracy,
                       history.best("val_accuracy").val_accuracy,
                       history.records[-1].train_accuracy))
    points.sort()
    x, final, best, train = zip(*points)

    figure, axis = plt.subplots(figsize=figsize)
    axis.plot(x, train, "s--", color="grey", label="Train (final)")
    axis.plot(x, final, "o-", label="Validation (final)")
    axis.plot(x, best, "^-", label="Validation (bestes Epoch)")
    if log_x:
        axis.set_xscale("log")
    axis.set_xlabel(xlabel), axis.set_ylabel("Accuracy")
    axis.set_title(f"Accuracy gegen {xlabel}")
    axis.grid(alpha=0.3), axis.legend(fontsize=8)
    figure.tight_layout()
    return figure
