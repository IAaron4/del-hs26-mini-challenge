"""The measurements and the comparison rule of section 10, computed from cached histories.

Labels are German to match the notebook.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .engine import History
from .models import CNNConfig, ConfigurableCNN
from .probes import ActivationStatsProbe
from .reporting import overfitting_turning_point, per_class_table, smooth
from .utils import seed_everything

SPEED_EPOCH = 9  # zero-based, i.e. after ten epochs
FORMATS = {"Minimum Val-Kosten": "{:.0f}", "s/Epoche": "{:.0f}", "Gradient Layer 1": "{:.2e}"}
ACCURACIES = ("beste Val-Acc", "nach 10 Epochen", "Train-Acc")


def variant(name: str) -> str:
    """Run name without its seed suffix: 'bn-lr0.03-s43' -> 'bn-lr0.03'."""
    return name.rsplit("-s", 1)[0]


def run_metrics(history: History, window: int = 5) -> dict[str, float]:
    """The measurements defined in 10.0, from the smoothed curves where noise would decide."""
    val = smooth(history.series("val_accuracy"), window)
    val_loss = smooth(history.series("val_loss"), window)
    raw = np.asarray(history.series("val_accuracy"))
    final = history.final_train or {"accuracy": float("nan"), "loss": float("nan")}
    first_rms = history.records[0].gradient_rms
    return {
        "beste Val-Acc": max(val),
        "nach 10 Epochen": val[SPEED_EPOCH],
        "Minimum Val-Kosten": overfitting_turning_point(history, window),
        "Train-Acc": final["accuracy"],
        # Smoothed validation end values: single epochs have outliers (5.4).
        "Gap Accuracy": final["accuracy"] - val[-1],
        "Gap Kosten": val_loss[-1] - final["loss"],
        # Deviation from the smoothed curve, so the learning trend itself does not
        # count; from epoch 10 on, after the steep start the window cannot follow.
        "Unruhe": float(np.mean(np.abs(raw - np.asarray(val))[SPEED_EPOCH + 1:])),
        "Gradient Layer 1": next(iter(first_rms.values())) if first_rms else float("nan"),
        "s/Epoche": float(np.mean(history.series("seconds"))),
    }


def metrics_table(histories: dict[str, History], window: int = 5) -> pd.DataFrame:
    """One row per run."""
    rows = [
        {"Variante": variant(name), "Seed": int(name.rsplit("-s", 1)[1]), **run_metrics(history, window)}
        for name, history in histories.items()
    ]
    return pd.DataFrame(rows)


def seed_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Mittelwert ± Standardabweichung pro Variante; bei einem Seed nur der Wert."""
    metrics = [column for column in table.columns if column not in ("Variante", "Seed")]
    rows = []
    for name, group in table.groupby("Variante", sort=False):
        row = {"Variante": name, "Seeds": len(group)}
        for metric in metrics:
            fmt = FORMATS.get(metric, "{:.3f}")
            values = group[metric]
            row[metric] = fmt.format(values.mean())
            if len(group) > 1:
                row[metric] += " ± " + fmt.format(values.std())
        rows.append(row)
    return pd.DataFrame(rows).set_index("Variante")


def control_range(table: pd.DataFrame, metric: str, control: str = "base") -> float:
    """Spread of the three control runs; the threshold for single-seed metrics other than accuracy."""
    values = table.loc[table["Variante"] == control, metric]
    return float(values.max() - values.min())


def compare(values_a, values_b, threshold: float) -> str:
    """Decision rule from 10.0. Says which side is higher; whether higher is better depends on the metric.

    With three runs on each side, one side is higher only if all its runs are
    above all runs of the other side. Otherwise the difference of the means has
    to be larger than `threshold`.
    """
    values_a, values_b = np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)
    if min(len(values_a), len(values_b)) >= 3:
        if values_a.min() > values_b.max():
            return "links höher"
        if values_b.min() > values_a.max():
            return "rechts höher"
        return "nicht entscheidbar"
    difference = values_a.mean() - values_b.mean()
    if abs(difference) > threshold:
        return "links höher" if difference > 0 else "rechts höher"
    return "nicht entscheidbar"


def decide(table: pd.DataFrame, comparisons: list[tuple[str, str, str]]) -> pd.DataFrame:
    """One row per (left, right, metric); 5 percentage points for accuracies, the spread of A otherwise."""
    rows = []
    for left, right, metric in comparisons:
        values = {side: table.loc[table["Variante"] == side, metric] for side in (left, right)}
        threshold = 0.05 if metric in ACCURACIES else control_range(table, metric)
        fmt = FORMATS.get(metric, "{:.3f}")
        rows.append({
            "links": left, "rechts": right, "Kennzahl": metric,
            "Mittel links": fmt.format(values[left].mean()), "Mittel rechts": fmt.format(values[right].mean()),
            "Ergebnis": compare(values[left], values[right], threshold),
        })
    return pd.DataFrame(rows)


def class_f1(histories: dict[str, History], class_names: list[str]) -> pd.DataFrame:
    """F1 per class from each run's final confusion matrix, one column per run."""
    return pd.DataFrame({
        name: per_class_table(history.confusion, class_names)["F1"] for name, history in histories.items()
    }).loc[class_names]


def plot_variants(histories: dict[str, History], labels: dict[str, str], title: str | None = None,
                  window: int = 5, figsize=(12, 4.5)):
    """Smoothed val accuracy and val costs per variant; with several seeds the mean and the min-max band."""
    figure, axes = plt.subplots(1, 2, figsize=figsize)
    for key, label in labels.items():
        runs = [history for name, history in histories.items() if variant(name) == key]
        for axis, series in zip(axes, ("val_accuracy", "val_loss")):
            curves = np.array([smooth(run.series(series), window) for run in runs])
            line, = axis.plot(curves.mean(axis=0), label=f"{label} ({len(runs)})")
            if len(runs) > 1:
                axis.fill_between(range(curves.shape[1]), curves.min(axis=0), curves.max(axis=0),
                                  color=line.get_color(), alpha=0.2)
    axes[0].set_title("Val-Accuracy, geglättet"), axes[0].set_ylabel("Accuracy")
    axes[1].set_title("Val-Kosten, geglättet"), axes[1].set_ylabel("Cross-Entropy")
    for axis in axes:
        axis.set_xlabel("Epoche"), axis.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    return figure


@torch.no_grad()
def activation_std_at_init(config: CNNConfig, inputs: torch.Tensor, seeds, zero_bias: bool = True) -> pd.DataFrame:
    """Std of every layer's output before the ReLU, right after initialisation, one column per seed.

    With `zero_bias` the biases are set to zero, so only the weights differ between schemes.
    """
    columns = {}
    for seed in seeds:
        seed_everything(seed)
        model = ConfigurableCNN(config).eval()
        if zero_bias:
            for module in model.modules():
                if getattr(module, "bias", None) is not None:
                    module.bias.zero_()
        with ActivationStatsProbe(model) as probe:
            model(inputs)
            columns[seed] = {name: std for name, (_mean, std) in probe.snapshot().items()}
    return pd.DataFrame(columns)
