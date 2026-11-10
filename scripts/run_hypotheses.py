"""Runs the training runs of the notebook, one wave per section or hypothesis.

Every run is saved under results/, so the script can be stopped and restarted
and only computes what is still missing.

    python scripts/run_hypotheses.py lernrate batchsize kapazitaet crossval   # sections 5 to 7
    python scripts/run_hypotheses.py control h01 h03                          # section 10
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from delmc.crossval import CrossValidator  # noqa: E402
from delmc.data import EuroSATDataModule  # noqa: E402
from delmc.experiments import ExperimentConfig, ExperimentRunner, ResultCache  # noqa: E402
from delmc.models import CNNConfig  # noqa: E402
from delmc.reporting import smooth  # noqa: E402
from delmc.transfer import TransferConfig, TransferRunner  # noqa: E402
from delmc.utils import select_device  # noqa: E402

SEEDS = (42, 43, 44)
# A variant picked as the best of a grid is confirmed on fresh seeds: the
# selection run itself won a comparison and is biased upwards.
CONFIRM_SEEDS = (43, 44, 45)
SUBSET = 6000
BASE = ExperimentConfig(
    name="base", group="h00-control", cnn=CNNConfig(), optimizer="sgd", lr=0.03,
    batch_size=64, epochs=30, train_subset=SUBSET, probe=True,
)
DIHEDRAL = ("hflip", "vflip", "rot90")


class ProgressTracker:
    """One line per epoch, so a run of several hours can be followed in the log."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.label = f"{config.group}/{config.name}"

    def log(self, metrics: dict, step=None) -> None:
        print(f"  {self.label} epoch {metrics['epoch']:>2}  train_acc={metrics['train_accuracy']:.3f}  "
              f"val_acc={metrics['val_accuracy']:.3f}  {metrics['seconds']:.0f}s", flush=True)


def seeded(config: ExperimentConfig, name: str, group: str, seeds=SEEDS) -> list[ExperimentConfig]:
    return [config.with_(name=f"{name}-s{seed}", group=group, seed=seed) for seed in seeds]


def best_smoothed(history) -> float:
    return max(smooth(history.series("val_accuracy")))


def diverged(history) -> bool:
    """NaN losses, or a first epoch far above the ln(10) = 2.3 of a random guess."""
    losses = [record.train_loss for record in history.records]
    return any(not math.isfinite(loss) for loss in losses) or losses[0] > 3.0


def run_compared(runner: ExperimentRunner, config: ExperimentConfig, partners=()) -> None:
    """On divergence, reruns the config and its comparison partners at learning rate 0.01.

    The spec allows a lower learning rate for other model sizes. Rerunning the
    partners too keeps every comparison at one learning rate.
    """
    if diverged(runner.run(config)):
        for run in (config, *partners):
            name = run.name.replace(f"-lr{run.lr:g}", "").replace("-s", "-lr0.01-s", 1)
            runner.run(run.with_(name=name, lr=0.01))


def lernrate(runner, cache):
    """Learning rate grid for the baseline (6.1)."""
    runner.run_many(BASE.with_(name=f"lr-{lr:g}", group="h0-lernrate", lr=lr, epochs=15)
                    for lr in (0.003, 0.01, 0.03, 0.1))


def batchsize(runner, cache):
    """Batch size grid at learning rate 0.03 (6.2)."""
    runner.run_many(BASE.with_(name=f"batch-{size}", group="h0-batchsize", batch_size=size, epochs=20)
                    for size in (32, 64, 128, 256))


def kapazitaet(runner, cache):
    """Baseline against the double width over 40 epochs (5.3 and 5.4)."""
    wide = BASE.cnn.with_(stage_widths=(64, 128, 256), fc_widths=(512,))
    runner.run(BASE.with_(name="base", group="h0-kapazitaet-lang", epochs=40))
    runner.run(BASE.with_(name="wide", group="h0-kapazitaet-lang", cnn=wide, epochs=40))


def crossval(runner, cache):
    """Five-fold cross-validation of the baseline on the subset, 20 epochs per fold (7)."""
    paths = [cache.root / "h0-crossval" / f"fold-{fold}.json" for fold in range(5)]
    if all(path.exists() for path in paths):
        print("cached  h0-crossval")
        return
    config = BASE.with_(probe=False)
    validator = CrossValidator(runner.data_module, config.build_model, config.build_optimizer, runner.device,
                               folds=5, epochs=20, seed=config.seed, batch_size=config.batch_size)
    for path, history in zip(paths, validator.run()):
        history.config.update(lr=config.lr, batch_size=config.batch_size, epochs=20,
                              train_subset=config.train_subset, seed=config.seed)
        history.save(path)


def control(runner, cache):
    runner.run_many(seeded(BASE, "base", "h00-control"))


def h01(runner, cache):
    group, bn = "h01-batchnorm", BASE.cnn.with_(batch_norm=True)
    runner.run_many(seeded(BASE.with_(cnn=bn), "bn-lr0.03", group))
    runner.run(BASE.with_(name="nobn-lr0.1-s42", group=group, lr=0.1))
    runner.run(BASE.with_(name="bn-lr0.1-s42", group=group, cnn=bn, lr=0.1))


def h02(runner, cache):
    group = "h02-optimizer"
    grids = {
        "momentum": ("sgd", 0.9, [0.001, 0.003, 0.01], 0.0003, 0.03),
        "adam": ("adam", 0.0, [1e-4, 3e-4, 1e-3], 3e-5, 3e-3),
    }
    for label, (optimizer, momentum, grid, below, above) in grids.items():
        def make(lr, seed=42):
            return BASE.with_(name=f"{label}-lr{lr:g}-s{seed}", group=group, optimizer=optimizer,
                              momentum=momentum, lr=lr, seed=seed)

        scores = {lr: best_smoothed(runner.run(make(lr))) for lr in grid}
        # Edge rule from 10.2: one value further when the optimum sits on the border.
        edge = {grid[0]: below, grid[-1]: above}.get(max(scores, key=scores.get))
        if edge is not None:
            scores[edge] = best_smoothed(runner.run(make(edge)))
        best = max(scores, key=scores.get)
        runner.run_many([make(best, seed).with_(name=f"{label}-lr{best:g}-confirm-s{seed}")
                         for seed in CONFIRM_SEEDS])


def h02_winner(cache: ResultCache) -> tuple[str, float, float]:
    """Optimizer, learning rate and momentum of the better three-seed variant in h02."""
    runs = cache.group("h02-optimizer")
    finalists = {name.rsplit("-s", 1)[0] for name in runs if "-confirm-" in name}
    winner = max(finalists, key=lambda prefix: mean(best_smoothed(runs[f"{prefix}-s{s}"]) for s in CONFIRM_SEEDS))
    config = runs[f"{winner}-s{CONFIRM_SEEDS[0]}"].config
    return config["optimizer"], config["lr"], config["momentum"]


def h03(runner, cache):
    group = "h03-augmentation"
    runner.run(BASE.with_(name="hflip-s42", group=group, augment=("hflip",)))
    runner.run_many(seeded(BASE.with_(augment=DIHEDRAL), "dihedral", group))
    runner.run(BASE.with_(name="dihedral-color-s42", group=group, augment=DIHEDRAL + ("color",)))


def h04(runner, cache):
    group = "h04-dropout"

    def fc(p, seed=42):
        return BASE.with_(name=f"fc{p}-s{seed}", group=group, cnn=BASE.cnn.with_(dropout=p), seed=seed)

    scores = {p: best_smoothed(runner.run(fc(p))) for p in (0.2, 0.5, 0.7)}
    runner.run(BASE.with_(name="conv0.2-s42", group=group, cnn=BASE.cnn.with_(conv_dropout=0.2)))
    best = max(scores, key=scores.get)
    runner.run_many([fc(best, seed).with_(name=f"fc{best}-confirm-s{seed}") for seed in CONFIRM_SEEDS])


CONTROL = BASE.with_(name="base-s42")
DEPTH6 = BASE.with_(name="depth6-s42", group="h05-depth", cnn=BASE.cnn.with_(convs_per_stage=2))
BN_BASE = BASE.with_(name="bn-lr0.03-s42", group="h01-batchnorm", cnn=BASE.cnn.with_(batch_norm=True))


def h05(runner, cache):
    group = "h05-depth"
    run_compared(runner, DEPTH6, partners=[CONTROL])
    run_compared(runner, BASE.with_(name="depth9-s42", group=group, cnn=BASE.cnn.with_(convs_per_stage=3)),
                 partners=[CONTROL, DEPTH6])
    run_compared(runner, BASE.with_(name="depth6-bn-s42", group=group,
                                    cnn=DEPTH6.cnn.with_(batch_norm=True)), partners=[DEPTH6, BN_BASE])


def h06(runner, cache):
    group = "h06-init"
    run_compared(runner, BASE.with_(name="depth6-he-s42", group=group, cnn=DEPTH6.cnn.with_(init="kaiming")),
                 partners=[DEPTH6])
    run_compared(runner, BASE.with_(name="base-he-s42", group=group, cnn=BASE.cnn.with_(init="kaiming")),
                 partners=[CONTROL])


def h07(runner, cache):
    group = "h07-width"
    runner.run(BASE.with_(name="half-s42", group=group,
                          cnn=BASE.cnn.with_(stage_widths=(16, 32, 64), fc_widths=(128,))))
    runner.run_many(seeded(BASE.with_(cnn=BASE.cnn.with_(stage_widths=(64, 128, 256), fc_widths=(512,))),
                           "double", group))


def h08(runner, cache):
    runner.run(BASE.with_(name="kernel5-s42", group="h08-kernel", cnn=BASE.cnn.with_(kernel_size=5)))


def h09(runner, cache):
    runner.run_many(seeded(BASE.with_(cnn=BASE.cnn.with_(pooling="avg")), "avg", "h09-pooling"))


def h10(runner, cache):
    optimizer, lr, momentum = h02_winner(cache)
    transfer = TransferRunner(runner.data_module, runner.device, cache)
    for mode in ("frozen", "finetune_last"):
        transfer.run(TransferConfig(name=f"resnet18-{mode}-s42", group="h10-transfer", mode=mode,
                                    optimizer=optimizer, lr=lr, momentum=momentum, epochs=BASE.epochs,
                                    train_subset=SUBSET))


def combined(cache: ResultCache) -> ExperimentConfig:
    """The own CNN with every change that won its comparison in section 10.

    Depth 6 only together with BatchNorm (h05), dropout at the confirmed FC rate
    (h04), dihedral augmentation (h03), and the optimizer of h02.
    """
    optimizer, lr, momentum = h02_winner(cache)
    return BASE.with_(name="combined-s42", group="h11-best", optimizer=optimizer, lr=lr, momentum=momentum,
                      cnn=DEPTH6.cnn.with_(batch_norm=True, dropout=0.7), augment=DIHEDRAL)


def best(runner, cache):
    runner.run(combined(cache))


def final(runner, cache):
    """Basismodell and the better candidate on the whole training split (2.4)."""
    data = EuroSATDataModule(root=REPO / "data", num_workers=4, train_subset=None).prepare()
    full = ExperimentRunner(data, runner.device, cache, tracker_factory=ProgressTracker)
    full.run(BASE.with_(name="base-full-s42", group="final", train_subset=None))

    own = cache.group("h11-best")["combined-s42"]
    transfer = cache.group("h10-transfer")["resnet18-finetune_last-s42"]
    if best_smoothed(own) > best_smoothed(transfer):
        full.run(combined(cache).with_(name="combined-full-s42", group="final", train_subset=None))
        return
    optimizer, lr, momentum = h02_winner(cache)
    # On the subset the val accuracy barely moved after the first epochs, and the val
    # costs bottomed out after about 1800 updates, six epochs on the full split.
    TransferRunner(data, runner.device, cache, tracker_factory=ProgressTracker).run(
        TransferConfig(name="resnet18-finetune_last-full-s42", group="final", mode="finetune_last",
                       optimizer=optimizer, lr=lr, momentum=momentum, epochs=15, train_subset=None))


WAVES = {"lernrate": lernrate, "batchsize": batchsize, "kapazitaet": kapazitaet, "crossval": crossval,
         "control": control, "h01": h01, "h02": h02, "h03": h03, "h04": h04, "h05": h05,
         "h06": h06, "h07": h07, "h08": h08, "h09": h09, "h10": h10, "best": best, "final": final}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("waves", nargs="+", choices=list(WAVES))
    args = parser.parse_args()

    data = EuroSATDataModule(root=REPO / "data", num_workers=4, train_subset=SUBSET).prepare()
    cache = ResultCache()
    runner = ExperimentRunner(data, select_device(), cache, tracker_factory=ProgressTracker)
    for wave in args.waves:
        print(f"=== {wave}", flush=True)
        WAVES[wave](runner, cache)


if __name__ == "__main__":
    main()
