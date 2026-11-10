"""Uploads cached run histories to Weights & Biases after the fact.

Training ran locally on the CPU without a live W&B connection. The curves are
replayed epoch by epoch from results/*.json and tagged "replayed", so their
provenance stays visible. Runs already present in the project are skipped.

    python scripts/replay_wandb.py h00-control h01-batchnorm
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import wandb  # noqa: E402

from delmc.engine import History  # noqa: E402

ENTITY = "University-Projects"
PROJECT = "DEL-EuroSAT-Land-Classifier-Sentinel-2"
RESULTS = REPO / "results"
HASH_SUFFIX = re.compile(r"-[0-9a-f]{12}$")


def replay(path: Path) -> None:
    group = path.parent.name
    history = History.load(path)
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=f"{group}/{HASH_SUFFIX.sub('', path.stem)}",
        group=group,
        job_type="replay",
        tags=["replayed", "zweite-etappe", group],
        config={**history.config, "replayed_from": str(path.relative_to(RESULTS))},
        reinit=True,
    )
    for record in history.records:
        run.log({
            "Epoch": record.epoch,
            "train_loss": record.train_loss,
            "train_accuracy": record.train_accuracy,
            "val_loss": record.val_loss,
            "val_accuracy": record.val_accuracy,
            "val_macro_f1": record.val_macro_f1,
            "seconds": record.seconds,
        })
    if history.final_train:
        run.summary.update({f"final_train_{key}": value for key, value in history.final_train.items()})
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("groups", nargs="+")
    args = parser.parse_args()

    existing = {run.name for run in wandb.Api().runs(f"{ENTITY}/{PROJECT}")}
    for group in args.groups:
        for path in sorted((RESULTS / group).glob("*.json")):
            name = f"{group}/{HASH_SUFFIX.sub('', path.stem)}"
            if name in existing:
                print(f"skip    {name}")
                continue
            replay(path)
            print(f"replay  {name}")


if __name__ == "__main__":
    main()
