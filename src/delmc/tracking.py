"""Experiment tracking against Weights & Biases, with a no-op twin."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

ENTITY = "University-Projects"
PROJECT = "DEL-EuroSAT-Land-Classifier-Sentinel-2"


@runtime_checkable
class Tracker(Protocol):
    """Minimal surface the Trainer depends on."""

    def start(
        self,
        name: str,
        config: dict[str, Any] | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
    ) -> "Tracker": ...

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None: ...

    def watch(self, model: Any) -> None: ...

    def finish(self) -> None: ...


class NullTracker:
    """Records nothing, so any run works offline or inside unit tests."""

    def start(self, name, config=None, group=None, tags=None):
        return self

    def log(self, metrics, step=None):
        pass

    def watch(self, model):
        pass

    def finish(self):
        pass


class WandbTracker:
    """One W&B run.

    The API key is read from the environment or ~/.netrc by the wandb client and
    is never accepted as an argument, so it cannot reach a tracked file.
    """

    def __init__(self, entity: str = ENTITY, project: str = PROJECT) -> None:
        self.entity = entity
        self.project = project
        self._run = None

    def start(self, name, config=None, group=None, tags=None):
        import wandb

        self._run = wandb.init(
            entity=self.entity,
            project=self.project,
            name=name,
            config=config,
            group=group,
            tags=tags,
            reinit=True,
        )
        return self

    def log(self, metrics, step=None):
        if self._run is None:
            raise RuntimeError("start() must be called before log()")
        self._run.log(metrics, step=step)

    def watch(self, model):
        import wandb

        # Per-layer gradient histograms; the gradient-flow study reads these.
        wandb.watch(model, log="all")

    def finish(self):
        if self._run is not None:
            self._run.finish()
            self._run = None
