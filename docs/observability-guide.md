# Observability and loggers

DiffBloch reports progress by emitting typed events. A logger is any object with one method:

```python
def report(event): ...
```

The scientific core hands loggers plain event objects; logger backends decide whether to print,
write CSV rows, or send values to an experiment tracker.

## Built-in loggers

| Logger | Role |
|---|---|
| `ConsoleLogger` | Default CLI live progress stream. |
| `CSVLogger` | Long-format CSV rows: `channel, step, metric, value`. |
| `WandbLogger` | Optional Weights & Biases backend. |
| `CometLogger` | Optional Comet ML backend. |
| `MultiLogger` | Fan-out to several loggers. |
| `RecordingLogger` | In-memory test/debug sink. |

The third-party logger modules import their SDKs lazily, so the base package does not require W&B
or Comet unless you use those backends.

## API example: console + CSV

```python
from pathlib import Path

from diffBloch.app import CSVLogger, ConsoleLogger, run_experiment
from diffBloch.observability import MultiLogger

logger = MultiLogger((
    ConsoleLogger(),
    CSVLogger(Path("quartz-events.csv")),
))

result = run_experiment("examples/experiments/quartz-checkpoint", logger=logger)
print(result.mean_r_obs)
```

## API example: custom logger

```python
from dataclasses import dataclass, field

from diffBloch.observability import Event

@dataclass
class LastEventLogger:
    last: Event | None = None

    def report(self, event: Event) -> None:
        self.last = event
```

## Event shape

Every event exposes:

- `channel` — e.g. `rotation`, `coupling`, `refinement`;
- `step` — a rotation index, refinement iteration, or `None` for run-level summaries;
- `measurements` — numeric values suitable for plotting or logging.

See the [observability API](api/observability.md) for the event classes.
