# Observability and loggers

A refinement run reports the quantities that matter to a crystallographer as it goes: wR2 and R_obs
per rotation during orientation fitting, thickness and its residual per rotation during thickness
fitting, and the epoch-by-epoch loss, wR2, and R_obs during structure refinement. diffBloch keeps
that reporting separate from the simulation itself — the numerical core never decides how or whether
a value gets displayed, it just hands out plain event objects, and whatever loggers you attach decide
what to do with them: print to the console, write a CSV row, or send to an experiment tracker such as
Weights & Biases. A logger is any object with one method:

```python
def report(event): ...
```

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

A minimal logger that just remembers the last event — enough to inspect, e.g., the final epoch's wR2
after a run:

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
- `measurements` — numeric values such as wR2, R_obs, or diffraction loss.

See the [observability API](api/observability.md) for the event classes.
