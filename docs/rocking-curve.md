# Rocking curve, mosaicity, and coupling

A rotation electron-diffraction frame integrates intensity while the crystal rocks through a small
angular range. DiffBloch models this by expanding each orientation into sampled virtual tilts,
solving those tilt sub-orientations, and reducing the tilt intensities into one calculated pattern.

## Pieces

| Piece | Role |
|---|---|
| `RockingCurve` | Number of tilt samples and shared integration geometry. |
| `integrate_rocking_curve` | `Plan -> Plan` step that adds virtual tilt sub-orientations. |
| `Mosaicity` / `mosaicity` | Moving-average broadening over the tilt axis. |
| `TiltSegmentUnion` / `couple_beams` | Coupled per-segment beam unions for rocking-curve solves. |

## API example: composing rocking-curve steps

```python
from diffBloch.preprocess import integrate_rocking_curve, mosaicity, pipeline
from diffBloch.specs import Mosaicity, RockingCurve

prepare_rocking_curve = pipeline([
    integrate_rocking_curve(RockingCurve(sampling=42)),
    mosaicity(Mosaicity(window=5)),
])

# updated_plan = prepare_rocking_curve(built_plan)
```

## API shape: coupling

Coupling is an optional `Plan -> Plan` step. It should come after tilt-independent steps that need
plain orientation plans.

```python
from diffBloch.preprocess import couple_beams
from diffBloch.specs import TiltSegmentUnion

couple = couple_beams(TiltSegmentUnion(n_splits=12, g_max=2.25, sg_max=0.01))

# coupled_plan = couple(plan_with_rocking_curve_tilts)
```

The default app recipe uses the same underlying policy for faithful per-trial coupling during
orientation fitting; `couple_beams` is the explicit composable step when callers want to settle a
coupled plan themselves.
