# Engine

The refinement engine: compiled per-orientation geometry plans, the forward simulation spine, the
objective, the hard molecular constraints and soft penalties composed into it, and the (deliberately
quarantined) imperative optimization loop.

Scientific composition is done with typed Python values, not config: build a problem with
`build_refinement_problem` and add hard constraints (e.g. `with_hydrogen_riding`) or soft penalties.

::: diffBloch.engine.plan

::: diffBloch.engine.forward

::: diffBloch.engine.constraints

::: diffBloch.engine.penalties

::: diffBloch.engine.losses

::: diffBloch.engine.refine
