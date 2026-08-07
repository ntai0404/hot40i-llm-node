# Decision and stop gates

## Mandatory milestones

See `roadmap/gates.yaml`. The agent may not skip numerical parity because a model happens to output readable text.

## Performance classification

- proof: correct official model service exists;
- P0: sustained decode >=0.25 tok/s;
- P1: >=0.5 tok/s;
- P2: >=1 tok/s;
- stretch: >=2 tok/s.

The final deployment gate does not require a fabricated performance threshold. It requires a correct, bounded, measured service and a final performance classification.

## Optimization stop rule

After all mandatory O00–O07 experiments, retain only optimizations that improve end-to-end behavior without violating correctness/memory. Negative results remain in the report. Do not keep searching indefinitely for an arbitrary headline.

## OS optimization decision

OS work is optional. Activate only if measurements show Android overhead is a material blocker after the inference/storage runtime is proven. Destructive work additionally requires explicit authorization + `RECOVERY_READY`.
