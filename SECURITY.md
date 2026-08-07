# Device and repository safety policy

The mandatory project path is non-destructive stock Android.

## Never execute by default

- bootloader unlock;
- `fastboot flash` / `fastboot erase`;
- partition writes or filesystem formatting;
- AVB/vbmeta bypass or modification;
- FRP/security bypass tooling;
- writes to modem/NV/persist/calibration partitions;
- factory reset/wipe as a performance experiment.

A destructive experiment is allowed only if **both** are true:

1. `PROJECT_STATE.yaml` records explicit current user authorization; and
2. `RECOVERY_READY` is PASS with exact variant, partition map, legitimate stock firmware/restore source, boot/recovery entry method and reviewed restore procedure.

The `host.device_lab` wrapper implements conservative allowlisted operations as defense in depth. It is not a sandbox against an external agent that ignores repository policy and invokes system tools directly; `AGENTS.md` remains binding.

## Safe primary operations

Read-only ADB probes, pushing test binaries/models to ordinary temp/user storage, running/stopping test processes, pulling artifacts and ADB TCP forwarding are expected on the primary path.

If the device becomes unstable or thermally constrained, stop the workload, preserve evidence and recover with a normal reboot/cool-down rather than destructive system changes.
