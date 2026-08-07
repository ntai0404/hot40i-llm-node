# Device safety and rollback gates

## Phase 1: stock Android research
Allowed: read-only ADB inspection, `adb push/pull`, launching native test processes under `/data/local/tmp`, ADB port forwarding.

Not allowed: bootloader unlock, partition writes, erase, vbmeta changes, FRP bypass, modifying modem/NV/persist/calibration data.

## Before any destructive phase
Require all of the following:
1. Exact device/variant identification.
2. Partition map captured.
3. Verified stock firmware or equivalent restore image.
4. Verified boot/download/recovery entry method.
5. A documented restore procedure that has been reviewed independently.
6. Explicit user authorization for the destructive phase.

A performance hypothesis is never sufficient justification for flashing.
