# USB service guide — A00 to A03

## Goal

Expose the phone as a deterministic local inference node without adding unnecessary protocol/runtime weight to the 4 GB device.

## A00 — minimal device service

Initial device endpoints should be small and internal, for example:

```text
GET  /health
GET  /metrics
POST /infer
```

`/health` reports runtime/model readiness, not merely that the HTTP process exists. `/metrics` may expose bounded runtime counters needed for host aggregation.

The device service should accept already-rendered token/prompt data or a deliberately small request contract until A01 is implemented.

## A01 — laptop Harmony/Responses gateway

The laptop owns:

- official Harmony render/parse;
- tool/reasoning channel handling;
- public OpenAI-style request adaptation;
- request validation;
- streaming aggregation.

This preserves OpenAI semantics while keeping the phone binary smaller and easier to profile.

## A02 — physical USB path

Primary stock-Android path:

```bash
hot40 forward --host-port 18080 --device-port 8080
```

Conceptually:

```text
client -> laptop gateway -> localhost forwarded port -> ADB USB -> device service
```

Automate forward re-establishment after device reconnect/reboot. Detect serial ambiguity rather than forwarding to an arbitrary device.

## A03 — real client demo

Demonstrate one end-to-end request through the exact final transport/protocol stack. Record:

- client request;
- Harmony/model provenance;
- device health before/after;
- response/tokens;
- latency/metrics;
- USB forwarding state.

A Wi-Fi-only demo does not satisfy the USB requirement.

## Service restart rule

Any code/config change to the gateway/device runtime requires a clean service restart before verification. Final F01/F03 explicitly start from a clean runtime/service state.
