# Third-party source policy

This repository does **not** vendor the large upstream inference frameworks in the handoff archive. Instead:

- `manifest.yaml` records why each upstream matters;
- `LOCK.yaml` records the exact immutable Git commit after R00 resolves it;
- `scripts/resolve_upstreams.py` resolves Git refs;
- `scripts/sync_third_party.py --require-locked` fetches/checks out the locked commit under ignored `third_party/src/`.

Before copying/adapting code, record the upstream license and preserve notices required by that license. The repository's MIT license applies only to project-authored code; it does not relicense upstream code.

Never base a reproducible benchmark on an unlocked `main` checkout.
