# H40M/1 — experimental flash-layout format

H40M is a project-local deterministic storage artifact for testing explicit placement and flash locality. It does not replace Safetensors/GGUF generally.

The canonical schema is `schemas/h40m_manifest.schema.json`.

Each tensor entry records at minimum:
- logical `name` and semantic `role`;
- `layer` and `expert_id` where applicable;
- `shape`, `dtype`, `quant_type`, physical `layout`;
- required `alignment`;
- physical `file_id`, `offset`, `length`;
- placement: `resident`, `cache`, `stream`, or `token_lookup`;
- content checksum and source-tensor provenance.

## M01 manifest stage

M01 creates a deterministic H40M/1 manifest from the official checkpoint inventory before payload bytes are repacked. Tensor ranges are assigned in a virtual `h40m/model.h40m` arena with fixed alignment and source-shard provenance. Until M02 materializes bytes, tensor `sha256` values use `checksum_kind: source_range_id_sha256`, a deterministic hash of tensor identity, source shard/checksum and planned H40M range. M02 must replace or supplement those with real content checksums when it writes payload files.

## Design rules
- Converter output is deterministic for the same pinned source checkpoint/config.
- Expert payload is stored in one/few large arenas, not thousands of tiny files.
- Initial alignment must come from D04 measurements, not folklore.
- Repacking cannot silently re-quantize or alter MXFP4 payload.
- Versioned physical layouts may change while logical tensor identity/provenance remain stable.
- The runtime validates manifest bounds before reading.

Later layouts may add co-access groups/prefetch hints, but hints are optimizations and never replace exact router selection.
