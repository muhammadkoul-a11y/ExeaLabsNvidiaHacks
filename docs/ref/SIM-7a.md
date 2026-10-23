# SIM-7a: appearance variation stage

Issue #70 (SIM-7, offline part). Code: `arrival/render/variation.py`, tests: `tests/test_variation.py`.

## Contract

- Input: a finished, validated dataset with mp4s. Output: a new directory (copy); the source is never modified.
- Selection: per episode, the round(fraction x length) frames with the smallest sha256(episode:frame:seed).
- Backends: `cosmos` (POST a JSON video request to NVIDIA Cosmos Transfer 2.5 NIM at `COSMOS_URL`, with a matching edge-control video; decode `b64_video`) and `local_stub` (seeded jitter + blur, geometry fixed). `COSMOS_API_KEY` is optional and sent as a bearer token for hosted endpoints.
- Acceptance per frame: edge-map correlation >= 0.6 and, with a mask, normalised masked change <= 0.35. Otherwise `rejected`, frame stays original. Backend errors give `failed`; after 3 consecutive failures the remaining requests are `skipped` (still `failed`, with `skipped: true`).
- Sidecar `meta/arrival_variation.jsonl`: original_id, variant_id, backend, model, version, parameters, latency_ms, status, reason, edge_corr, mask_change.
- Label `arrival_appearance_variation`: `cosmos` only if a cosmos frame was accepted; `local_stub` if only the stub produced frames; else `none`. The summary records backend call time separately from the added stage wall time (copy, calls and video writes). USD stays null without endpoint billing data.

## Operational constraints

- Cosmos Transfer 2.5 NIM is video-based (not a per-frame PNG API) and documents input videos of 93–480 frames. Shorter episodes must be padded for inference and trimmed back to their original length; longer episodes need chunking before they can be sent.
- `COSMOS_URL` must include the deployed endpoint path, typically `/v1/infer`. A local NIM deployment may not require `COSMOS_API_KEY`; hosted endpoints may require one.
- The current integration has not been tested against a live NVIDIA endpoint. Use a local mock only for request/response contract tests, not as proof that credentials, deployment settings, or model output quality are correct.
