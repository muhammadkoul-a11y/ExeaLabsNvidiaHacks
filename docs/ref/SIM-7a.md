# SIM-7a: appearance variation stage

Issue #70 (SIM-7, offline part). Code: `arrival/render/variation.py`, tests: `tests/test_variation.py`.

## Contract

- Input: a finished, validated dataset with mp4s. Output: a new directory (copy); the source is never modified.
- Selection: per episode, the round(fraction x length) frames with the smallest sha256(episode:frame:seed).
- Backends: `cosmos` (Cosmos 3 Generator via the Nebius vLLM-Omni Videos API: multipart `POST /v1/videos`, poll the job, download `/content`; basic V2V is the default, with derived edge transfer opt-in) and `local_stub` (seeded jitter + blur, geometry fixed). `COSMOS_URL` is the endpoint base URL and `COSMOS_API_KEY` is optional bearer auth.
- Acceptance per frame: edge-map correlation >= 0.6 and, with a mask, normalised masked change <= 0.35. Otherwise `rejected`, frame stays original. Backend errors give `failed`; after 3 consecutive failures the remaining requests are `skipped` (still `failed`, with `skipped: true`).
- Sidecar `meta/arrival_variation.jsonl`: original_id, variant_id, backend, model, version, parameters, latency_ms, status, reason, edge_corr, mask_change.
- Label `arrival_appearance_variation`: `cosmos` only if a cosmos frame was accepted; `local_stub` if only the stub produced frames; else `none`. The summary records backend call time separately from the added stage wall time (copy, calls and video writes). USD stays null without endpoint billing data.

## Operational constraints

- This client targets the **Cosmos 3 Generator vLLM-Omni Videos API** exposed by the Nebius [endpoint-cosmos3-generator template](https://github.com/nebius/serverless-ai-cookbook/tree/main/templates/endpoint-cosmos3-generator). It does not use the older Cosmos Transfer 2.5 NIM `/v1/infer` request schema. The request uses a source MP4 as the video reference; derived edge-transfer hints are optional and disabled by default.
- The current Cosmos3 480-tier client accepts requested frame counts from 5 through 300 and relies on the model server for internal temporal alignment. It pads only clips shorter than 5 by repeating the final frame, then trims that padding from the result. Clips above 300 frames are rejected rather than silently truncated; chunking is not implemented. Output is resized back to the dataset's original frame dimensions before geometry checks.
- By default, the request uploads the episode as `input_reference` and uses basic video-to-video conditioning with the documented first-two-latent-frame anchors (`condition_frame_indexes_vision: [0, 1]`). NVIDIA's current vLLM-Omni recipe lists a transfer-hint resolution-parsing issue for NPU builds; because the team's exact endpoint and build are unknown, basic V2V remains the conservative default. Set `COSMOS_TRANSFER_ENABLED=true` only after confirming edge-transfer mode works on the team's exact endpoint; when enabled, the client requests a derived edge control. The existing per-frame edge-correlation check remains the acceptance gate.
- `COSMOS_URL` must be the endpoint base URL from the Nebius console; the client appends `/v1/videos`. The repository does not contain the team's endpoint. The current code matches the Nebius template's asynchronous job protocol, not the separate synchronous `/v1/infer` NIM protocol. Hosted endpoint authentication may require `COSMOS_API_KEY`.
- The current integration has **not** been tested against a live endpoint. NVIDIA's current [Cosmos3-Nano vLLM-Omni recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/cosmos3/Cosmos3-Nano.md) lists a resolution-parsing issue for Transfer V2V with control hints on NPU builds. The Nebius template is GPU-oriented, so the basic-V2V default is a conservative choice until the team's exact endpoint and build are tested. Use the local mock only for protocol tests, not as proof that credentials, deployment settings, or model output quality are correct.
- Test collection requires the project dependencies in `pyproject.toml`, including `pyarrow`. Run `python -m pip install -e ".[dev,sim]"` before `python -m pytest -q tests/test_variation.py` in a network-enabled development environment.


## Live endpoint smoke test

A live request is intentionally excluded from pull-request CI. To verify a deployed endpoint, add the repository Actions secrets `COSMOS_URL` (the HTTPS endpoint base URL) and, only if required by the endpoint, `COSMOS_API_KEY`. Then manually run **Actions → Cosmos live endpoint smoke test → Run workflow**. This sends one small synthetic 25-frame clip and may incur inference charges. It reports endpoint completion and a first-frame edge-correlation check; it does not validate real robot-scene geometry. Never put credentials in the repository or PR comments.
