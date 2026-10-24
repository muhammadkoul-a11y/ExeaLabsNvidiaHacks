# Access

The implementing agent does not open these accounts and does not block on them. A person has to log in. Model ids change; recheck the cookbook pages before wiring a client.

## Open these

| What | URL | Why |
|---|---|---|
| Nebius console | <https://console.nebius.com/> | Project, quotas, Object Storage, Serverless Jobs |
| Nebius Builder Program | <https://dev.nebius.com/builders> | Credits, including Token Factory credit after email verification |
| Token Factory | <https://tokenfactory.nebius.com/> | Nemotron calls |
| Token Factory docs | <https://docs.tokenfactory.nebius.com/> | Base URL and model ids. The code default is `https://api.tokenfactory.nebius.com/v1` and must be checked against this doc |
| Nemotron cookbook | <https://github.com/nebius/token-factory-cookbook/tree/main/models/nemotron> | Which Nemotron 3 ids exist |
| Nemotron 3 Nano Omni note | <https://github.com/nebius/token-factory-cookbook/blob/main/models/nemotron/nemotron3-nano-omni.md> | Scan-reading model |
| Serverless Jobs quickstart | <https://docs.nebius.com/serverless/quickstart/jobs> | Fan-out later |
| Serverless cookbook | <https://github.com/nebius/serverless-ai-cookbook> | Job and endpoint recipes |
| Cosmos 3 Generator template | <https://github.com/nebius/serverless-ai-cookbook/tree/main/templates/endpoint-cosmos3-generator> | Target endpoint for video-to-video appearance transfer |
| NVIDIA Cosmos 3 Generator transfer guide | <https://github.com/NVIDIA/cosmos/tree/main/cookbooks/cosmos3/generator/transfer> | Edge/depth/segmentation controls and transfer parameters |
| LeRobot fine-tune job | <https://github.com/nebius/serverless-ai-cookbook/tree/main/robotics/lerobot-finetune-job> | Fallback policy if GR00T rejects the embodiment |
| Isaac GR00T | <https://github.com/NVIDIA/Isaac-GR00T> | N1.7 code, Apache 2.0 |
| GR00T hardware note | <https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md> | Fine-tune wants 40 GB+ VRAM |
| GR00T data prep | <https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md> | `meta/modality.json` |
| GR00T weights | <https://huggingface.co/nvidia/GR00T-N1.7-3B> | Not gated itself. Its Cosmos-Reason2-2B VLM backbone is gated on Hugging Face: accept that license with the same account before loading |
| NVIDIA build | <https://build.nvidia.com/> | Only if a model is hosted there and not on Token Factory |
| RoboCasa | <https://github.com/robocasa/robocasa> | Layout format and atomic tasks |
| MimicGen guide | <https://github.com/robocasa/robocasa/blob/main/docs/use_cases/mimicgen.md> | `PnPCounterToSink` template path |
| Layout example | <https://github.com/robocasa/robocasa/blob/main/robocasa/models/assets/scenes/kitchen_layouts/train/layout011.yaml> | Full layout the subset emitter is aiming at |
| Tavily Search API | <https://tavily.com/> | Appliance manual and safety clearance research ($3,000 sponsor prize) |

## Cosmos choice for appearance variation

The repository's SIM-7 scope names the **Cosmos 3 Generator endpoint**, so this branch now targets the Nebius [Cosmos 3 Generator template](https://github.com/nebius/serverless-ai-cookbook/tree/main/templates/endpoint-cosmos3-generator), not the separate Cosmos Transfer 2.5 NIM API. The template serves Cosmos3-Nano through vLLM-Omni. Its video API is asynchronous: `POST /v1/videos`, poll `GET /v1/videos/{id}`, then download `GET /v1/videos/{id}/content`. Basic V2V is the default until the team's exact deployment is verified. NVIDIA's current vLLM-Omni recipe lists a transfer-hint resolution-parsing bug for NPU builds; the Nebius template is GPU-oriented, so this is a conservative default rather than a confirmed limitation of the team's endpoint. The client can opt into derived-edge transfer with `COSMOS_TRANSFER_ENABLED=true` after the exact endpoint is tested; do not send the older Transfer 2.5 `/v1/infer` JSON contract to this endpoint. See NVIDIA's [Cosmos3 transfer cookbook](https://github.com/NVIDIA/cosmos/tree/main/cookbooks/cosmos3/generator/transfer) and Nebius template's endpoint README.

Configure `COSMOS_URL` as the **base HTTPS endpoint URL** from the Nebius console (not a `/v1/infer` URL). Configure `COSMOS_API_KEY` with the endpoint's bearer token only if authentication is enabled. Do not commit credentials. The template deploys Cosmos3-Nano on an H100-class preset; confirm the team's actual endpoint URL, auth setting, quota, and model readiness before running paid inference. No live endpoint is recorded in this repository, so live inference remains a separate validation step.

## What to request

1. Nebius account and a project id.
2. Token Factory API key. Copy `.env.example` to an untracked `.env` and set `TOKEN_FACTORY_API_KEY`.
3. Serverless Jobs quota for at least one GPU preset. The jobs quickstart uses platform `gpu-l40s-a` and preset `1gpu-8vcpu-32gb`. Confirm those ids in the console; do not assume they are still current.
4. Hugging Face access: `nvidia/GR00T-N1.7-3B` is not gated, but the Cosmos-Reason2-2B backbone it loads is; accept that model's license first. Record whether `NEW_EMBODIMENT` is the supported path for a 7-DoF arm. It was, in the public README, via `--embodiment-tag NEW_EMBODIMENT` and a modality config. Recheck before writing that config.
5. An Object Storage bucket, S3-compatible, only when artifacts leave the laptop.
6. Tavily Search API key. Put it in an untracked `.env` as `TAVILY_API_KEY` (optional; offline cached fixture is checked in).

## What not to do

- Do not put keys in the repo, the README, or the dataset.
- Do not spend credits on a fine-tune of the placeholder dataset.
- Scanner apps and RoomPlan category lists were unverified in the spec. The normalizer in this repo fixes the subset it accepts. The normalizer now handles Apple's keyed-enum encoding and is tested against 2 public real exports (`tests/fixtures/roomplan/`); our own capture can replace `scans/hero_kitchen/roomplan.json` later. Polycam and Scaniverse exports are not accepted until a converter is written.
