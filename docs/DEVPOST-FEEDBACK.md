# Devpost feedback draft: Token Factory, AI Cloud and the NVIDIA models

Paste the three sections into the Devpost feedback field. The NVIDIA section is from runs we did. The Nebius sections have two parts: what we saw writing the code, and lines marked **[after the real call]** that must be filled in with what actually happened. Delete any bracketed line you can't back with a run. `python scripts/smoke_live_apis.py` makes the Token Factory and Tavily calls and writes the model ids, latency and source of each to `docs/results/live-apis-<date>.json`; quote that file, not memory.

## NVIDIA models

**GR00T N1.7-3B (fine-tune).** We fine-tuned it on a simulated kitchen built from a LiDAR room scan: a 7-joint arm with a gripper, two camera views, an 8-number state and a text instruction, written as a LeRobot v2 dataset. The checkpoint is at https://huggingface.co/Ish1105/arrival-gr00t-n17-run4.
- What worked: a modality config plus the LeRobot v2 format was enough to load our data. A 100-step smoke run took 58 seconds, and a full run (4,000 steps at batch 32, visual tuning on) took about 51 minutes on one 96 GB GPU. Clean-data fine-tuning scored 26 of 48 in our sim.
- What hurt: we had to pin an Isaac-GR00T commit (`dac54b3`) to get a working setup. The 3B model is about 12 GB in F32, and inference needs 16 GB or more of GPU memory. It was easy to score on the training GPU by accident, and that gave misleading numbers, so we now use one policy server per client and paired trials.
- What we learned: open-loop error on expert frames was 0.3 cm at the next step, but closed loop the gripper closed 2 to 8 cm from the mug, so the failure was compounding drift, not perception. DART-style noisy demos helped grasping but taught the policy to under-pull drawers until we filtered out 102 marginal pulls (39 of 48, against 22 of 48 unfiltered).
- Wish: a documented, versioned recipe for a new embodiment (state and action layout, normalisation, camera names), and a published example of a closed-loop sim evaluation.
- Limits of our claim: simulation only, one room, two tasks, one seed, one run per model.

**Cosmos-Reason2 (the GR00T backbone) and Cosmos.** [Say what you actually used. We ran the fine-tune on GR00T's Cosmos-Reason2 backbone and have an appearance-variation stage with a local stub only. If you did not call a Cosmos endpoint, say so.]

**Nemotron.** The planner client (`arrival/models/token_factory.py`) and the VLM planner in sim collection call Nemotron through Token Factory: `nvidia/nemotron-3-super-120b-a12b` plans each task and `nvidia/Nemotron-3_5-Lightning` writes instruction wordings. Measured on Oct 26 (`docs/results/planner-benchmark-2026-10-26.json`): about 2.6 s median per plan. On 15 indirect requests Super picked the right task 13 times against 6 for our keyword parser, and on 54 Lightning-written wordings 52 against 47. Its misses were telling: "cupboard" twice became a drawer. The JSON format held in every reply, but 7 of the first 69 plans put a fixture id in the `object` field (`"object": "fridge_1"` for "open the fridge"), which our strict check rejected and sent to the fallback. We now accept that shape and call at temperature 0. Replies also start with blank lines, so parse from the first `{`.

**Nemotron 3.5 Super VL (object detection).** `arrival/twin/detect.py` sends each posed scan frame to `nvidia/Nemotron-3.5-Super-VL` for catalog objects as a label, a pixel box and a confidence, then back-projects the box with the frame's depth. Measured on Oct 26 (`docs/results/live-detect-2026-10-26.json`): on six 128 x 96 renders of our kitchen twin it boxed the mug in all three frames that have one and listed nothing in the three empty ones, and the placed mug landed 1.2 cm from its true centre. The boxes were tight enough that two of three were rejected by our geometry check, so one view did the placing. The six frames took about 25 s in all. These were rendered frames, not a real scan.

## Nebius Token Factory

- Seen from the code: the OpenAI-compatible endpoint made the client small (the standard SDK shape, a base URL and a key). We keep model names in environment variables because we could not tell which were available to us without a key.
- Model availability: `GET /v1/models` listed 30 models for our key, 5 of them Nemotron (Nano 30B, Super 120B, Super-VL, Lightning, Ultra 550B). That list is the only reliable way we found to get exact ids. Our first setup put an NVIDIA `nvapi-` key in the model fields, because the two kinds of credentials look alike from the outside; a sentence in the quickstart separating build.nvidia.com keys from Token Factory keys and model ids would have saved that.
- No rate limit was hit across more than 150 calls in about ten minutes. **[Fill in: sign-up to first response time and the price per call from the console. Delete this line if you don't have them.]**

## Nebius AI Cloud

- Seen from the code: we wrote a Serverless AI Jobs backend for sharded simulation with an Object Storage bucket (`arrival/jobs/nebius.py`, `arrival fanout --backend nebius`). It has only run against in-process fakes. The shard design (resume markers, retry) was the harder part; mapping it onto jobs was the easy part.
- Not run on Nebius: built and tested against fakes only. We spent the time on the Token Factory integration instead.
- Do not say it ran on Nebius unless the real call happened. The fine-tune did not run on Nebius.
