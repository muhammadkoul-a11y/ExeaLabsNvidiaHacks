# Running the GR00T steps on a GPU box

A recipe that worked on a molab notebook with one NVIDIA RTX PRO 6000 Blackwell (96 GB), 20 CPUs, 160 GB RAM, driver 595, Oct 21 2026. Nothing here needs a Nebius account. It is a record of what ran, not a guarantee for other machines.

## Two Python environments

Isaac-GR00T pins Python 3.12, `torch 2.9.0` (CUDA 12.8 wheels) and `flash-attn 2.8.3`; keep it apart from the sim.

```bash
git clone --depth 1 https://github.com/NVIDIA/Isaac-GR00T.git && cd Isaac-GR00T && uv sync --python 3.12
uv venv --python 3.12 simenv && uv pip install --python simenv/bin/python -e "<arrival repo>[sim]"
```

Trap: a shallow clone leaves `scripts/deployment/dgpu/wheels/torchcodec-0.8.0-cp312-cp312-linux_aarch64.whl` as a Git LFS pointer, and `uv sync` fails with "Invalid zip file structure" even on x86_64. Either `git lfs pull`, or fetch the 512 KB file from `media.githubusercontent.com` and check its sha256 against the `oid` in the pointer.

Weights need a Hugging Face read token (`HF_TOKEN`, set in the environment, never in a file) with the `nvidia/Cosmos-Reason2-2B` license accepted: `nvidia/GR00T-N1.7-3B` (6.5 GB) and `nvidia/Cosmos-Reason2-2B` (4.6 GB).

## Data

Headless rendering worked with `MUJOCO_GL=egl` (`osmesa` did not). One episode takes about 15 s at 224 px, so shard it:

```bash
for i in $(seq 0 15); do MUJOCO_GL=egl python -m arrival sim --episodes 320 --people 2 --res 224 --seed 7 --shard $i/16 --out data & done; wait
python -m arrival merge-shards data/shards --out merged && python -m arrival validate merged/datasets/hero_kitchen_sim
```

320 episodes (56,200 frames) took about 23 minutes on 16 processes. Do not start the same shard twice: both runs write into the same folder. Do not run the shards while a fine-tune trains; the EGL renderers and the training data loader slowed it from 1.4 to 0.6 steps/s.

## Stats, fine-tune, serve

```bash
python gr00t/data/stats.py --dataset-path merged/datasets/hero_kitchen_sim --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path configs/gr00t/arrival_panda_config.py        # stats.json, relative_stats.json: all finite
python gr00t/experiment/launch_finetune.py --base-model-path <GR00T-N1.7-3B dir> --dataset-path merged/datasets/hero_kitchen_sim \
    --embodiment-tag NEW_EMBODIMENT --modality-config-path configs/gr00t/arrival_panda_config.py --num-gpus 1 \
    --output-dir ckpt --max-steps 100 --save-steps 100 --global-batch-size 32 --dataloader-num-workers 4
```

100 steps at batch 32 took 58 s (about 2.3 steps/s, 41 GB of GPU memory). At batch 64 with `--tune-visual` it used 92.7 GB, which leaves no room to serve a checkpoint on the same GPU. The checkpoint folder was 16 GB with the optimizer state. No loss line showed up in the stdout log of the smoke run; use W&B or the trainer state if you need a curve.

A 6,000-step run (`--save-steps 1000 --save-total-limit 3 --global-batch-size 32 --tune-visual --state-dropout-prob 0.1`) took 77 minutes at about 1.4 steps/s and 66 GB of GPU memory; the 16 GB checkpoints add up (three kept, plus the final model, was 93 GB).

Do not score a checkpoint on the same GPU while training: serving it and running the eval cut training from 1.4 steps/s to 0.15. Score after training, or on a second GPU. An eval of 24 episodes takes about 15 minutes and uses many CPU cores.

Then serve with `gr00t_shim` (see [POL-3.md](POL-3.md)) and run `python -m arrival eval --policy vla --checkpoint-id <label>`.

## Process notes

Start long jobs with `setsid nohup ... &`. In the notebook, `ps` could not see them; counting `/proc/*/cmdline` did.
