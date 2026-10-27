# GR00T's own loader on today's datasets (Oct 27 2026)

Do the datasets `arrival sim` writes today load in Isaac-GR00T? The offline check (`scripts/check_dataset_loader.py`) compares them with GR00T's loader source; this run used the loader itself.

## Setup

- Isaac-GR00T **`dac54b3`** (the commit the fine-tunes used), installed with `uv sync --python 3.12`: torch 2.9.0+cu128 and torchcodec 0.8.0, run on **CPU** (a 4-core Linux Codespace, no GPU). GR00T's video decoder needs an FFmpeg 4–7 runtime. Ubuntu's `ffmpeg` 6.1.1 package worked; with no FFmpeg, every video read fails with `Could not load libtorchcodec`.
- Arrival at `03690dd` (main, Oct 27). Three datasets from `hero_kitchen`, 3 episodes each, `--people 1 --res 224 --seed 3` (224 px is the fine-tune's resolution):

| Name | `arrival sim` flags | Episodes (frames) | Videos |
|---|---|---|---|
| arm | `--families counter_to_sink,open_drawer` | 3 (176, 146, 176) | ego_view, wrist |
| mobile | `--families counter_to_counter,counter_to_sink --base mobile` | 3 (449, 238, 355) | ego_view, wrist |
| three cameras | `--families counter_to_sink,open_drawer --exocentric` | 3 (176, 146, 176) | ego_view, wrist, overview |

All three pass `arrival validate`. Its warnings are the expected ones: the parked base's action dims are constant on a fixed base, and so is the mobile base's yaw when it only drives along a counter.

## What GR00T read

`<gr00t>/.venv/bin/python scripts/check_gr00t_loader.py <config> <dataset>` builds GR00T's `LeRobotEpisodeLoader` with the dataset's modality config and reads **every episode**:

| Dataset | Config | Columns GR00T returned (first frame) | Problems |
|---|---|---|---|
| arm | `arrival_panda_config.py` | `video.ego_view`, `video.wrist` (224, 224, 3) uint8; `state` / `action.single_arm` (7,), `state` / `action.gripper` (1,) float32; `language.annotation.human.task_description` | **none** |
| mobile | `arrival_panda_mobile_config.py` | the same, plus `state.base` and `action.base` (3,) float32 | **none** |
| three cameras | `arrival_panda_multiview_config.py` | the same as arm, plus `video.overview` (224, 224, 3) uint8 | **none** |

For each episode the script checks that the row count matches the episode length, every video frame decodes to H×W×3 uint8, every state and action column the config names exists and is finite, and every row has an instruction.

GR00T's `gr00t/data/stats.py --embodiment-tag NEW_EMBODIMENT --modality-config-path <config>` also ran on all three. It exited 0 and wrote `meta/stats.json` and `meta/relative_stats.json`, which fine-tuning reads.

## What this does not show

No fine-tune or inference ran (no GPU). This is the data path only, the step the README's status table marked *"⚠️ Not yet loaded by GR00T itself since the latest fixes"*. The datasets themselves are not committed: they are reproducible from the commands above, and a 3-episode 224 px dataset is about 10 MB of video.
