"""Load an Arrival sim dataset through Isaac-GR00T's own LeRobotEpisodeLoader, with one of our modality configs.

The offline check (scripts/check_dataset_loader.py) compares a dataset with GR00T's loader source; this one runs the
loader. Run it with an Isaac-GR00T interpreter (the checkout the fine-tunes used is dac54b3), not the arrival venv:

    <gr00t>/.venv/bin/python scripts/check_gr00t_loader.py configs/gr00t/arrival_panda_config.py \
        out/datasets/hero_kitchen_sim

Every episode is read: decoded video frames (H x W x 3, uint8), the state and action columns the config names
(finite), and the instruction. Prints what came out and exits 0 when nothing is wrong, 1 otherwise. GR00T's video
decoder (torchcodec 0.8) needs an FFmpeg 4 to 7 runtime on the machine.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


def check(config: Path, dataset: Path) -> list[str]:
    import numpy as np

    sys.path.append(str(config.resolve().parent))
    importlib.import_module(config.stem)  # registers NEW_EMBODIMENT
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag

    cfg = MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value]
    print("config:", config.name, {k: list(v.modality_keys) for k, v in cfg.items()})
    loader = LeRobotEpisodeLoader(dataset, cfg)
    lengths = loader.get_episode_lengths()
    print("episodes:", len(loader), "lengths:", list(lengths))
    problems = []
    for i in range(len(loader)):
        df = loader[i]
        if i == 0:
            for c in df.columns:
                v = df[c].iloc[0]
                print(f"   {c}:", ("str", v[:40]) if isinstance(v, str) else (np.asarray(v).shape, str(np.asarray(v).dtype)))
        if len(df) != lengths[i]:
            problems.append(f"episode {i}: {len(df)} rows, its length says {lengths[i]}")
        for key in cfg["video"].modality_keys:
            col = next((c for c in df.columns if c.startswith("video") and c.endswith(key)), None)
            frame = None if col is None else np.asarray(df[col].iloc[0])
            if frame is None or frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                problems.append(f"episode {i}: video {key}: {None if frame is None else (frame.shape, frame.dtype)}")
        for part in ("state", "action"):
            for key in cfg[part].modality_keys:
                col = next((c for c in df.columns if c.startswith(part) and c.endswith(key)), None)
                if col is None:
                    problems.append(f"episode {i}: no {part}.{key} column")
                elif not np.all(np.isfinite(np.stack(df[col].to_numpy()).astype(float))):
                    problems.append(f"episode {i}: {col} is not finite")
        lang = [c for c in df.columns if c.startswith("language")]
        if not lang or not all(isinstance(v, str) and v for v in df[lang[0]]):
            problems.append(f"episode {i}: no instruction")
    return problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("config", type=Path, help="modality config, e.g. configs/gr00t/arrival_panda_config.py")
    p.add_argument("dataset_dir", type=Path)
    a = p.parse_args(argv)
    problems = check(a.config, a.dataset_dir)
    print("problems:", problems or "none")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
