"""Shrink an RL4COTrainer checkpoint to what MaxSimSplitter needs.

A full checkpoint stores the frozen sentence encoder twice (policy.lm.* and the rollout baseline's
copy) plus trainer state: about 7 GB with bge-m3. MaxSimSplitter only loads `policy.*` weights and
takes the encoder from `--embedding-model`, so this keeps `policy.*` without `policy.lm.*`
(a few MB to tens of MB).

The output keeps the `epoch=*-step=*.ckpt` name, so passing its folder as --splitter-checkpoint works.

Example:
  python benchmarks/dialect/slim_rl_checkpoint.py --src /content/rl_ckpt_bge-m3 \
    --out-dir /content/drive/MyDrive/dialects/rl_ckpt_bge-m3
"""

import argparse
import glob
import os
import re
import sys

import torch

# Full checkpoints pickle trainer objects (MaxSimEnv, ...) by their top-level module names.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "rl-training-algorithm"))


def pick(src: str) -> str:
    """The best checkpoint (ModelCheckpoint save_top_k=1 keeps it as epoch=*-step=*.ckpt), else last.ckpt."""
    if os.path.isfile(src):
        return src
    pat = re.compile(r"epoch=(\d+)-step=(\d+)\.ckpt$")
    best = sorted((f for f in glob.glob(os.path.join(src, "*.ckpt")) if pat.search(f)),
                  key=lambda f: tuple(int(x) for x in pat.search(f).groups()))
    if best:
        return best[-1]
    last = os.path.join(src, "last.ckpt")
    if os.path.isfile(last):
        return last
    raise FileNotFoundError(f"no checkpoint in {src}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Checkpoint file or RL4COTrainer --checkpoint_dir")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    path = pick(args.src)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k: v for k, v in ckpt["state_dict"].items()
          if k.startswith("policy.") and not k.startswith("policy.lm.")}
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, os.path.basename(path))
    torch.save({"state_dict": sd, "epoch": ckpt.get("epoch"), "global_step": ckpt.get("global_step"),
                "source": path}, out)
    mb = os.path.getsize(out) / 2**20
    print(f"{path} (epoch {ckpt.get('epoch')}) -> {out}: {len(sd)} tensors, {mb:.1f} MB")


if __name__ == "__main__":
    main()
