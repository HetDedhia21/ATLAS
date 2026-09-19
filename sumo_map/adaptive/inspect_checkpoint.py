"""
inspect_checkpoint.py

Converts stage4_checkpoint.pt into a plain-text JSON file you can open
and read in VS Code (or any editor) - no torch needed to view the
output, only to run this converter once.

Produces TWO files:
  - stage4_checkpoint_summary.json  : shapes + basic stats per layer
                                       (mean/std/min/max) - small, quick
                                       to skim, good for "does this look
                                       like a trained network" sanity
                                       checks.
  - stage4_checkpoint_full.json     : every actual weight value, in full.
                                       Bigger, but still just plain
                                       numbers - useful if you actually
                                       want to inspect specific weights.

Run from wherever the .pt file lives, e.g.:
    python inspect_checkpoint.py stage4_checkpoint.pt
"""

import sys
import json
import torch


def tensor_stats(t):
    flat = t.flatten()
    return {
        "shape": list(t.shape),
        "num_params": t.numel(),
        "mean": flat.mean().item(),
        "std": flat.std().item() if flat.numel() > 1 else 0.0,
        "min": flat.min().item(),
        "max": flat.max().item(),
    }


def main():
    if len(sys.argv) < 2:
        path = "stage4_checkpoint.pt"
        print(f"No path given, defaulting to ./{path}")
    else:
        path = sys.argv[1]

    ckpt = torch.load(path, map_location="cpu")

    summary = {}
    full = {}

    for junction_id, state_dict in ckpt.items():
        summary[junction_id] = {}
        full[junction_id] = {}
        for param_name, tensor in state_dict.items():
            summary[junction_id][param_name] = tensor_stats(tensor)
            full[junction_id][param_name] = {
                "shape": list(tensor.shape),
                "values": tensor.tolist(),  # nested list, matches tensor shape
            }

    summary_path = path.replace(".pt", "_summary.json")
    full_path = path.replace(".pt", "_full.json")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(full_path, "w") as f:
        json.dump(full, f, indent=2)

    print(f"Wrote {summary_path}  (shapes + stats, small, read this first)")
    print(f"Wrote {full_path}  (every weight value, larger)")


if __name__ == "__main__":
    main()