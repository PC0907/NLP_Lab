"""Diagnostic: did the all_tokens extraction actually save 2D (multi-token)
activations? Run on ONE extracted domain's activations dir."""
import sys, glob, numpy as np
from pathlib import Path

acts_dir = sys.argv[1] if len(sys.argv) > 1 else "."
files = [f for f in glob.glob(f"{acts_dir}/*.npz")]
if not files:
    print("No .npz files found in", acts_dir); sys.exit(1)

print(f"Inspecting {files[0]}")
with np.load(files[0]) as z:
    keys = list(z.keys())
    print(f"  {len(keys)} keys")
    ndims = {}
    spanlens = []
    for k in keys[:5000]:
        a = z[k]
        ndims[a.ndim] = ndims.get(a.ndim, 0) + 1
        if a.ndim == 2:
            spanlens.append(a.shape[0])
    print("  ndim distribution:", ndims)
    if spanlens:
        import statistics
        print(f"  2D arrays: span lengths mean={statistics.mean(spanlens):.2f} "
              f"median={statistics.median(spanlens)} max={max(spanlens)} "
              f"min={min(spanlens)}")
        print(f"  multi-token (>1) fraction among 2D: "
              f"{sum(1 for s in spanlens if s>1)}/{len(spanlens)}")
    else:
        print("  *** NO 2D ARRAYS *** -> extraction saved 1D (last_token/mean),")
        print("      NOT all_tokens. last==mean==max is therefore expected.")
    # show a couple example shapes
    for k in keys[:3]:
        print("   example:", k, "->", z[k].shape)