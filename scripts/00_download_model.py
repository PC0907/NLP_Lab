"""Cache a model locally for the Bender compute nodes.

Run this ONCE on the Bender LOGIN NODE before submitting the extraction job.
Compute nodes do not have internet access, so the model must be cached first.

Usage:
    python scripts/00_download_model.py                    # default model
    python scripts/00_download_model.py Qwen/Qwen3.5-9B    # any other model

The weights are saved under:
    ~/.cache/huggingface/hub/models--<org>--<name>/

Time estimate: 10-30 minutes depending on network speed.
Disk space needed: roughly 2 GB per billion parameters in bf16
(~14 GB for a 7B, ~18 GB for a 9B) free in your home directory or wherever
    HF_HOME / TRANSFORMERS_CACHE points to.

If you are low on quota in ~/ you can redirect the cache:
    export HF_HOME=/path/to/large/disk
    python scripts/00_download_model.py

No HuggingFace token is needed — this model is fully public.
"""

from __future__ import annotations

import sys
import time

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


def main() -> int:
    # Takes the model id as an argument so any model can be cached with this
    # helper, not just the first one the project used:
    #   python scripts/00_download_model.py Qwen/Qwen3.5-9B
    model_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    print(f"Downloading: {model_name}")
    print("Expect 10-40 minutes depending on size and network.")
    print("=" * 60)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: transformers is not installed. Activate your venv first.")
        print("  source ~/nlp_lab/bin/activate   (A40 nodes)")
        print("  source ~/nlp_lab_a100/bin/activate  (A100 nodes)")
        return 1

    # Download tokenizer first (fast, confirms connectivity)
    print("\n[1/2] Downloading tokenizer...")
    t0 = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=False,
        )
        print(f"  Tokenizer downloaded in {time.time() - t0:.1f}s")
        print(f"  Vocab size: {tokenizer.vocab_size:,}")
    except Exception as e:
        print(f"  ERROR downloading tokenizer: {e}")
        return 1

    # Download model weights (the slow part)
    print("\n[2/2] Downloading model weights...")
    print("  Progress is shown by HuggingFace below.")
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=False,
            # Do NOT load to GPU here — we only want to cache the weights.
            # The extraction job will load to GPU with dtype=bfloat16.
            torch_dtype="auto",
            device_map="cpu",
        )
        elapsed = time.time() - t0
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n  Model downloaded in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  Parameters: {n_params/1e9:.2f}B")
        print(f"  Hidden size: {model.config.hidden_size}")
        print(f"  Num layers:  {model.config.num_hidden_layers}")
        del model  # free CPU RAM immediately
    except Exception as e:
        print(f"  ERROR downloading model: {e}")
        return 1

    print("\n" + "=" * 60)
    print("Download complete. Model is cached in:")
    print("  ~/.cache/huggingface/hub/")
    print("\nYou can now submit the extraction job:")
    print("  sbatch run_deepseek_extract.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())