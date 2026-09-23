#!/usr/bin/env python3
"""Smoke test: Gemma-4-E4B on one ExtractBench document.

Checks, in order, before any pipeline change:
  1. transformers can import AutoModelForMultimodalLM and load the model.
     (An earlier Gemma-4 attempt failed here: the installed transformers did
     not recognise the architecture.)
  2. Layer count and hidden size, read from config.text_config if nested.
  3. Text-only generation through the processor's chat template.
  4. The shape of the hidden states generate() returns. The pipeline expects
     hidden_states[step][layer] of shape (batch, seq, hidden). A multimodal
     model may differ, and then the pipeline's capture needs adapting.
  5. Whether the output parses as JSON with the pipeline's own parser, and
     whether its keys match the schema (the gate Llama and R1-Distill failed).

Uses the parsed-text cache via the benchmark loader, so the input text is
identical to what Qwen saw.

Usage:
  python gemma4_smoke.py --config configs/exp_fresh_alltokens.yaml --doc-index 24
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from importlib import import_module
from pathlib import Path

import torch

sys.path.insert(0, "scripts")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/exp_fresh_alltokens.yaml")
    p.add_argument("--model", default="google/gemma-4-E4B")
    p.add_argument("--doc-index", type=int, default=24,
                   help="24 = swimming table2: short, and Qwen extracts it well.")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    return p.parse_args()


def schema_paths(schema):
    defs = schema.get("$defs", {}) or schema.get("definitions", {})
    out = set()

    def resolve(node):
        for _ in range(20):
            if isinstance(node, dict) and "$ref" in node:
                node = defs.get(node["$ref"].split("/")[-1], {})
            else:
                break
        return node

    def walk(node, prefix):
        node = resolve(node)
        if not isinstance(node, dict):
            return
        for alt in node.get("anyOf", []) + node.get("oneOf", []):
            walk(alt, prefix)
        for k, v in (node.get("properties") or {}).items():
            path = f"{prefix}.{k}" if prefix else k
            out.add(path)
            walk(v, path)
        if node.get("items"):
            walk(node["items"], prefix)

    walk(schema, "")
    return out


def json_paths(obj, prefix=""):
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            out.add(path)
            out |= json_paths(v, path)
    elif isinstance(obj, list):
        for v in obj:
            out |= json_paths(v, prefix)
    return out


def main():
    a = parse_args()
    import transformers
    print(f"transformers {transformers.__version__}, torch {torch.__version__}")

    # ---- 1. import + load -------------------------------------------------
    try:
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError as e:
        print(f"FAIL at import: {e}")
        print("This transformers version does not provide AutoModelForMultimodalLM.")
        print("Do NOT upgrade the main environment. Use a separate one.")
        return 1

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForMultimodalLM.from_pretrained(
        a.model, device_map="auto", dtype=torch.bfloat16)
    model.eval()
    print(f"loaded in {time.time() - t0:.0f}s, class {type(model).__name__}")

    # ---- 2. architecture --------------------------------------------------
    tc = getattr(model.config, "text_config", model.config)
    print(f"text layers {getattr(tc, 'num_hidden_layers', '?')}, "
          f"hidden {getattr(tc, 'hidden_size', '?')}, "
          f"config nested: {hasattr(model.config, 'text_config')}")

    # ---- document + prompt, exactly as the extractor builds it -----------
    from probe_extraction.config import load_config
    from probe_extraction.extraction.prompts import build_extraction_prompt
    from probe_extraction.extraction.parser import parse_json_output
    ex = import_module("01_extract")
    cfg = load_config(a.config)
    doc = list(ex.load_benchmark(cfg))[a.doc_index]
    print(f"document {doc.doc_id} ({len(doc.text)} chars)")
    system_msg, user_msg = build_extraction_prompt(
        schema=doc.schema, document_text=doc.text,
        include_schema=cfg.extraction.include_schema)

    def msgs(with_system):
        if with_system:
            return [{"role": "system", "content": [{"type": "text", "text": system_msg}]},
                    {"role": "user", "content": [{"type": "text", "text": user_msg}]}]
        return [{"role": "user", "content": [{"type": "text",
                                              "text": system_msg + "\n\n" + user_msg}]}]
    try:
        inputs = processor.apply_chat_template(
            msgs(True), add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")
        print("chat template: system role accepted")
    except Exception as e:
        print(f"chat template: system role rejected ({type(e).__name__}); merged into user")
        inputs = processor.apply_chat_template(
            msgs(False), add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    n_prompt = inputs["input_ids"].shape[1]
    print(f"prompt tokens {n_prompt}")

    # ---- 3. generate ------------------------------------------------------
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=a.max_new_tokens,
                             do_sample=False, output_hidden_states=True,
                             return_dict_in_generate=True)
    gen = out.sequences[0, n_prompt:]
    text = processor.decode(gen, skip_special_tokens=True)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    stopped = len(gen) > 0 and int(gen[-1]) in eos
    print(f"generated {len(gen)} tokens in {time.time() - t0:.0f}s, "
          f"finish={'stop' if stopped else 'length'}")
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    # ---- 4. hidden-state shape -------------------------------------------
    hs = out.hidden_states
    print(f"hidden_states: {len(hs)} steps; {len(hs[0])} tensors per step "
          f"(expected layers+1 = {getattr(tc, 'num_hidden_layers', 0) + 1})")
    print(f"  prefill last-layer shape {tuple(hs[0][-1].shape)}")
    if len(hs) > 1:
        print(f"  decode  last-layer shape {tuple(hs[1][-1].shape)}")
    if hs[0][-1].dim() != 3:
        print("  WARNING: not (batch, seq, hidden). Pipeline capture needs adapting.")

    # ---- 5. parse + schema keys ------------------------------------------
    print("--- first 600 chars ---")
    print(text[:600])
    parsed, err, _ = parse_json_output(text)
    if parsed is None:
        print(f"PARSE FAIL: {err}")
        verdict = "RED (no JSON)"
        extra = missing = set()
    else:
        want, got = schema_paths(doc.schema), json_paths(parsed)
        extra, missing = sorted(got - want), sorted(want - got)
        print(f"schema keys {len(want)} | emitted {len(got)} | "
              f"matched {len(want & got)}")
        print(f"EXTRA   ({len(extra)}): {extra[:15]}")
        print(f"MISSING ({len(missing)}): {missing[:15]}")
        verdict = "GREEN" if not extra else "RED (keys diverge)"
    print(f"VERDICT: {verdict}")

    outp = Path("artifacts/gemma4_smoke/smoke.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({
        "model": a.model, "doc_id": doc.doc_id, "prompt_tokens": n_prompt,
        "generated_tokens": len(gen), "stopped": stopped,
        "layers_per_step": len(hs[0]),
        "prefill_shape": list(hs[0][-1].shape),
        "verdict": verdict, "extra_keys": list(extra), "missing_keys": list(missing),
        "raw_output": text,
    }, indent=2))
    print(f"saved {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())