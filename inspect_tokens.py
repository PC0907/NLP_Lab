"""Inspect which token the 'last_token' activation position lands on.

Re-tokenizes each field's generated text with the Qwen tokenizer, then reports
what kind of token sits at the field's final span position -- content token
vs JSON structural punctuation (" } , etc). Answers the supervisor's
'which token are you capturing' question. CPU-only, no model weights.
"""
import json, glob, sys
from collections import Counter
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3.5-4B"
EXTRACTIONS = "artifacts/qwen35_4b_pooled/extractions"
SAMPLE_FIELDS = 600   # cap, for speed

# Token strings we consider "structural" rather than content.
STRUCTURAL = {'"', '},', '},\n', '}', ',', ',\n', '"\n', '":', ' "', '\n', ' ', '":\n', '],', ']'}

def classify(tok: str) -> str:
    t = tok.strip()
    if t == "":
        return "whitespace"
    if all(c in '"{}[],:' for c in t):
        return "structural"
    if t in ('"', '",', '"}', '"],'):
        return "structural"
    return "content"

def main() -> int:
    print(f"Loading tokenizer {MODEL} ...")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    at_end = Counter()       # classification of token at span[1]-1
    at_endidx = Counter()    # classification of token at span[1]
    examples = []
    n_fields = 0

    for path in sorted(glob.glob(f"{EXTRACTIONS}/*.json")):
        if path.split("/")[-1].startswith("_"):
            continue
        d = json.load(open(path))
        text = d.get("raw_generated_text", "")
        if not text:
            continue
        # Re-tokenize the generated text. token_span indexes into THIS.
        ids = tok(text, add_special_tokens=False)["input_ids"]
        toks = tok.convert_ids_to_tokens(ids)
        # Qwen tokens use a byte-BPE marker for spaces; decode each for readability.
        decoded = [tok.convert_tokens_to_string([t]) for t in toks]

        for f in d.get("fields", []):
            if f.get("is_empty"):
                continue
            span = f.get("token_span")
            if not span or len(span) != 2:
                continue
            s, e = span
            if e <= 0 or e > len(decoded):
                continue
            n_fields += 1

            tok_at_last_idx = decoded[e-1] if 0 <= e-1 < len(decoded) else ""
            tok_at_e        = decoded[e]   if 0 <= e   < len(decoded) else "(out of range)"

            c_last = classify(tok_at_last_idx)
            at_end[c_last] += 1
            if tok_at_e != "(out of range)":
                at_endidx[classify(tok_at_e)] += 1

            if len(examples) < 25:
                examples.append((f["path_str"], repr(f.get("value"))[:30],
                                  repr(tok_at_last_idx), repr(tok_at_e)))
            if n_fields >= SAMPLE_FIELDS:
                break
        if n_fields >= SAMPLE_FIELDS:
            break

    print(f"\nInspected {n_fields} fields.\n")
    print("=== token at span[1]-1 (likely the 'last_token' position) ===")
    for k, v in at_end.most_common():
        print(f"  {v:5d}  ({100*v/max(n_fields,1):5.1f}%)  {k}")
    print("\n=== token at span[1] (the index after) ===")
    for k, v in at_endidx.most_common():
        print(f"  {v:5d}  ({100*v/max(n_fields,1):5.1f}%)  {k}")
    print("\n=== sample fields (value | token@span[1]-1 | token@span[1]) ===")
    for ps, val, t1, t2 in examples:
        print(f"  {ps:32s} {val:32s}  last-1={t1:12s}  end={t2}")
    return 0

if __name__ == "__main__":
    sys.exit(main())