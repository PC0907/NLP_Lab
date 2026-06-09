"""Build a pooled four-domain experiment directory for combined probe/LODO runs.

Copies labels/ and activations/ from each source experiment into one pooled
directory, dropping field types where gold-vs-model disagreement is systematic
and NOT a genuine model trust signal — i.e. where keeping them would teach the
probe field identity or penalize the model for annotation/representation
mismatches rather than real extraction errors.

Dropped field types and rationale:
  - 10kq segment_name, data_period: systematic metadata bias (model emits
    "Company"/period where gold has "NA"/different period).
  - academic citations: gold stores full bibliographic strings; the model
    emits short author-year keys. A single set-valued field collapses ~11
    correct short-form citations and a few genuine garbage entries into one
    binary label — uninformative as a trust signal. Documented as a
    representation-convention mismatch.
  - swimming records: gold systematically under-annotates record markers
    (WR/CR/NR) that the model correctly extracts from the document, producing
    false-positive "hallucination" labels.   [enable only if confirmed systematic]
"""

import json
import shutil
import sys
from pathlib import Path

ARTIFACTS = Path("artifacts")
POOLED = ARTIFACTS / "llama31_8b_pooled"
SOURCES = {
    "llama31_8b_pymupdf":  {"citations"},
    "llama31_8b_swimming": set(),
    "llama31_8b_credit":   set(),
    "llama31_8b_10kq":     {"segment_name", "data_period"},
}


def main() -> int:
    (POOLED / "labels").mkdir(parents=True, exist_ok=True)
    (POOLED / "activations").mkdir(parents=True, exist_ok=True)
    (POOLED / "extractions").mkdir(parents=True, exist_ok=True)

    total_docs = total_labels = total_dropped = 0

    for exp_name, drop_fields in SOURCES.items():
        src = ARTIFACTS / exp_name
        src_labels = src / "labels"
        src_acts = src / "activations"
        if not src_labels.is_dir():
            print(f"  WARNING: {src_labels} missing, skipping {exp_name}")
            continue

        for lab_path in sorted(src_labels.glob("*.json")):
            if lab_path.name.startswith("_"):
                continue
            d = json.load(lab_path.open())

            # Drop the systematic-bias field types for this source.
            kept = []
            for l in d.get("labels", []):
                field_type = l["path_str"].split(".")[-1]
                if field_type in drop_fields:
                    total_dropped += 1
                    continue
                kept.append(l)
            d["labels"] = kept
            total_labels += len(kept)

            # Write filtered labels into the pooled dir.
            (POOLED / "labels" / lab_path.name).write_text(
                json.dumps(d, indent=2, ensure_ascii=False)
            )

            # Copy the matching activations file unchanged.
            act_path = src_acts / f"{lab_path.stem}.npz"
            if act_path.exists():
                shutil.copy2(act_path, POOLED / "activations" / act_path.name)
            else:
                print(f"  WARNING: no activations for {lab_path.stem}")

            # Copy the matching extraction file (token logprobs for baselines).
            ext_path = src / "extractions" / f"{lab_path.stem}.json"
            if ext_path.exists():
                shutil.copy2(ext_path, POOLED / "extractions" / ext_path.name)
            else:
                print(f"  WARNING: no extraction file for {lab_path.stem}")

            total_docs += 1

    print(f"Pooled {total_docs} docs, {total_labels} labels kept, "
          f"{total_dropped} dropped (systematic-bias / representation-mismatch fields).")
    print(f"Pooled dir: {POOLED}")
    return 0


if __name__ == "__main__":
    sys.exit(main())