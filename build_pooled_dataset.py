"""Build a pooled four-domain experiment directory for combined probe/LODO runs.

Copies labels/ and activations/ from each source experiment into one pooled
directory. For finance/10kq, drops the systematic-bias metadata field types
(segment_name, data_period) which would otherwise let the probe learn field
identity instead of a genuine trust signal.
"""
import json, shutil, sys
from pathlib import Path

ARTIFACTS = Path("artifacts")
POOLED = ARTIFACTS / "qwen35_4b_pooled"

# source experiment dir -> set of field-type names to drop from its labels
SOURCES = {
    "qwen35_4b_pymupdf":  set(),                          # academic
    "qwen35_4b_swimming": set(),
    "qwen35_4b_credit":   set(),
    "qwen35_4b_10kq":     {"segment_name", "data_period"},  # systematic-bias fields
}

def main() -> int:
    (POOLED / "labels").mkdir(parents=True, exist_ok=True)
    (POOLED / "activations").mkdir(parents=True, exist_ok=True)

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
            total_docs += 1

    print(f"Pooled {total_docs} docs, {total_labels} labels kept, "
          f"{total_dropped} dropped (10kq metadata fields).")
    print(f"Pooled dir: {POOLED}")
    return 0

if __name__ == "__main__":
    sys.exit(main())