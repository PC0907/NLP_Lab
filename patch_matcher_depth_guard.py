#!/usr/bin/env python3
"""Add a recursion-depth guard to matcher._walk to stop the infinite
empty-substitution loop (RecursionError on the 2B run).

The loop: when one side is an empty dict/list and the other is None, _walk
re-substitutes an empty value and recurses on the SAME path without making
progress toward a leaf, looping until Python's recursion limit. The data is NOT
deeply nested (max real depth ~6) -- it's an in-place loop. This guard caps the
recursion: past _MAX_WALK_DEPTH it emits a type_mismatch label for the path and
returns, so the field is labelled an error (which an unresolvable match IS)
instead of crashing the whole labeling run.

Non-invasive: the 4B never recursed deep, so its labels are unaffected -- the
guard only fires on pathological loops (which the 4B didn't trigger).

Run from repo root (edits src/probe_extraction/labeling/matcher.py in place,
after backing it up):
  python patch_matcher_depth_guard.py
"""
import re, shutil, sys
from pathlib import Path

p = Path("src/probe_extraction/labeling/matcher.py")
if not p.exists():
    print(f"ERROR: {p} not found (run from repo root)"); sys.exit(1)

src = p.read_text()
orig = src

# 0. backup
shutil.copy(p, p.with_suffix(".py.bak"))

# 1. add the constant near the top (after the first import block / module docstring).
if "_MAX_WALK_DEPTH" not in src:
    # insert after the first line that starts with 'import' or 'from' block end;
    # simplest: put it right before 'def label_extraction'
    anchor = "def label_extraction("
    src = src.replace(anchor, "_MAX_WALK_DEPTH = 200  # guard against empty-substitution recursion loops\n\n\n" + anchor, 1)

# 2. add `_depth: int = 0,` to the _walk signature (keyword-only block).
#    Insert it right after the 'unmatched_extracted: list[...],' line in the SIGNATURE.
sig_pat = re.compile(
    r"(def _walk\(\s*self,\s*\*,.*?unmatched_extracted: list\[list\[str \| int\]\],\n)",
    re.DOTALL,
)
def add_depth_param(m):
    return m.group(1) + "        _depth: int = 0,\n"
src, n_sig = sig_pat.subn(add_depth_param, src, count=1)
print(f"signature patched: {n_sig}")

# 3. add the guard at the very start of the _walk BODY (right after the docstring).
#    The body begins with the docstring ending in '"""' then the
#    'sub_schema = _resolve_schema_for_value(schema, gold)' line. Insert before it.
guard = (
    "        # Depth guard: the empty-substitution branches can loop in place\n"
    "        # (gold/extracted empty<->None) without descending. Real data is\n"
    "        # shallow (~6); past the cap, treat as an unresolvable mismatch.\n"
    "        if _depth > _MAX_WALK_DEPTH:\n"
    "            self._emit_type_mismatch(\n"
    "                path=path, gold=gold, extracted=extracted, labels=labels,\n"
    "            )\n"
    "            return\n\n"
)
body_anchor = "        sub_schema = _resolve_schema_for_value(schema, gold)"
assert body_anchor in src, "body anchor (sub_schema = ...) not found"
src = src.replace(body_anchor, guard + body_anchor, 1)

# 4. thread `_depth=_depth + 1` into every RECURSIVE self._walk(...) call.
#    Each recursive call ends with 'unmatched_extracted=unmatched_extracted,\n            )'
#    We add the depth kwarg before the closing paren of each self._walk call.
#    Match self._walk(...) blocks and inject _depth before their closing ').
walk_call_pat = re.compile(
    r"(self\._walk\(\s*.*?unmatched_extracted=unmatched_extracted,\n)(\s*)\)",
    re.DOTALL,
)
def add_depth_arg(m):
    indent = m.group(2)
    return m.group(1) + indent + "    _depth=_depth + 1,\n" + indent + ")"
src, n_calls = walk_call_pat.subn(add_depth_arg, src)
print(f"recursive _walk calls patched: {n_calls}")

if src == orig:
    print("WARNING: no changes made"); sys.exit(1)

p.write_text(src)
print(f"patched {p} (backup at {p.with_suffix('.py.bak')})")

# quick syntax check
import ast
try:
    ast.parse(src)
    print("syntax OK")
except SyntaxError as e:
    print(f"SYNTAX ERROR after patch: {e}")
    print("restoring backup...")
    shutil.copy(p.with_suffix(".py.bak"), p)
    sys.exit(1)