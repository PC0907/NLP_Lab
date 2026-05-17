# save as check_gold_coverage.py
import json, glob
from probe_extraction.data.real_kie import RealKIE

b = RealKIE('datasets/realkie', domains=['nda'])
golds = {d.doc_id: d.gold for d in b}

total_err = total_lab = gold_none_err = real_err = 0
for f in glob.glob('artifacts/qwen35_4b_nda/labels/*.json'):
    if '_summary' in f: continue
    d = json.load(open(f))
    for l in d['labels']:
        total_lab += 1
        if l['is_error']:
            total_err += 1
            if l['gold_value'] is None:
                gold_none_err += 1
            else:
                real_err += 1

print(f'labels: {total_lab}')
print(f'errors: {total_err}')
print(f'  gold=None errors (suspect): {gold_none_err}')
print(f'  gold-present errors (real): {real_err}')