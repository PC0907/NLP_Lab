import json, glob

for f in sorted(glob.glob('artifacts/qwen35_4b_invoices/labels/*.json')):
    if '_summary' in f:
        continue
    d = json.load(open(f))
    print('===', d['doc_id'], '|', d['n_errors'], '/', d['n_total'])
    for l in d['labels']:
        print('  err=%d  %-18s gold=%-40r ext=%r' % (
            l['is_error'], l['path_str'], l['gold_value'], l['extracted_value']))
    print()
