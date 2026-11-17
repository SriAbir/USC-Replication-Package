#!/usr/bin/env python3
"""USC RQ2 reproduction. Requires: pip install artifact-tool
Input manifest columns: conversation,file,setting,stop_rows (e.g. 4 or 4,6).
Workbook first sheet must have turn_number and metric columns.
"""
import argparse, csv, math, statistics
from pathlib import Path
from artifact_tool import Blob, SpreadsheetFile

METRICS = ['BERTScore_F1', 'Raw_SBERT', 'TaskAware_SBERT',
           'Consecutive_BLEU', 'Consecutive_ROUGE_L', 'WMSC_LoopScore']
GROUNDING = METRICS[:3]

def load_turns(path):
    wb = SpreadsheetFile.import_xlsx(Blob.load(str(path)))
    sh = wb.worksheets.get_item_at(0)
    # All supplied workbooks have fewer than 250 turns and 20 columns.
    matrix = sh.get_range('A1:T500').values
    headers = [str(x).strip() if x is not None else '' for x in matrix[0]]
    if 'turn_number' not in headers:
        raise ValueError(f'{path.name}: missing turn_number column')
    ix = {name: i for i, name in enumerate(headers) if name}
    result = {}
    for row in matrix[1:]:
        t = row[ix['turn_number']]
        if not isinstance(t, (int, float)) or not math.isfinite(t) or int(t) != t:
            continue
        values = {}
        for metric in METRICS:
            if metric not in ix:
                continue
            v = row[ix[metric]]
            if isinstance(v, (int, float)) and math.isfinite(v):
                values[metric] = float(v)
        result[int(t)] = values
    return result

def manifest_rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            stops = sorted({int(v.strip()) for v in r['stop_rows'].split(',') if v.strip()})
            if stops:
                yield r, stops

def output_csv(path, records, columns):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(records)

DEFAULT_OUTPUT = 'AUROC_per_conversation.csv'
SUMMARY_COLUMN = 'AUROC'

def compute(ref, stops, turns):
    records = []
    for metric in METRICS:
        if not any(metric in v for v in turns.values()):
            continue
        positives = [turns[t][metric] for t in stops if t in turns and metric in turns[t]]
        negatives = [v[metric] for t,v in turns.items() if t > max(stops) and metric in v]
        # Pairwise AUROC: ties receive 0.5 credit. Higher score is treated as positive.
        auc = (sum(1.0 if a > b else 0.5 if a == b else 0.0
                   for a in positives for b in negatives) / (len(positives)*len(negatives))
               if positives and negatives else '')
        records.append(dict(conversation=ref['conversation'], file=ref['file'], setting=ref['setting'],
                            metric=metric, stop_rows=','.join(map(str,stops)), last_stop=max(stops),
                            n_stop=len(positives), n_poststop=len(negatives), AUROC=auc))
    return records
COLUMNS = ['conversation','file','setting','metric','stop_rows','last_stop','n_stop','n_poststop','AUROC']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='CSV with conversation,file,setting,stop_rows')
    parser.add_argument('--data-dir', default='.', help='Directory containing metric XLSX files')
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    cache = {}
    results = []
    for ref, stops in manifest_rows(args.manifest):
        name = ref['file']
        path = Path(args.data_dir) / name
        if name not in cache:
            if not path.exists():
                raise FileNotFoundError(f'Missing conversation workbook: {path}')
            cache[name] = load_turns(path)
        turns = cache[name]
        results.extend(compute(ref, stops, turns))
    output_csv(args.output, results, COLUMNS)
    print(f'Wrote {len(results)} rows to {args.output}')
    for setting in sorted({r['setting'] for r in results}):
        for metric in sorted({r['metric'] for r in results if r['setting'] == setting}):
            values = [r[SUMMARY_COLUMN] for r in results if r['setting'] == setting and r['metric'] == metric and r[SUMMARY_COLUMN] != '']
            if values:
                print(f'{setting:23s} {metric:22s} N={len(values):2d} mean={statistics.mean(values):.6f}')

if __name__ == '__main__':
    main()
