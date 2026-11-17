#!/usr/bin/env python3
"""Extract actual agent messages from ChatDev .log files (not prompts or summaries)."""
import argparse
import csv
import re
from pathlib import Path

EVENT = re.compile(r'^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [A-Z]+\] (.*)$')
TURN = re.compile(r'^(.+?): \*\*(.+?)<->(.+?) on : (.+?), turn (\d+)\*\*\s*$')

def extract(path):
    lines = path.read_text(encoding='utf-8-sig', errors='replace').splitlines()
    events = []
    for i, line in enumerate(lines):
        m = EVENT.match(line)
        if m:
            events.append((i, m.group(1)))
    results = []
    for k, (start, header) in enumerate(events):
        m = TURN.match(header)
        if not m:
            continue
        end = events[k + 1][0] if k + 1 < len(events) else len(lines)
        block = '\n'.join(lines[start + 1:end]).strip()
        # ChatDev prepends a bracketed system/role prompt. The actual generated
        # response starts after the first line containing only ']'.
        split = re.search(r'(?m)^.*\]\s*$', block)
        if split:
            block = block[split.end():].strip()
        else:
            # Fail closed rather than accidentally export prompts as responses.
            print(f'WARNING: skipped block without prompt delimiter at line {start+1}: {header}')
            continue
        if not block:
            continue
        speaker, role_a, role_b, phase, turn = m.groups()
        results.append(dict(row=len(results)+1, speaker=speaker, phase=phase,
                            phase_turn=int(turn), role_pair=f'{role_a}<->{role_b}',
                            message=block, source_file=path.name, log_line=start+1))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+', help='ChatDev .log files')
    parser.add_argument('--output', default='chatdev_conversations.csv')
    args = parser.parse_args()
    rows = []
    for name in args.inputs:
        path = Path(name)
        extracted = extract(path)
        print(f'{path.name}: {len(extracted)} messages')
        rows.extend(extracted)
    with open(args.output, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['source_file','row','speaker','phase','phase_turn','role_pair','message','log_line'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {len(rows)} messages to {args.output}')

if __name__ == '__main__':
    main()
