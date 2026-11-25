#!/usr/bin/env python3
"""Build the standard video + 16-joint curve gallery for collected episodes."""
import json
import os
import sys
from pathlib import Path
import numpy as np

from recap.generate_rollout_galleries import make_entry, write_page, OUT, rel

# The page fetches chart_data.json, which browsers block under file://. Inline the
# payloads and answer those fetches from memory so the page works without a server.
SHIM = (
    '<script>window.__CURVES__=__DATA__;'
    '(function(){var raw=window.fetch;'
    'window.fetch=function(input,init){var key=String((input&&input.url)||input);'
    'if(Object.prototype.hasOwnProperty.call(window.__CURVES__,key)){'
    'var body=JSON.stringify(window.__CURVES__[key]);'
    'return Promise.resolve(new Response(body,{status:200,'
    "headers:{'Content-Type':'application/json'}}));}"
    'return raw(input,init);};})();</script>')


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', default='robotwin_success_qpos16_pilot3.html')
    args = p.parse_args()
    root = args.dataset.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    entries, curves = [], {}
    for e in manifest['episodes']:
        folder = root / 'episodes' / e['id']
        raw = folder / f"{e['id']}.npz"
        with np.load(raw, allow_pickle=False) as d:
            state, action = d['state'], d['action']
            names = d['joint_names'].tolist()
        payload = dict(state=np.round(state, 5).tolist(), action=np.round(action, 5).tolist(),
                       names=names, units=(['rad'] * 6 + ['m'] * 2) * 2)
        chart = folder / 'chart_data.json'
        chart.write_text(json.dumps(payload, separators=(',', ':')))
        curves[rel(chart, OUT)] = payload
        entries.append(make_entry(OUT, e['id'], None, e['seed'], True, True,
            e['frames'] / 25, e['frames'], None,
            [folder / 'head.mp4', folder / 'left_wrist.mp4', folder / 'right_wrist.mp4'],
            [('episode NPZ', raw), ('collection report', folder / 'report.json'),
             ('environment config', folder / 'environment_config.json')], chart=chart))
    count = len(entries)
    seconds = sum(e['frames'] for e in entries) / 25.
    write_page(args.output, f'RoboTwin 成功数据采集 · {count} 条', f'RoboTwin 采集 · {count} 条',
               entries,
               dict(errors=0, collection_attempts=manifest.get('attempts'),
                    count=count, total_seconds=seconds))
    page_path = OUT / args.output
    page = page_path.read_text(encoding='utf-8')
    shim = SHIM.replace('__DATA__', json.dumps(curves, separators=(',', ':')).replace('<', '\\u003c'))
    marker = '<meta charset="utf-8">'
    if marker not in page:
        raise RuntimeError('gallery template changed: charset meta tag not found')
    page_path.write_text(page.replace(marker, marker + shim, 1), encoding='utf-8')
    print(f'Inlined {len(curves)} curve payload(s) into {page_path.name}', flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
