"""Generate a synchronized gallery for a current run_rollout50 output."""
import argparse
import json
from pathlib import Path
import numpy as np

from generate_rollout_galleries import make_entry, write_page, CAMERAS, ROOT, OUT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=str, required=True)
    p.add_argument('--title', default='当前权重 rollout 测试')
    args = p.parse_args()
    run = args.run.resolve()
    summary = json.loads((run / 'summary.json').read_text())
    entries = []
    for row in summary['episodes']:
        folder = run / 'rollouts' / row['id']
        chart = folder / 'chart_data.json'
        with np.load(folder / 'rollout.npz', allow_pickle=False) as data:
            state = data['state']
            action = np.vstack([data['applied_targets'], data['applied_targets'][-1:]])
        chart.write_text(json.dumps({
            'state': state.tolist(), 'action': action.tolist(),
            'names': [f'joint_{i+1}' for i in range(16)],
            'units': ['rad'] * 6 + ['m'] * 2 + ['rad'] * 6 + ['m'] * 2,
        }, separators=(',', ':')))
        videos = [folder / f'{name}.mp4' for name in CAMERAS]
        links = [('rollout report', folder / 'report.json'),
                 ('rollout state / actions', folder / 'rollout.npz')]
        entries.append(make_entry(OUT, row['id'], row.get('source_episode'),
            row.get('seed'), bool(row.get('official_success')), bool(row.get('completed')),
            row.get('simulated_seconds', 15), row.get('steps'),
            row.get('advantage_condition', 'positive'), videos, links,
            row.get('error'), chart=chart))
    write_page(args.output, args.title, 'rollout 测试', entries,
               dict(errors=summary.get('errors', 0)))


if __name__ == '__main__':
    main()
