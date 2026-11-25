#!/usr/bin/env python3
"""Run 50 closed-loop RoboTwin rollouts from the completed round-0 checkpoint."""
import json, subprocess, time, shutil, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/robotwin_success_80'
PYTHON = Path(sys.executable)
INFER = ROOT / 'scripts/infer_smol_robotwin_qpos16.py'

def write(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp'); tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)); tmp.replace(path)

def recycle(path):
    destination = ROOT / '回收站' / time.strftime('%Y%m%d_%H%M%S') / path.parent.parent.name / path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination = destination.with_name(destination.name + '_' + str(time.time_ns()))
    shutil.move(str(path), str(destination))

def main():
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint', type=Path, required=True); p.add_argument('--output', type=Path, required=True); p.add_argument('--count', type=int, default=50); p.add_argument('--device', default='cuda'); p.add_argument('--task-prompt'); p.add_argument('--advantage',choices=('positive','negative'))
    p.add_argument('--data', type=Path, default=DATA, help='success dataset containing manifest and original scene configs')
    p.add_argument('--python', type=Path, default=PYTHON, help='RoboTwin inference interpreter')
    p.add_argument('--split', choices=('train','validation'), help='restrict scene selection to this split')
    p.add_argument('--rollout-steps', type=int, default=375)
    p.add_argument('--flow-steps', type=int, default=4)
    a=p.parse_args()
    assert a.rollout_steps > 0 and a.flow_steps > 0
    a.checkpoint=a.checkpoint.resolve(); a.output=a.output.resolve()
    assert a.checkpoint.is_file() and a.python.is_file()
    a.output.mkdir(parents=True, exist_ok=True); (a.output/'rollouts').mkdir(exist_ok=True)
    data=a.data.resolve()
    entries=json.loads((data/'manifest.json').read_text())['episodes']
    if a.split: entries=[e for e in entries if e['split']==a.split]
    assert 0 < a.count <= len(entries), 'count must fit the requested scene split'
    selection_file=a.output/'selection.json'
    if selection_file.exists():
        saved=json.loads(selection_file.read_text()); assert saved['checkpoint']==str(a.checkpoint.resolve()) and saved['count']==a.count
        assert saved.get('data',str(data))==str(data) and saved.get('split')==a.split
        assert saved.get('advantage')==a.advantage and saved.get('task_prompt')==a.task_prompt
        assert saved.get('rollout_steps',375)==a.rollout_steps and saved.get('flow_steps',4)==a.flow_steps
        plan=saved['entries']
    else:
        selected=np.random.default_rng(20261108).choice(len(entries), a.count, replace=False)
        plan=[entries[int(i)] for i in selected]
        write(selection_file, {'checkpoint':str(a.checkpoint.resolve()),'count':a.count,'seed':20261108,'data':str(data),'split':a.split,'advantage':a.advantage,'task_prompt':a.task_prompt,'rollout_steps':a.rollout_steps,'flow_steps':a.flow_steps,'entries':plan})
    rows=[]; existing=set(); start=time.time()
    for n,e in enumerate(plan):
        episode_dir=a.output/'rollouts'/f'episode_{n:03d}'
        rp=episode_dir/'report.json'
        if rp.exists():
            row=json.loads(rp.read_text())
            videos={name:(episode_dir/f'{name}.mp4').is_file()
                    and (episode_dir/f'{name}.mp4').stat().st_size>0
                    for name in ('head','left_wrist','right_wrist')}
            if row.get('completed') and all(videos.values()):
                row['videos']=videos
                row.update(id=f'episode_{n:03d}',source_episode=e['id'],seed=e.get('seed'))
                rows.append(row); existing.add(n)
            else:
                recycle(episode_dir)
    def progress():
        successes=sum(bool(x.get('official_success')) for x in rows)
        finished=len(rows)==a.count
        write(a.output/'summary.json', {'status':'completed' if finished else 'running','finished':len(rows),'total':a.count,'successes':successes,'success_rate':successes/len(rows) if rows else None,'errors':sum(not r.get('completed',False) for r in rows),'episodes':rows,'elapsed_seconds':time.time()-start})
        rate=f'{successes/len(rows):.1%}' if rows else '尚无结果'
        (a.output/'测试进度.md').write_text('# 闭环 rollout 测试\n\n'+f'- 状态：{"完成" if finished else "运行中"}\n- 完成：{len(rows)}/{a.count}\n- 官方成功：{successes}\n- 成功率：{rate}\n- Advantage：positive\n')
    progress()
    for n,e in enumerate(plan):
        if n in existing: continue
        out=a.output/'rollouts'/f'episode_{n:03d}'; raw=data/e['npz']; begin=time.time()
        # An interrupted inference can leave an empty output directory without
        # report.json. Remove that stale directory before retrying the episode.
        if out.exists() and not (out/'report.json').exists():
            recycle(out)
        cmd=[str(a.python),str(INFER),'--mode','rollout','--raw',str(raw),'--checkpoint',str(a.checkpoint),'--output',str(out),'--device',a.device,'--flow-steps',str(a.flow_steps),'--rollout-steps',str(a.rollout_steps)]
        if a.task_prompt: cmd.extend(['--task-prompt',a.task_prompt])
        if a.advantage: cmd.extend(['--advantage',a.advantage])
        with (a.output/f'episode_{n:03d}.log').open('w') as log: result=subprocess.run(cmd,cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT)
        report=json.loads((out/'report.json').read_text()) if (out/'report.json').exists() else {'completed':False,'error':'missing report'}
        videos={name:(out/f'{name}.mp4').is_file() and (out/f'{name}.mp4').stat().st_size>0 for name in ('head','left_wrist','right_wrist')}
        report['videos']=videos
        if not all(videos.values()):
            report['completed']=False
            report['video_error']='one or more camera videos are missing or empty'
        report.update(id=f'episode_{n:03d}',source_episode=e['id'],seed=e.get('seed'),exit_code=result.returncode,wall_seconds=time.time()-begin)
        rows.append(report); progress(); print(json.dumps({'finished':len(rows),'total':a.count,'id':report['id'],'success':report.get('official_success'),'completed':report.get('completed')},ensure_ascii=False),flush=True)
    progress()
    if any(not row.get('completed') or row.get('exit_code', 0) != 0 for row in rows):
        raise SystemExit('one or more rollout episodes failed; see summary.json')

if __name__=='__main__': main()
