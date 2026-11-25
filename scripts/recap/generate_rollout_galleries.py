#!/usr/bin/env python3
"""Generate static, synchronized three-camera galleries for RECAP episodes."""
import json
import os
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'recap_runs' / 'visualizations'
CAMERAS = ('head', 'left_wrist', 'right_wrist')

HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
:root{color-scheme:light;--ink:#17222b;--muted:#63727d;--line:#d8e0e4;--surface:#fff;--canvas:#f3f6f6;--teal:#087f82;--teal-soft:#e0f3f1;--green:#18794e;--green-soft:#e4f4eb;--red:#b93838;--red-soft:#fae9e7;--amber:#996500;--amber-soft:#fff2cf}
*{box-sizing:border-box}body{margin:0;background:var(--canvas);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}.layout{display:grid;grid-template-columns:280px minmax(0,1fr);min-height:100vh}.sidebar{position:sticky;top:0;height:100vh;overflow:auto;background:var(--surface);border-right:1px solid var(--line);padding:18px 14px}.brand{font-size:18px;font-weight:700;margin:0 0 4px}.sub{font-size:12px;color:var(--muted);margin-bottom:14px}.stats{display:grid;grid-template-columns:repeat(2,1fr);gap:6px;margin-bottom:14px}.stat{border:1px solid var(--line);padding:8px;border-radius:6px;background:#fff}.stat strong{display:block;font-size:17px;line-height:1.2}.stat span{font-size:11px;color:var(--muted)}.search{width:100%;height:36px;border:1px solid var(--line);border-radius:6px;padding:0 10px;margin:0 0 8px;font:inherit;background:#fff}.filters{display:flex;gap:5px;margin:0 0 10px}.filters button,.episode,.control{font:inherit;cursor:pointer}.filters button{flex:1;border:1px solid var(--line);background:#fff;border-radius:5px;padding:5px 3px;font-size:12px;color:var(--muted)}.filters button.active{background:var(--teal-soft);color:var(--teal);border-color:#8ac9c5}.episode-list{display:grid;gap:4px}.episode{width:100%;display:grid;grid-template-columns:1fr auto;align-items:center;text-align:left;border:1px solid transparent;border-radius:5px;background:transparent;padding:7px 8px;color:var(--ink)}.episode:hover{background:#f2f6f6}.episode.active{background:var(--teal-soft);border-color:#8ac9c5}.epname{font-variant-numeric:tabular-nums;font-weight:600}.epmeta{font-size:11px;color:var(--muted)}.pill{border-radius:4px;padding:2px 6px;font-size:11px;font-weight:650;white-space:nowrap}.success{color:var(--green);background:var(--green-soft)}.failure{color:var(--red);background:var(--red-soft)}.unknown{color:var(--amber);background:var(--amber-soft)}main{min-width:0;padding:22px clamp(14px,2vw,30px)}.top{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;flex-wrap:wrap;border-bottom:1px solid var(--line);padding-bottom:14px;margin-bottom:14px}.top h1{font-size:21px;line-height:1.25;margin:0 0 5px}.subtitle{color:var(--muted);font-size:12px}.summary{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.metric{background:#fff;border:1px solid var(--line);border-radius:6px;padding:7px 10px;font-size:12px;color:var(--muted)}.metric strong{color:var(--ink);font-size:15px;margin-right:4px}.selected{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin:12px 0 4px}.selected h2{font-size:17px;margin:0}.detail{font-size:12px;color:var(--muted);margin-bottom:12px}.videos{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.video-panel{min-width:0;background:#111a1e;border-radius:6px;overflow:hidden}.video-title{display:flex;justify-content:space-between;align-items:center;background:#fff;border:1px solid var(--line);border-bottom:0;padding:7px 10px;font-size:12px;font-weight:600}.video-title small{font-weight:400;color:var(--muted)}video{display:block;width:100%;aspect-ratio:1/1;object-fit:contain;background:#101619}.controls{display:grid;grid-template-columns:auto auto minmax(80px,1fr) auto auto;gap:7px;align-items:center;background:#fff;border:1px solid var(--line);border-radius:6px;padding:8px;margin:10px 0}.control{border:1px solid var(--line);background:#fff;color:var(--ink);border-radius:5px;padding:6px 10px}.control:hover{background:var(--teal-soft);border-color:#8ac9c5}.play{background:var(--teal);border-color:var(--teal);color:#fff}.play:hover{background:#06696b;color:#fff}input[type=range]{width:100%;accent-color:var(--teal)}.clock{font-variant-numeric:tabular-nums;font-size:12px;color:var(--muted);min-width:72px;text-align:right}.links{display:flex;gap:12px;flex-wrap:wrap;margin-top:10px;font-size:12px}.links a{color:var(--teal);text-decoration:none}.links a:hover{text-decoration:underline}.error{color:var(--red);font-size:12px;min-height:18px}.empty{color:var(--muted);padding:12px;font-size:12px}
 .joint-section{margin-top:18px}.joint-heading{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;margin-bottom:8px}.joint-heading h3{margin:0;font-size:16px}.legend{display:flex;gap:14px;color:var(--muted);font-size:12px}.key{display:inline-flex;align-items:center;gap:6px}.line{width:20px;border-top:2px solid #1684c7}.line.action{border-top:2px dashed #e17b27}.chart-grid{display:grid;grid-template-columns:repeat(4,minmax(180px,1fr));gap:8px}.chart-card{background:#fff;border:1px solid var(--line);border-radius:6px;padding:8px;min-width:0}.chart-title{display:flex;justify-content:space-between;gap:6px;font-size:12px;font-weight:650;margin-bottom:3px}.chart-values{font-size:10px;color:var(--muted);font-variant-numeric:tabular-nums;white-space:nowrap}.chart-card canvas{display:block;width:100%;height:116px}
@media(max-width:950px){.layout{grid-template-columns:220px minmax(0,1fr)}main{padding:14px}.videos{gap:6px}.video-title{padding:6px;font-size:11px}.video-title small{display:none}.controls{grid-template-columns:auto auto 1fr auto}.clock{grid-column:1/-1;text-align:left}}
@media(max-width:1100px){.chart-grid{grid-template-columns:repeat(2,minmax(180px,1fr))}}
@media(max-width:650px){.layout{grid-template-columns:1fr}.sidebar{position:static;height:auto;border-right:0;border-bottom:1px solid var(--line);padding:12px}.episode-list{grid-template-columns:repeat(3,minmax(0,1fr));max-height:220px;overflow:auto}.episode{padding:5px}.epmeta{display:none}.videos{grid-template-columns:1fr}.video-panel{display:grid;grid-template-columns:86px minmax(0,1fr);align-items:stretch}.video-title{align-items:center;border:0;border-right:1px solid var(--line)}video{aspect-ratio:4/3}.controls{grid-template-columns:auto auto minmax(50px,1fr) auto}.clock{grid-column:auto;min-width:auto}.selected h2{font-size:16px}.chart-grid{grid-template-columns:1fr 1fr}.chart-card canvas{height:100px}.chart-values{font-size:9px}}
</style></head><body><div class="layout"><aside class="sidebar"><div class="brand">__SHORT_TITLE__</div><div class="sub">堆叠红色方块，再放置绿色方块</div><div class="stats" id="stats"></div><input id="search" class="search" type="search" placeholder="搜索回合 / seed" aria-label="搜索回合"><div id="filters" class="filters"></div><div id="episodes" class="episode-list"></div></aside><main><header class="top"><div><h1>__TITLE__</h1><div class="subtitle">RoboTwin · 25 Hz · 双臂 · 3 路视频</div></div><div id="summary" class="summary"></div></header><div class="selected"><h2 id="episode-title">加载中</h2><span id="badge" class="pill unknown"></span></div><div id="detail" class="detail"></div><section class="videos"><div class="video-panel"><div class="video-title">头部相机 <small>head</small></div><video id="head" muted playsinline preload="metadata"></video></div><div class="video-panel"><div class="video-title">左腕相机 <small>left wrist</small></div><video id="left" muted playsinline preload="metadata"></video></div><div class="video-panel"><div class="video-title">右腕相机 <small>right wrist</small></div><video id="right" muted playsinline preload="metadata"></video></div></section><section class="controls"><button id="previous" class="control" title="上一条">上一条</button><button id="play" class="control play">播放</button><input id="seek" type="range" min="0" max="1" step="1" value="0" aria-label="视频进度"><span id="clock" class="clock">0.00 s</span><button id="next" class="control" title="下一条">下一条</button></section><div id="links" class="links"></div><div id="error" class="error"></div></main></div>
<section id="joint-section" class="joint-section" hidden><div class="joint-heading"><h3>16 维关节轨迹</h3><div class="legend"><span class="key"><i class="line"></i>state</span><span class="key"><i class="line action"></i>action</span></div></div><div id="charts" class="chart-grid"></div></section></main></div>
<script type="application/json" id="episode-data">__DATA__</script><script>
const DB=JSON.parse(document.getElementById('episode-data').textContent),$=id=>document.getElementById(id),videos=[$('head'),$('left'),$('right')],seek=$('seek');let selected=-1,filter='all',loadToken=0,curveData=null;const fmt=n=>Number(n||0).toFixed(1),pass=e=>e.success===true,fail=e=>e.success===false;
function badge(e){return pass(e)?['成功','success']:fail(e)?['失败','failure']:['未完成','unknown']}
function drawStats(){const total=DB.episodes.length,ok=DB.episodes.filter(pass).length,bad=DB.episodes.filter(fail).length;$('stats').innerHTML=`<div class="stat"><strong>${total}</strong><span>回合</span></div><div class="stat"><strong>${ok}/${total}</strong><span>成功</span></div><div class="stat"><strong>${bad}</strong><span>失败</span></div><div class="stat"><strong>${DB.summary.errors??0}</strong><span>运行错误</span></div>`;$('summary').innerHTML=`<div class="metric"><strong>${ok}/${total}</strong>官方成功</div><div class="metric"><strong>${fmt(total?ok*100/total:0)}%</strong>成功率</div><div class="metric"><strong>${DB.episodes.filter(e=>e.completed).length}/${total}</strong>完整回合</div>`+((DB.summary.metrics||[]).map(m=>`<div class="metric"><strong>${m.value}</strong>${m.label}</div>`).join(''))}
function visible(){const q=$('search').value.trim().toLowerCase();return DB.episodes.map((e,i)=>({...e,i})).filter(e=>(filter==='all'||(filter==='success'&&pass(e))||(filter==='failure'&&fail(e)))&&(!q||`${e.id} ${e.source_episode||''} ${e.seed??''}`.toLowerCase().includes(q)))}
function renderList(){const rows=visible();$('episodes').innerHTML=rows.length?rows.map(e=>{const [txt,cls]=badge(e);return `<button class="episode ${e.i===selected?'active':''}" data-i="${e.i}" aria-pressed="${e.i===selected}"><span><span class="epname">${e.id}</span><br><span class="epmeta">${e.source_episode?`${e.source_episode} · `:''}seed ${e.seed??'—'}${e.comparison_label?` · ${e.comparison_label}`:''}</span></span><span class="pill ${cls}">${txt}</span></button>`}).join(''):'<div class="empty">没有符合条件的回合</div>';document.querySelectorAll('.episode').forEach(b=>b.onclick=()=>select(+b.dataset.i))}
function pause(){videos.forEach(v=>v.pause());$('play').textContent='播放'}
function drawCharts(){if(!curveData)return;const box=$('charts');if(!box.dataset.ready){box.innerHTML=curveData.names.map((name,j)=>`<article class="chart-card"><div class="chart-title"><span>${name}</span><span class="chart-values" id="curve-value-${j}"></span></div><canvas id="curve-${j}"></canvas></article>`).join('');box.dataset.ready='1'}const state=curveData.state,action=curveData.action,index=Math.min(state.length-1,Math.max(0,Math.floor(videos[0].currentTime*25)));for(let j=0;j<16;j++){const canvas=$(`curve-${j}`),rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=Math.max(1,rect.width),h=Math.max(1,rect.height);canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,w,h);const a=state.map(r=>r[j]),b=action.map(r=>r[j]),values=a.concat(b),lo=Math.min(...values),hi=Math.max(...values),span=Math.max(hi-lo,1e-6),min=lo-.08*span,max=hi+.08*span,L=34,R=5,T=7,B=17,W=w-L-R,H=h-T-B,x=i=>L+i/Math.max(1,a.length-1)*W,y=v=>T+(max-v)/(max-min)*H;c.strokeStyle='#e1e7e8';c.beginPath();c.moveTo(L,T);c.lineTo(L,h-B);c.lineTo(w-R,h-B);c.stroke();c.fillStyle='#7c898e';c.font='9px system-ui';c.textAlign='right';c.fillText(max.toFixed(2),L-3,T+3);c.fillText(min.toFixed(2),L-3,h-B);c.strokeStyle='#1684c7';c.lineWidth=1.5;c.beginPath();a.forEach((v,i)=>i?c.lineTo(x(i),y(v)):c.moveTo(x(i),y(v)));c.stroke();c.strokeStyle='#e17b27';c.lineWidth=1.7;c.setLineDash([4,3]);c.beginPath();b.forEach((v,i)=>i?c.lineTo(x(i),y(v)):c.moveTo(x(i),y(v)));c.stroke();c.setLineDash([]);c.strokeStyle='#263238';c.lineWidth=1;c.beginPath();c.moveTo(x(index),T);c.lineTo(x(index),h-B);c.stroke();const digits=curveData.units[j]==='m'?4:3;$(`curve-value-${j}`).textContent=`s ${a[index].toFixed(digits)} · a ${b[index].toFixed(digits)}`}}
async function select(i){if(i<0||i>=DB.episodes.length)return;const token=++loadToken;pause();selected=i;renderList();const e=DB.episodes[i],[txt,cls]=badge(e);$('episode-title').textContent=`${e.id}${e.source_episode?` · 来源 ${e.source_episode}`:''}`;$('badge').className=`pill ${cls}`;$('badge').textContent=txt;$('detail').textContent=`seed ${e.seed??'—'} · ${e.frames?`${e.frames} 帧 · `:''}${fmt(e.duration)} 秒 · ${e.comparison_label?`${e.comparison_label} · `:''}${e.advantage?`Advantage: ${e.advantage} · `:''}${e.completed?'完整执行':'未完整执行'}${e.error?` · ${e.error}`:''}`;for(let i=0;i<videos.length;i++){videos[i].src=e.videos[i];videos[i].load()}seek.value=0;seek.max=Math.max(1,Math.round((e.duration||15)*25));$('clock').textContent='0.00 s';$('error').textContent='';$('links').innerHTML=e.links.map(x=>`<a href="${x.href}" target="_blank" rel="noreferrer">${x.label}</a>`).join('');curveData=null;$('joint-section').hidden=!e.chart;if(e.chart){try{const response=await fetch(e.chart);if(!response.ok)throw Error('HTTP '+response.status);curveData=await response.json();if(token===loadToken)drawCharts()}catch(err){if(token===loadToken)$('error').textContent=`关节曲线数据加载失败：${err.message}`}}videos.forEach(v=>v.onloadedmetadata=()=>{if(token!==loadToken)return;const d=Math.min(...videos.map(x=>x.duration).filter(Number.isFinite));if(Number.isFinite(d)){seek.max=Math.max(1,Math.floor(d*25));$('clock').textContent=`0.00 / ${d.toFixed(2)} s`}})}
function currentFrame(){return Math.floor(videos[0].currentTime*25)}
$('filters').innerHTML=(DB.filters||[['all','全部'],['success','成功'],['failure','失败']]).map(([v,t])=>`<button data-filter="${v}" class="${v==='all'?'active':''}">${t}</button>`).join('');document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{filter=b.dataset.filter;document.querySelectorAll('[data-filter]').forEach(x=>x.classList.toggle('active',x===b));renderList()});$('search').oninput=renderList;
$('play').onclick=async()=>{if(!videos[0].paused){pause();return}if(videos[0].ended)videos.forEach(v=>v.currentTime=0);else videos.slice(1).forEach(v=>v.currentTime=videos[0].currentTime);try{await Promise.all(videos.map(v=>v.play()));$('play').textContent='暂停'}catch(e){$('error').textContent=`视频无法播放：${e.message}`;pause()}};
seek.oninput=()=>{pause();const t=Number(seek.value)/25;videos.forEach(v=>v.currentTime=t);$('clock').textContent=`${t.toFixed(2)} s`;drawCharts()};videos[0].addEventListener('timeupdate',()=>{const t=videos[0].currentTime;for(const v of videos.slice(1))if(Math.abs(v.currentTime-t)>.08)v.currentTime=t;seek.value=Math.floor(t*25);$('clock').textContent=`${t.toFixed(2)} / ${(videos[0].duration||0).toFixed(2)} s`;drawCharts()});videos[0].addEventListener('ended',pause);videos.forEach(v=>v.addEventListener('error',()=>{$('error').textContent='当前视频文件加载失败。'}));window.addEventListener('resize',drawCharts);function step(d){const rows=visible(),n=rows.findIndex(e=>e.i===selected);if(rows.length)select(rows[(n+d+rows.length)%rows.length].i)}$('previous').onclick=()=>step(-1);$('next').onclick=()=>step(1);document.addEventListener('keydown',e=>{if(e.target.matches('input'))return;if(e.key==='ArrowLeft')step(-1);if(e.key==='ArrowRight')step(1);if(e.code==='Space'){e.preventDefault();$('play').click()}});drawStats();renderList();if(DB.episodes.length)select(0);
</script></body></html>'''


def ensure_browser_video(source: Path) -> Path:
    """Transcode OpenCV mp4v videos to H.264 for local browser playback."""
    if source.stem.endswith('_browser'):
        return source
    import av

    target = source.with_name(source.stem + '_browser.mp4')
    if target.exists() and target.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        try:
            with av.open(str(target)) as check:
                if check.streams.video[0].codec_context.name == 'h264':
                    return target
        except Exception:
            pass
    temp = target.with_name(target.stem + '.tmp.mp4')
    try:
        with av.open(str(source)) as input_container:
            input_stream = input_container.streams.video[0]
            rate = input_stream.average_rate or Fraction(25, 1)
            with av.open(str(temp), 'w', format='mp4', options={'movflags': '+faststart'}) as output:
                output_stream = output.add_stream('libx264', rate=rate)
                output_stream.width = input_stream.width
                output_stream.height = input_stream.height
                output_stream.pix_fmt = 'yuv420p'
                # PyAV/FFmpeg builds bundled with the RoboTwin environments can
                # reject an explicit x264 profile even though the encoder works.
                output_stream.options = {'preset': 'fast', 'crf': '22'}
                index = 0
                for frame in input_container.decode(input_stream):
                    frame = frame.reformat(format='yuv420p')
                    frame.pts = index
                    frame.time_base = Fraction(rate.denominator, rate.numerator)
                    for packet in output_stream.encode(frame):
                        output.mux(packet)
                    index += 1
                for packet in output_stream.encode():
                    output.mux(packet)
        os.replace(temp, target)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    return target


def rel(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


def make_entry(base: Path, ep_id: str, source_id: str | None, seed, success, completed,
               duration, frames, advantage, video_sources, links, error=None, chart=None):
    videos = [ensure_browser_video(p.resolve()) for p in video_sources]
    return dict(id=ep_id, source_episode=source_id, seed=seed, success=success,
                completed=completed, duration=duration, frames=frames, advantage=advantage,
                videos=[rel(p, base) for p in videos], chart=rel(chart.resolve(), base) if chart else None,
                links=[dict(label=label, href=rel(path.resolve(), base)) for label, path in links],
                error=error)


def write_page(filename: str, title: str, short_title: str, episodes: list[dict], summary: dict):
    destination = OUT / filename
    payload = dict(episodes=episodes, summary=summary)
    data = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    page = (HTML.replace('__TITLE__', title).replace('__SHORT_TITLE__', short_title)
            .replace('__DATA__', data))
    destination.write_text(page, encoding='utf-8')
    print(f'Wrote {destination} ({len(episodes)} episodes)', flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    success_root = ROOT / 'data' / 'robotwin_success_80'
    manifest = json.loads((success_root / 'manifest.json').read_text())
    train = [e for e in manifest['episodes'] if e['split'] == 'train']
    success_entries = []
    for i, e in enumerate(train, 1):
        folder = success_root / e['directory']
        with __import__('numpy').load(success_root / e['npz'], allow_pickle=False) as raw:
            frames = len(raw['state'])
        success_entries.append(make_entry(OUT, e['id'], None, e['seed'], True, True,
            e['duration_seconds'], frames, None,
            [folder / f'{name}_browser.mp4' for name in CAMERAS],
            [('原始 episode NPZ', success_root / e['npz']),
             ('关节曲线数据 JSON', folder / 'chart_data.json'),
             ('采集报告', folder / 'report.json')], chart=folder / 'chart_data.json'))
        if i % 10 == 0:
            print(f'Prepared training episodes {i}/{len(train)}', flush=True)
    assert len(success_entries) == 60
    write_page('success_train_60.html', '成功训练集 · 60 条示范', '成功训练集',
               success_entries, dict(errors=0))

    for run, filename, title, short_title in (
        ('rollout_round0_50_positive_fixed', 'new_checkpoint_test_50.html',
         '新权重测试 · 50 条 rollout', '新权重测试'),
        ('old_checkpoint_same_scenes_30', 'old_checkpoint_test_30.html',
         '旧权重测试 · 30 条 rollout', '旧权重测试')):
        run_root = ROOT / 'recap_runs' / run
        summary = json.loads((run_root / 'summary.json').read_text())
        rows = summary['episodes']
        entries = []
        for i, row in enumerate(rows, 1):
            folder = run_root / 'rollouts' / row['id']
            entries.append(make_entry(OUT, row['id'], row.get('source_episode'),
                row.get('seed'), bool(row['official_success']), bool(row.get('completed')),
                row.get('simulated_seconds', 15), row.get('steps'), row.get('advantage_condition'),
                [folder / f'{name}.mp4' for name in CAMERAS],
                [('rollout report', folder / 'report.json'),
                 ('rollout state / actions', folder / 'rollout.npz')],
                row.get('error')))
            if i % 10 == 0:
                print(f'Prepared {short_title} episodes {i}/{len(rows)}', flush=True)
        assert len(entries) == summary['total']
        write_page(filename, title, short_title, entries,
                   dict(errors=summary.get('errors', 0)))


if __name__ == '__main__':
    main()
