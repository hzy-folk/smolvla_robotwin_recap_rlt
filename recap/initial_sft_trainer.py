"""RoboTwin qpos16 SmolVLA fine-tuning. No run starts merely by importing this module."""
import argparse
import collections
import contextlib
import fcntl
import json
import math
import os
import random
import signal
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "collection"))
from dataset import ROOT
from robotwin_smol_data import DEFAULT_DATA, CONTRACT, FEATURES, RobotwinDataset, make_batch, sha
from robotwin_smol_model import BASE, make, torch
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


def cpu_tree(x):
    if torch.is_tensor(x):
        return x.detach().cpu().clone()
    if isinstance(x, dict):
        return {k: cpu_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(cpu_tree(v) for v in x)
    return x


def atomic_json(path, value):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


class Sampler:
    def __init__(self, ids, seed):
        self.ids = np.asarray(ids)
        self.rng = np.random.default_rng(seed)
        self.order, self.position, self.epochs = self.rng.permutation(self.ids), 0, 0

    def next(self, size):
        chunks = []
        while size:
            n = min(size, len(self.order) - self.position)
            chunks.append(self.order[self.position:self.position + n])
            self.position += n
            size -= n
            if self.position == len(self.order):
                self.order, self.position = self.rng.permutation(self.ids), 0
                self.epochs += 1
        return np.concatenate(chunks)

    def state_dict(self):
        return dict(order=torch.tensor(self.order), position=self.position,
                    epochs=self.epochs, rng=self.rng.bit_generator.state)

    def load_state_dict(self, saved):
        self.order, self.position, self.epochs = saved['order'].numpy(), saved['position'], saved['epochs']
        self.rng.bit_generator.state = saved['rng']


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data', type=Path, default=DEFAULT_DATA)
    p.add_argument('--cache', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--batch-size', type=int, default=64, help='effective batch per optimizer update')
    p.add_argument('--micro-batch', type=int, default=8, help='physical batch; accumulate to --batch-size')
    p.add_argument('--warmup-steps', type=int, default=1000)
    p.add_argument('--validation-windows', type=int, default=128)
    p.add_argument('--validation-every', type=int, default=500)
    p.add_argument('--save-every', type=int, default=5000)
    p.add_argument('--save-at', type=str, help='comma-separated optimizer steps to checkpoint, e.g. 8000,12000')
    p.add_argument('--delete-checkpoint-after', type=str,
                   help='delete STEP checkpoint after verified AFTER checkpoint, e.g. 8000:12000')
    p.add_argument('--report-every', type=int, default=2000)
    p.add_argument('--seed', type=int, default=191)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--pause-at', type=int, help='pause after completing and checkpointing this optimizer step')
    p.add_argument('--resume-training-source-sha', help='explicit previous trainer SHA256 allowed on resume; all other config/source hashes must match')
    a = p.parse_args()
    assert min(a.steps, a.batch_size, a.micro_batch, a.validation_windows,
               a.validation_every, a.save_every, a.report_every) > 0
    assert a.micro_batch <= a.batch_size and 0 <= a.warmup_steps < a.steps
    assert a.pause_at is None or 0 < a.pause_at <= a.steps
    save_steps = sorted({int(x) for x in a.save_at.split(',')}) if a.save_at else None
    assert save_steps is None or (save_steps and all(0 < x <= a.steps for x in save_steps))
    delete_pair = tuple(map(int, a.delete_checkpoint_after.split(':'))) if a.delete_checkpoint_after else None
    assert delete_pair is None or (len(delete_pair) == 2 and delete_pair[0] < delete_pair[1]
                                   and delete_pair[1] <= a.steps)
    if save_steps is not None:
        assert delete_pair is None or set(delete_pair).issubset(save_steps)
    run = a.output.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / '.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert a.resume or not (run / 'training_config.json').exists(), 'use a new output or --resume'
    torch.set_num_threads(4)
    if a.device == 'cuda':
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    halted = [False]
    signal.signal(signal.SIGINT, lambda *_: halted.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: halted.__setitem__(0, True))
    data = RobotwinDataset(a.data, a.cache)
    stats = data.statistics(torch)
    config = dict(platform='robotwin', contract=data.contract, seed=a.seed,
        data_root=str(data.source), data_fingerprint=data.fingerprint,
        frames=len(data.state), train_frames=len(data.indices('train')), validation_frames=len(data.indices('val')),
        batch_size=a.batch_size, micro_batch=a.micro_batch,
        gradient_accumulation=math.ceil(a.batch_size / a.micro_batch), steps=a.steps,
        warmup_steps=a.warmup_steps, optimizer='AdamW', lr=1e-4, betas=[.9, .95], eps=1e-8,
        weight_decay=1e-10, grad_clip=10., scheduler='official cosine decay', min_lr=2.5e-6,
        validation_every=a.validation_every, validation_windows=a.validation_windows,
        save_every=a.save_every, save_steps=save_steps, delete_checkpoint_after=delete_pair,
        report_every=a.report_every, device=a.device,
        precision='bf16 autocast' if a.device == 'cuda' else 'float32',
        camera_keys=list(FEATURES), base_model_sha256=sha(BASE),
        source_sha256={name: sha(path) for name, path in {
            'initial_sft_trainer.py': Path(__file__),
            'initial_sft_data.py': Path(__file__).with_name('initial_sft_data.py'),
            'robotwin_smol_data.py': Path(__file__).parents[1] / 'collection' / 'robotwin_smol_data.py',
            'robotwin_smol_model.py': Path(__file__).parents[1] / 'collection' / 'robotwin_smol_model.py'}.items()})
    print(json.dumps(dict(event='configuration', **config), ensure_ascii=False), flush=True)
    model = make(stats).to(a.device)
    trainable = {n: v for n, v in model.named_parameters() if v.requires_grad}
    optimizer = torch.optim.AdamW(trainable.values(), lr=1e-4, betas=(.9, .95), eps=1e-8, weight_decay=1e-10)
    scheduler = CosineDecayWithWarmupSchedulerConfig(num_warmup_steps=a.warmup_steps,
        num_decay_steps=a.steps, peak_lr=1e-4, decay_lr=2.5e-6).build(optimizer, a.steps)
    sampler = Sampler(data.indices('train'), a.seed)
    first, last_saved, val_loss = 1, -1, None
    if a.resume:
        latest = json.loads((run / 'latest.json').read_text())
        saved = torch.load(run / latest['path'], map_location='cpu', weights_only=False)
        previous_config = saved['config']
        if a.resume_training_source_sha:
            previous_config = json.loads(json.dumps(previous_config))
            assert previous_config['source_sha256'][Path(__file__).name] == a.resume_training_source_sha
            previous_config['source_sha256'][Path(__file__).name] = config['source_sha256'][Path(__file__).name]
        assert json.loads(json.dumps(previous_config)) == json.loads(json.dumps(config)), 'resume configuration/data/source mismatch'
        assert set(saved['trainable']) == set(trainable)
        model.load_state_dict(saved['trainable'], strict=False)
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        sampler.load_state_dict(saved['sampler'])
        random.setstate(saved['rng']['python']); np.random.set_state(saved['rng']['numpy'])
        torch.set_rng_state(saved['rng']['cpu'])
        if a.device == 'cuda':
            torch.cuda.set_rng_state_all(saved['rng']['cuda'])
        first, last_saved, val_loss = saved['step'] + 1, saved['step'], saved['val_loss']
        assert a.pause_at is None or a.pause_at >= first, 'pause step already reached'
        del saved
        atomic_json(run / 'training_config.json', config)
    else:
        atomic_json(run / 'training_config.json', config)
    (run / 'pid').write_text(str(os.getpid()))
    progress_path = run / '训练进度.md'
    progress_rows, checkpoint_rows = [], []
    if a.resume and (run / 'events.jsonl').exists():
        for line in (run / 'events.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row['event'] == 'progress':
                progress_rows.append(row)
            elif row['event'] == 'checkpoint':
                checkpoint_rows.append(f'{row["step"]} 步：已保存并校验 `{Path(row["path"]).name}`。')
            elif row['event'] == 'checkpoint_deleted':
                checkpoint_rows.append(f'{row["step"]} 步：检查点已删除。')

    def write_progress(status, detail=''):
        local_time = lambda stamp: datetime.fromtimestamp(stamp, ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
        lines = [
            '# RoboTwin SmolVLA 训练进度', '',
            f'- 状态：{status}',
            f'- 配置：总更新步数 {a.steps}；物理 batch {a.micro_batch}；有效 batch {a.batch_size}；梯度累积 {math.ceil(a.batch_size / a.micro_batch)} 次。',
            f'- 设备：{a.device}；精度：{config["precision"]}；随机种子：{a.seed}',
            f'- 数据：{config["frames"]} 帧（训练 {config["train_frames"]}，验证 {config["validation_frames"]}）；数据指纹 `{config["data_fingerprint"]}`。',
            f'- 验证：每 {a.validation_every} 步，固定 {a.validation_windows} 个窗口。',
            f'- 保存：{", ".join(map(str, save_steps)) + " 步" if save_steps else f"每 {a.save_every} 步"}'
            + (f'；确认 {delete_pair[1]} 步检查点后删除 {delete_pair[0]} 步检查点。' if delete_pair else '。'),
            f'- 汇报：每 {a.report_every} 步记录训练 loss、验证 loss、剩余时长和预计结束时间。',
            f'- 临时暂停：{str(a.pause_at) + " 步完成后额外保存完整检查点并暂停" if a.pause_at else "未设置"}。',
        ]
        if detail:
            lines.extend(['', detail])
        lines.extend(['', '## 进度记录', '', '| 更新步 | 训练 loss（近100步均值） | 验证 loss | 学习率 | 预计剩余 | 预计结束（北京时间） | 峰值显存 MiB |',
                      '|---:|---:|---:|---:|---:|---|---:|'])
        for row in progress_rows:
            eta = f'{row["remaining_update_seconds"] / 3600:.2f} 小时'
            finish = local_time(row['estimated_finish_unix'])
            peak = row.get('peak_allocated_MiB')
            lines.append(f'| {row["step"]} | {row["train_loss"]:.6g} | {row["validation_loss"]:.6g} | {row["lr"]:.3g} | {eta} | {finish} | {peak:.0f} |' if peak is not None else f'| {row["step"]} | {row["train_loss"]:.6g} | {row["validation_loss"]:.6g} | {row["lr"]:.3g} | {eta} | {finish} | - |')
        lines.extend(['', '## 检查点', ''])
        lines.extend(f'- {item}' for item in checkpoint_rows) if checkpoint_rows else lines.append('- 尚未保存')
        progress_path.write_text('\n'.join(lines) + '\n')

    def emit(event, **values):
        row = dict(event=event, unix_time=time.time(), **values)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        with (run / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        return row

    def amp():
        return torch.autocast('cuda', dtype=torch.bfloat16) if a.device == 'cuda' else contextlib.nullcontext()

    val_ids = np.random.default_rng(a.seed + 1).choice(data.indices('val'), a.validation_windows, replace=False)

    def validate():
        model.eval()
        total = 0.
        # Fixed noise/time stream; validation cannot change training RNG.
        with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()] if a.device == 'cuda' else []):
            torch.manual_seed(a.seed + 1)
            for offset in range(0, len(val_ids), a.micro_batch):
                ids = val_ids[offset:offset + a.micro_batch]
                with amp():
                    loss, _ = model(make_batch(data, ids, torch, a.device))
                assert torch.isfinite(loss)
                total += float(loss) * len(ids)
        model.train()
        return total / len(val_ids)

    def checkpoint(step, reason):
        nonlocal last_saved
        path = run / f'checkpoint_{step:06d}.pt'
        state = dict(step=step, platform='robotwin', state_dim=16, action_dim=16, config=config,
            stats=stats, joint_names=data.joint_names, joint_units=data.joint_units,
            trainable={n: v.detach().cpu().clone() for n, v in trainable.items()},
            optimizer=cpu_tree(optimizer.state_dict()), scheduler=scheduler.state_dict(),
            sampler=sampler.state_dict(), val_loss=val_loss,
            rng=dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                     cuda=torch.cuda.get_rng_state_all() if a.device == 'cuda' else []))
        tmp = path.with_suffix('.pt.tmp')
        with tmp.open('wb') as stream:
            torch.save(state, stream); stream.flush(); os.fsync(stream.fileno())
        # Verify serialized trainable weights before promoting/removing anything.
        restored = torch.load(tmp, map_location='cpu', weights_only=False)
        assert restored['step'] == step and restored['config'] == config
        assert all(torch.equal(restored['trainable'][n], v) for n, v in state['trainable'].items())
        os.replace(tmp, path)
        atomic_json(run / 'latest.json', dict(step=step, path=path.name))
        last_saved = step
        emit('checkpoint', step=step, reason=reason, path=str(path))
        checkpoint_rows.append(f'{step} 步：已保存并校验 `{path.name}`。')
        if delete_pair and step == delete_pair[1]:
            old = run / f'checkpoint_{delete_pair[0]:06d}.pt'
            if old.exists():
                old.unlink()
                emit('checkpoint_deleted', step=delete_pair[0], after_verified_step=step, path=str(old))
                checkpoint_rows.append(f'{delete_pair[0]} 步：{step} 步检查点校验通过后已删除。')
            else:
                checkpoint_rows.append(f'{delete_pair[0]} 步：未发现需删除的检查点。')
        write_progress('训练中')

    timings, losses = collections.deque(maxlen=100), collections.deque(maxlen=100)
    step = first - 1
    wall_training_started = time.monotonic()
    try:
        if a.resume:
            emit('resumed', step=first - 1, pause_at=a.pause_at,
                 previous_training_source_sha=a.resume_training_source_sha,
                 training_source_sha=config['source_sha256'][Path(__file__).name])
        val_loss = validate()
        emit('initial_validation', step=step, loss=val_loss,
             trainable_parameters=sum(v.numel() for v in trainable.values()))
        write_progress('训练中', f'初始验证 loss：{val_loss:.6g}。')
        for step in range(first, a.steps + 1):
            if halted[0] or (run / 'STOP').exists():
                step -= 1
                break
            if a.device == 'cuda':
                torch.cuda.synchronize()
            started = time.monotonic()
            ids = sampler.next(a.batch_size)
            optimizer.zero_grad(set_to_none=True)
            update_loss = 0.
            for offset in range(0, a.batch_size, a.micro_batch):
                sub = ids[offset:offset + a.micro_batch]
                weight = len(sub) / a.batch_size
                with amp():
                    loss, details = model(make_batch(data, sub, torch, a.device))
                assert torch.isfinite(loss)
                (loss * weight).backward()
                update_loss += float(loss.detach()) * weight
                del loss, details
            grad = torch.nn.utils.clip_grad_norm_(trainable.values(), 10.)
            assert torch.isfinite(grad)
            optimizer.step(); scheduler.step()
            if a.device == 'cuda':
                torch.cuda.synchronize()
            timings.append(time.monotonic() - started); losses.append(update_loss)
            if step % a.validation_every == 0 or step == a.steps:
                val_loss = validate()
            if (step in save_steps if save_steps is not None else step % a.save_every == 0) or step == a.steps:
                checkpoint(step, 'scheduled')
            if step % a.report_every == 0 or step == a.steps:
                wall_seconds_per_update = (time.monotonic() - wall_training_started) / (step - first + 1)
                remaining = (a.steps - step) * wall_seconds_per_update
                row = emit('progress', step=step, target_steps=a.steps, train_loss=float(np.mean(losses)),
                    validation_loss=val_loss, lr=optimizer.param_groups[0]['lr'], grad_norm=float(grad),
                    seconds_per_update=float(np.mean(timings)), remaining_update_seconds=remaining,
                    wall_seconds_per_update=wall_seconds_per_update,
                    estimated_finish_unix=time.time() + remaining,
                    eta_note='wall-clock average since first update; future validation and saving cadence may shift ETA',
                    completed_epochs=sampler.epochs,
                    peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20 if a.device == 'cuda' else None)
                atomic_json(run / 'status.json', row)
                progress_rows.append(row)
                write_progress('训练中')
            if step == a.pause_at:
                if step != last_saved:
                    checkpoint(step, 'requested_pause')
                break
        if step != last_saved:
            checkpoint(step, 'pause')
        atomic_json(run / 'status.json', emit('completed' if step == a.steps else 'paused', step=step,
            train_loss=float(np.mean(losses)) if losses else None, validation_loss=val_loss))
        write_progress('已完成' if step == a.steps else '已暂停', f'最终步数：{step}；验证 loss：{val_loss:.6g}。')
    except BaseException as exc:
        atomic_json(run / 'status.json', emit('error', step=step, error=repr(exc)))
        write_progress('发生错误', repr(exc))
        raise


if __name__ == '__main__':
    main()
