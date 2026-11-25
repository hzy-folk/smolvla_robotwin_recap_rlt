#!/usr/bin/env python3
"""Fine-tune official LeRobot SmolVLA on the audited RoboTwin qpos16 NPZ data."""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from dataset import AUDIT  # noqa: E402
from robotwin_smol_data import (  # noqa: E402
    CAMERAS, RobotwinDataset, make_batch, sha,
)
from robotwin_smol_model import make  # noqa: E402
from smol_model import torch  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--steps', type=int, default=24000)
    p.add_argument('--batch-size', type=int, default=56)
    p.add_argument('--micro-batch', type=int, default=56)
    p.add_argument('--warmup-steps', type=int, default=1000)
    p.add_argument('--validation-every', type=int, default=500)
    p.add_argument('--validation-windows', type=int, default=128)
    p.add_argument('--save-at', default='8000,16000,24000')
    p.add_argument('--save-every', type=int, default=8000)
    p.add_argument('--report-every', type=int, default=100)
    p.add_argument('--seed', type=int, default=191)
    p.add_argument('--task-prompt', default='stack the red block and then the green block.')
    p.add_argument('--advantage', choices=('positive', 'negative', 'none'), default='none')
    p.add_argument('--smoke-test', action='store_true',
                   help='run one forward/backward/optimizer step without saving or starting training')
    p.add_argument('--smoke-batch-size', type=int, default=1)
    return p.parse_args()


def language_with_advantage(language, condition):
    if condition == 'none':
        return language
    return f'{language}. Advantage: {condition}'


def get_batch(data, indices, language, device):
    batch = make_batch(data, indices, torch, device)
    batch['task'] = [language] * len(indices)
    return batch


def validate(model, data, indices, language, device, count, seed):
    rng = np.random.default_rng(seed)
    picks = rng.choice(indices, size=min(count, len(indices)), replace=False)
    model.eval()
    losses = []
    with torch.inference_mode():
        for start in range(0, len(picks), 8):
            batch_ids = picks[start:start + 8]
            batch = get_batch(data, batch_ids, language, device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                loss, _ = model(batch)
            losses.append(float(loss))
    model.train()
    return float(np.mean(losses))


def checkpoint_payload(model, data, step, args, language, base_sha):
    trainable_parameters = {name for name, parameter in model.named_parameters()
                            if parameter.requires_grad}
    trainable = {name: value.detach().cpu().contiguous()
                 for name, value in model.state_dict().items() if name in trainable_parameters}
    return {
        'platform': 'robotwin', 'step': step, 'state_dim': 16, 'action_dim': 16,
        'trainable': trainable, 'stats': data.statistics(torch),
        'config': {
            'contract': data.contract,
            'base_model_sha256': base_sha,
            'language': language,
            'advantage': args.advantage,
            'chunk_size': 16, 'n_action_steps': 8, 'fps': 25,
            'optimizer': 'official_lerobot_smolvla_adamw_cosine_warmup',
            'physical_batch_size': args.micro_batch,
            'batch_size': args.batch_size, 'micro_batch': args.micro_batch,
            'gradient_accumulation': args.batch_size // args.micro_batch,
            'steps': args.steps, 'warmup_steps': args.warmup_steps,
        },
        'trainable_names': sorted(trainable),
        'base_model': str(AUDIT / 'followup_runs/models/smolvla/model.safetensors'),
    }


def save_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def main():
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.micro_batch < 1:
        raise ValueError('steps and batch sizes must be positive')
    if args.batch_size % args.micro_batch:
        raise ValueError('batch-size must be divisible by micro-batch')
    if not args.smoke_test and (args.batch_size != 56 or args.micro_batch != 56):
        raise ValueError('formal training requires physical batch 56 and effective batch 56; set --batch-size 56 --micro-batch 56')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    data = RobotwinDataset(args.data)
    train_ids, val_ids = data.indices('train'), data.indices('validation')
    assert len(train_ids) and len(val_ids)
    language = language_with_advantage(args.task_prompt, args.advantage)
    stats = data.statistics(torch)
    model = make(stats).to(device)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError('official SmolVLA config produced no trainable parameters')
    optimizer_cfg = model.config.get_optimizer_preset()
    optimizer = optimizer_cfg.build(trainable)
    scheduler_cfg = model.config.get_scheduler_preset()
    scheduler_cfg.num_warmup_steps = args.warmup_steps
    scheduler = scheduler_cfg.build(optimizer, args.steps)
    base_path = AUDIT / 'followup_runs/models/smolvla/model.safetensors'
    base_sha = sha(base_path)

    if args.smoke_test:
        count = min(args.smoke_batch_size, len(train_ids))
        starts = np.random.choice(train_ids, size=count, replace=False)
        batch = get_batch(data, starts, language, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            loss, metrics = model(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'non-finite smoke loss: {loss.item()}')
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, optimizer_cfg.grad_clip_norm)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f'non-finite smoke gradient norm: {grad_norm}')
        optimizer.step()
        print(json.dumps({
            'smoke_test': 'passed', 'optimizer_step': 'in-memory only',
            'checkpoint_written': False, 'batch_size': count,
            'physical_batch_size': count,
            'loss': float(metrics['loss']), 'grad_norm': float(grad_norm),
            'trainable_tensors': len(trainable), 'train_frames': len(train_ids),
            'validation_frames': len(val_ids), 'language': language,
            'contract': data.contract['version'],
        }, ensure_ascii=False), flush=True)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    save_steps = {int(x) for x in args.save_at.split(',') if x.strip()}
    if args.save_every:
        save_steps.update(range(args.save_every, args.steps + 1, args.save_every))
    save_steps.add(args.steps)
    config = {
        'platform': 'robotwin', 'contract': data.contract,
        'data_root': str(data.source), 'data_fingerprint': data.fingerprint,
        'frames': len(data.state), 'train_frames': len(train_ids),
        'validation_frames': len(val_ids), 'batch_size': args.batch_size,
        'physical_batch_size': args.micro_batch,
        'micro_batch': args.micro_batch,
        'gradient_accumulation': args.batch_size // args.micro_batch,
        'steps': args.steps, 'warmup_steps': args.warmup_steps,
        'optimizer': 'official SmolVLA AdamW preset', 'lr': model.config.optimizer_lr,
        'scheduler': 'official SmolVLA cosine decay with warmup preset',
        'scheduler_decay_steps': scheduler_cfg.num_decay_steps,
        'validation_every': args.validation_every,
        'validation_windows': args.validation_windows,
        'save_steps': sorted(save_steps), 'device': str(device),
        'precision': 'bf16 autocast' if device.type == 'cuda' else 'fp32',
        'advantage': args.advantage, 'language': language,
        'base_model_sha256': base_sha,
    }
    (args.output / 'training_config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False))
    progress = args.output / 'training_progress.jsonl'
    progress_md = args.output / '训练进度.md'
    rng = np.random.default_rng(args.seed)
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    running = []
    progress_md.write_text(
        '# SmolVLA 成功数据微调\n\n'
        f'- 状态：训练中\n- 数据：`{data.source}`\n'
        f'- 训练设置：物理 batch {args.micro_batch}，有效 batch {args.batch_size}，'
        f'梯度累积 {args.batch_size // args.micro_batch}\n'
        f'- 总步数：{args.steps}\n- 保存步数：{", ".join(map(str, sorted(save_steps)))}\n'
        f'- 提示词：`{language}`\n- 进度、loss 和 ETA 每 {args.report_every} 步更新。\n',
        encoding='utf-8')
    for step in range(1, args.steps + 1):
        accumulation = args.batch_size // args.micro_batch
        step_losses = []
        for _ in range(accumulation):
            chosen = rng.choice(train_ids, size=args.micro_batch, replace=False)
            batch = get_batch(data, chosen, language, device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                loss, metrics = model(batch)
                scaled_loss = loss / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite loss at step {step}: {loss.item()}')
            scaled_loss.backward()
            step_losses.append(float(metrics['loss']))
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, optimizer_cfg.grad_clip_norm)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f'non-finite gradient at step {step}')
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        running.append(float(np.mean(step_losses)))
        lr = optimizer.param_groups[0]['lr']

        if step % args.validation_every == 0 or step == 1:
            val_loss = validate(model, data, val_ids, language, device,
                                args.validation_windows, args.seed + step)
        else:
            val_loss = None
        if step % args.report_every == 0 or step == 1 or step in save_steps:
            elapsed = time.time() - start_time
            eta = elapsed / step * (args.steps - step)
            row = {'step': step, 'train_loss': float(np.mean(running)),
                   'validation_loss': val_loss, 'lr': lr,
                   'elapsed_seconds': elapsed, 'eta_seconds': eta}
            with progress.open('a') as f:
                f.write(json.dumps(row) + '\n')
            progress_md.write_text(
                '# SmolVLA 成功数据微调\n\n'
                f'- 状态：训练中（{step}/{args.steps}，{step / args.steps:.1%}）\n'
                f'- 数据：`{data.source}`\n'
                f'- 训练设置：物理 batch {args.micro_batch}，有效 batch {args.batch_size}，'
                f'梯度累积 {args.batch_size // args.micro_batch}\n'
                f'- 当前训练 loss：{row["train_loss"]:.6f}\n'
                f'- 最近验证 loss：{val_loss if val_loss is not None else "本次未验证"}\n'
                f'- 学习率：{lr:.8g}\n- 已用：{elapsed / 3600:.2f} 小时\n'
                f'- 按当前速度预计剩余：{eta / 3600:.2f} 小时\n'
                f'- 保存步数：{", ".join(map(str, sorted(save_steps)))}\n'
                f'- 提示词：`{language}`\n', encoding='utf-8')
            print(json.dumps(row), flush=True)
            running.clear()
        if step in save_steps:
            payload = checkpoint_payload(model, data, step, args, language, base_sha)
            save_checkpoint(args.output / f'checkpoint_{step:06d}.pt', payload)
            print(json.dumps({'checkpoint_saved': step}), flush=True)

    (args.output / 'status.json').write_text(json.dumps(
        {'status': 'completed', 'step': args.steps, 'elapsed_seconds': time.time() - start_time}, indent=2))
    progress_md.write_text(
        '# SmolVLA 成功数据微调\n\n'
        f'- 状态：已完成（{args.steps}/{args.steps}）\n'
        f'- 训练设置：物理 batch {args.micro_batch}，有效 batch {args.batch_size}，'
        f'梯度累积 {args.batch_size // args.micro_batch}\n'
        f'- 数据：`{data.source}`\n'
        f'- 检查点：{", ".join(map(str, sorted(save_steps)))}\n'
        f'- 总耗时：{(time.time() - start_time) / 3600:.2f} 小时\n'
        f'- 提示词：`{language}`\n', encoding='utf-8')


if __name__ == '__main__':
    main()
