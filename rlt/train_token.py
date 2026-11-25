"""阶段一：训练 RL token。

冻结 SmolVLA，只训练一个外挂的编码器-解码器，目标是让它用一个（或几个）
向量把 prefix 表示重建出来。整个训练是自监督的，不需要动作标签，也不需要
奖励，因此这一步可以脱离仿真环境单独跑。

用法（在项目根目录执行）：

    python rlt/train_token.py --config rlt/configs/robotwin_piper.yaml \
        --checkpoint runs/<平台>/<检查点>.pt --dataset <数据集目录> --steps 5000

自检时可以不加载任何权重：

    python rlt/train_token.py --synthetic --steps 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlt import bootstrap  # noqa: E402

bootstrap.bootstrap()

import torch  # noqa: E402

from rlt.config import load_config  # noqa: E402
from rlt.model import RLTokenModel, prefix_and_actions, select_prefix  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 RL token（阶段一）")
    parser.add_argument("--config", default=None, help="RLT 的 YAML 配置")
    parser.add_argument("--checkpoint", default=None, help="SmolVLA 微调检查点，,.pt")
    parser.add_argument("--dataset", default=None, help="RobotwinDataset 的数据目录")
    parser.add_argument("--split", default="train", choices=("train", "validation"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=None, help="遍历完整训练集指定轮数；真实数据模式使用 prefix 缓存")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None, help="输出检查点路径")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--checkpoint-count", type=int, default=3,
                        help="完整数据集训练时均匀保存的检查点数")
    parser.add_argument("--cache-prefixes", action="store_true",
                        help="先缓存 train/validation 的冻结 VLA prefix，再训练 token")
    parser.add_argument("--delete-prefix-cache", action="store_true",
                        help="训练成功后删除 prefix 缓存；默认保留")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--synthetic", action="store_true", help="用随机张量自检，不加载模型")
    parser.add_argument("--dim", type=int, default=960, help="synthetic 模式下的 prefix 维度")
    return parser.parse_args()


def synthetic_batch(batch_size: int, length: int, dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = torch.randn(batch_size, length, dim, device=device)
    mask = torch.ones(batch_size, length, dtype=torch.bool, device=device)
    return prefix, mask


def build_dataset_batch(dataset, indices, language: str, device: torch.device) -> dict:
    from robotwin_smol_data import CAMERAS, observation_batch

    images = [dataset.images[key][indices] for key in CAMERAS]
    return observation_batch(*images, dataset.state[indices], language, torch, device)


def cache_prefixes(policy, dataset, frames, config, language: str, device: torch.device,
                   batch_size: int, cache_path: Path, mask_path: Path, label: str):
    prefix_cache = None
    mask_cache = None
    for start in range(0, len(frames), batch_size):
        batch_indices = frames[start : start + batch_size]
        batch = build_dataset_batch(dataset, batch_indices, language, device)
        with torch.no_grad():
            result = prefix_and_actions(policy, batch, with_actions=False)
        selected, selected_mask = select_prefix(
            result.prefix, result.mask, result.image_tokens, config.token.z_source
        )
        if prefix_cache is None:
            cache_shape = (len(frames), selected.shape[1], selected.shape[2])
            prefix_cache = np.lib.format.open_memmap(
                cache_path, mode="w+", dtype=np.float16, shape=cache_shape
            )
            mask_cache = np.lib.format.open_memmap(
                mask_path, mode="w+", dtype=np.bool_, shape=(len(frames), selected.shape[1])
            )
            print(f"[cache:{label}] shape={cache_shape} dtype=float16", flush=True)
        prefix_cache[start : start + len(batch_indices)] = selected.float().cpu().numpy().astype(np.float16)
        mask_cache[start : start + len(batch_indices)] = selected_mask.cpu().numpy()
        if (start // batch_size + 1) % 100 == 0 or start + len(batch_indices) == len(frames):
            print(f"[cache:{label}] {start + len(batch_indices)}/{len(frames)}", flush=True)
    prefix_cache.flush()
    mask_cache.flush()
    del prefix_cache, mask_cache
    return np.load(cache_path, mmap_mode="r"), np.load(mask_path, mmap_mode="r")


@torch.no_grad()
def evaluate_reconstruction(model, prefix_cache, mask_cache, batch_size: int,
                            device: torch.device) -> float:
    was_training = model.training
    model.eval()
    weighted_loss = 0.0
    total_rows = 0
    for start in range(0, len(prefix_cache), batch_size):
        end = min(start + batch_size, len(prefix_cache))
        prefix = torch.as_tensor(np.array(prefix_cache[start:end]), device=device, dtype=torch.float32)
        mask = torch.as_tensor(np.array(mask_cache[start:end]), device=device, dtype=torch.bool)
        weighted_loss += float(model.reconstruction_loss(prefix, mask)) * (end - start)
        total_rows += end - start
    model.train(was_training)
    return weighted_loss / max(total_rows, 1)


def save_checkpoint(path: Path, model: RLTokenModel, config, step: int, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "dim": model.dim,
        "embed_dim": model.embed_dim,
        "num_tokens": model.num_tokens,
        "token": vars(config.token),
        "data": vars(config.data),
        "step": step,
        "extra": extra or {},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device or config.runtime.device)

    assert args.synthetic or (args.checkpoint and args.dataset), "真实训练需要 --checkpoint 与 --dataset"
    assert args.epochs is None or (args.epochs > 0 and not args.synthetic), "--epochs 只支持真实数据且必须为正数"
    assert args.epochs is None or args.split == "train", "完整遍历训练只允许使用 train 划分，validation 保持留出"
    assert not args.cache_prefixes or (not args.synthetic and args.split == "train"), \
        "--cache-prefixes 需要真实数据并使用 train 划分"

    policy = None
    dataset = None
    frames = None
    if args.synthetic:
        dim = args.dim
    else:
        from rlt.model import load_policy, prefix_dim
        from robotwin_smol_data import RobotwinDataset

        policy = load_policy(args.checkpoint, device=str(device))
        dim = prefix_dim(policy)
        dataset = RobotwinDataset(args.dataset)
        frames = dataset.indices("train" if args.split == "train" else "validation")
        assert len(frames) > 0, "数据划分是空的"

    model = RLTokenModel(
        dim=dim,
        embed_dim=config.token.embed_dim,
        num_tokens=config.token.num_tokens,
        layers=config.token.num_layers,
        heads=config.token.num_heads,
        ratio=config.token.mlp_ratio,
        dropout=config.token.dropout,
    )

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    start_step = 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        assert saved["dim"] == model.dim, "检查点维度与当前模型不一致"
        assert int(saved.get("embed_dim", model.embed_dim)) == model.embed_dim, "检查点内部维度与当前模型不一致"
        model.load_state_dict(saved["state_dict"])
        start_step = int(saved.get("step", 0))
        print(f"[resume] {args.resume} step={start_step}")

    out = Path(args.out) if args.out else config.out_dir() / "rlt_token.pt"
    generator = torch.Generator(device="cpu").manual_seed(config.runtime.seed)
    language = config.data.language if config.data.language else getattr(dataset, "language", "")
    prefix_cache = None
    mask_cache = None
    validation_cache = None
    validation_mask = None
    validation_cache_path = None
    validation_mask_path = None
    validation_metrics = []
    cache_dir = None
    if args.epochs is not None or args.cache_prefixes:
        cache_path = out.with_suffix(".prefix_cache.npy")
        mask_path = out.with_suffix(".prefix_mask.npy")
        validation_frames = dataset.indices("validation")
        validation_cache_path = out.with_name(out.stem + ".validation_prefix_cache.npy")
        validation_mask_path = out.with_name(out.stem + ".validation_prefix_mask.npy")
        cache_dir = out.parent / "prefix_cache"
        if args.cache_prefixes:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / cache_path.name
            mask_path = cache_dir / mask_path.name
            validation_cache_path = cache_dir / validation_cache_path.name
            validation_mask_path = cache_dir / validation_mask_path.name
        cache_paths = (cache_path, mask_path, validation_cache_path, validation_mask_path)
        existing_caches = [path.exists() for path in cache_paths]
        if all(existing_caches):
            print(f"[cache] reusing frozen prefix caches from {cache_dir}", flush=True)
            prefix_cache = np.load(cache_path, mmap_mode="r")
            mask_cache = np.load(mask_path, mmap_mode="r")
            validation_cache = np.load(validation_cache_path, mmap_mode="r")
            validation_mask = np.load(validation_mask_path, mmap_mode="r")
            assert len(prefix_cache) == len(frames), "训练 prefix 缓存长度与数据集不符"
            assert len(validation_cache) == len(validation_frames), "验证 prefix 缓存长度与数据集不符"
        else:
            if any(existing_caches):
                raise FileExistsError(f"incomplete prefix cache set exists: {cache_paths}")
            print(f"[cache] extracting frozen VLA prefix for {len(frames)} train frames", flush=True)
            prefix_cache, mask_cache = cache_prefixes(
                policy, dataset, frames, config, language, device, args.batch_size,
                cache_path, mask_path, "train",
            )
            print(f"[cache] extracting held-out prefix for {len(validation_frames)} validation frames", flush=True)
            validation_cache, validation_mask = cache_prefixes(
                policy, dataset, validation_frames, config, language, device, args.batch_size,
                validation_cache_path, validation_mask_path, "validation",
            )
        initial_validation_loss = evaluate_reconstruction(
            model, validation_cache, validation_mask, args.batch_size, device
        )
        validation_metrics.append({
            "step": start_step,
            "progress": 0.0,
            "validation_reconstruction_mse": initial_validation_loss,
        })
        print(f"[validation] initial_mse={initial_validation_loss:.6f}", flush=True)
        del policy
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.epochs is not None:
        steps_per_epoch = (len(frames) + args.batch_size - 1) // args.batch_size
        total_steps = steps_per_epoch * args.epochs
        assert args.checkpoint_count > 0, "--checkpoint-count 必须为正数"
        checkpoint_steps = sorted({
            max(1, (total_steps * index + args.checkpoint_count - 1) // args.checkpoint_count)
            for index in range(1, args.checkpoint_count + 1)
        })
        checkpoint_index = 0
    else:
        total_steps = args.steps
    print(
        f"[train] dim={model.dim} embed_dim={model.embed_dim} tokens={model.num_tokens} "
        f"device={device} batch_size={args.batch_size} steps={total_steps} "
        f"epochs={args.epochs if args.epochs is not None else 'step-limited'}",
        flush=True,
    )

    model.train()
    started = time.time()
    running = 0.0
    global_step = start_step
    if args.epochs is not None:
        for epoch in range(args.epochs):
            order = torch.randperm(len(frames), generator=generator).numpy()
            for start in range(0, len(order), args.batch_size):
                row_ids = order[start : start + args.batch_size]
                prefix = torch.as_tensor(np.array(prefix_cache[row_ids]), device=device, dtype=torch.float32)
                mask = torch.as_tensor(np.array(mask_cache[row_ids]), device=device, dtype=torch.bool)
                loss = model.reconstruction_loss(prefix, mask)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                global_step += 1
                running += float(loss.detach())

                if global_step % args.log_every == 0:
                    average = running / args.log_every
                    elapsed = time.time() - started
                    print(
                        f"[step {global_step}/{total_steps}] epoch={epoch + 1}/{args.epochs} "
                        f"loss={average:.4f} elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                    running = 0.0

                if global_step in checkpoint_steps:
                    validation_loss = evaluate_reconstruction(
                        model, validation_cache, validation_mask, args.batch_size, device
                    )
                    checkpoint_index += 1
                    progress = global_step / total_steps
                    extra = {
                        "epochs": args.epochs,
                        "train_frames": len(frames),
                        "frame_visits": len(frames) * args.epochs,
                        "batch_size": args.batch_size,
                        "learning_rate": args.learning_rate,
                        "validation_reconstruction_mse": validation_loss,
                        "validation_frames": len(validation_cache),
                        "progress": progress,
                        "checkpoint_index": checkpoint_index,
                        "checkpoint_count": len(checkpoint_steps),
                    }
                    checkpoint_path = out.with_name(
                        f"{out.stem}_progress_{checkpoint_index}of{len(checkpoint_steps)}{out.suffix}"
                    )
                    save_checkpoint(checkpoint_path, model, config, global_step, extra=extra)
                    validation_metrics.append({
                        "step": global_step,
                        "progress": progress,
                        "validation_reconstruction_mse": validation_loss,
                    })
                    print(
                        f"[save] {checkpoint_path} validation_mse={validation_loss:.6f} "
                        f"progress={progress:.3f}", flush=True
                    )
            print(f"[epoch {epoch + 1}/{args.epochs}] complete", flush=True)
    else:
        for step in range(start_step, args.steps):
            if args.synthetic:
                prefix, mask = synthetic_batch(args.batch_size, 97, model.dim, device)
                image_tokens = 48
            elif prefix_cache is not None:
                indices = torch.randint(
                    0, len(frames), (args.batch_size,), generator=generator
                ).numpy()
                prefix = torch.as_tensor(
                    np.array(prefix_cache[indices]), device=device, dtype=torch.float32
                )
                mask = torch.as_tensor(
                    np.array(mask_cache[indices]), device=device, dtype=torch.bool
                )
                image_tokens = prefix.shape[1]
            else:
                indices = torch.randint(
                    0, len(frames), (args.batch_size,), generator=generator
                ).numpy()
                batch = build_dataset_batch(dataset, frames[indices], language, device)
                with torch.no_grad():
                    result = prefix_and_actions(policy, batch, with_actions=False)
                prefix, mask = select_prefix(
                    result.prefix, result.mask, result.image_tokens, config.token.z_source
                )
                prefix = prefix.float()
                image_tokens = result.image_tokens

            loss = model.reconstruction_loss(prefix, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            global_step += 1

            running += float(loss.detach())
            if global_step % args.log_every == 0:
                average = running / args.log_every
                elapsed = time.time() - started
                print(
                    f"[step {global_step}/{total_steps}] loss={average:.4f} "
                    f"tokens={image_tokens} elapsed={elapsed:.1f}s",
                    flush=True,
                )
                running = 0.0

            if global_step % args.save_every == 0:
                validation_loss = None
                if validation_cache is not None:
                    validation_loss = evaluate_reconstruction(
                        model, validation_cache, validation_mask, args.batch_size, device
                    )
                    validation_metrics.append({
                        "step": global_step,
                        "progress": global_step / args.steps,
                        "validation_reconstruction_mse": validation_loss,
                    })
                if args.cache_prefixes:
                    checkpoint_path = out.with_name(
                        f"{out.stem}_step_{global_step:06d}{out.suffix}"
                    )
                    extra = {
                        "validation_reconstruction_mse": validation_loss,
                        "validation_frames": len(validation_cache),
                        "batch_size": args.batch_size,
                        "learning_rate": args.learning_rate,
                        "resume_from": args.resume,
                        "target_total_steps": args.steps,
                    }
                    save_checkpoint(checkpoint_path, model, config, global_step, extra=extra)
                else:
                    checkpoint_path = out
                    save_checkpoint(checkpoint_path, model, config, global_step)
                print(
                    f"[save] {checkpoint_path}"
                    + (f" validation_mse={validation_loss:.6f}" if validation_loss is not None else ""),
                    flush=True,
                )

    final_out = out
    if args.epochs is not None:
        final_out = out.with_name(
            f"{out.stem}_progress_{len(checkpoint_steps)}of{len(checkpoint_steps)}{out.suffix}"
        )
    elif args.cache_prefixes and args.steps % args.save_every == 0:
        final_out = out.with_name(f"{out.stem}_step_{args.steps:06d}{out.suffix}")
    else:
        save_checkpoint(out, model, config, global_step, extra={
            "epochs": args.epochs,
            "train_frames": len(frames) if frames is not None else None,
            "frame_visits": len(frames) * args.epochs if args.epochs is not None else None,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        })
    print(f"[save] {final_out}", flush=True)

    summary = {
        "out": str(final_out),
        "dim": model.dim,
        "steps": global_step,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_frames": len(frames) if frames is not None else None,
        "frame_visits": len(frames) * args.epochs if args.epochs is not None else None,
        "synthetic": bool(args.synthetic),
        "validation_frames": len(validation_cache) if validation_cache is not None else None,
        "validation_metrics": validation_metrics,
        "validation_relative_improvement": (
            1.0 - validation_metrics[-1]["validation_reconstruction_mse"] / validation_metrics[0]["validation_reconstruction_mse"]
            if len(validation_metrics) > 1 and validation_metrics[0]["validation_reconstruction_mse"] > 0
            else None
        ),
    }
    summary_path = out.with_name(out.stem + "_summary.json") if args.cache_prefixes else out.parent / "train_token_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    if args.delete_prefix_cache and (args.epochs is not None or args.cache_prefixes):
        cache_path.unlink(missing_ok=True)
        mask_path.unlink(missing_ok=True)
        validation_cache_path.unlink(missing_ok=True)
        validation_mask_path.unlink(missing_ok=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
