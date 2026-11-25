"""Model-only subprocess: JSON protocol, current RGB/qpos/language inputs only."""
import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--flow-steps', type=int, default=4)
    p.add_argument('--rlt-token')
    p.add_argument('--rlt-config')
    p.add_argument('--actor-snapshot')
    p.add_argument('--actor-deterministic', action='store_true', default=None)
    a = p.parse_args()
    protocol = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        from robotwin_smol_model import Policy, torch
        torch.set_num_threads(4); torch.manual_seed(300)
        policy = Policy(a.checkpoint, a.device, a.flow_steps)
        policy_cpu_rng = torch.get_rng_state()
        policy_cuda_rng = torch.cuda.get_rng_state_all() if a.device == 'cuda' else None
        token = None
        token_config = None
        actor_runtime = None
        normalizer = None
        if a.rlt_token:
            project_root = Path(__file__).resolve().parents[3]
            sys.path.insert(0, str(project_root))
            from rlt import bootstrap
            bootstrap.bootstrap()
            from rlt.config import load_config
            from rlt.model import RLTokenModel
            config_path = a.rlt_config or str(project_root / 'rlt/configs/robotwin_piper.yaml')
            token_config = load_config(config_path)
            saved = torch.load(a.rlt_token, map_location='cpu', weights_only=False)
            token = RLTokenModel(
                dim=int(saved['dim']),
                embed_dim=int(saved.get('embed_dim', token_config.token.embed_dim)),
                num_tokens=int(saved['num_tokens']),
                layers=token_config.token.num_layers,
                heads=token_config.token.num_heads,
                ratio=token_config.token.mlp_ratio,
                dropout=token_config.token.dropout,
            )
            token.load_state_dict(saved['state_dict'])
            token = token.to(a.device).eval()
            assert token.z_dim == 512
        if a.actor_snapshot:
            if token is None:
                raise ValueError('--actor-snapshot requires --rlt-token')
            from rlt.online import ActorRuntime, ChunkFeature, build_normalizer
            actor_runtime = ActorRuntime(
                token.z_dim, token_config, Path(a.actor_snapshot), a.device,
                deterministic=(token_config.env.deterministic_actor if a.actor_deterministic is None
                               else a.actor_deterministic),
            )
            normalizer = build_normalizer(token_config)
            actor_runtime.maybe_reload()
            if actor_runtime.version < 1:
                raise RuntimeError(f'No trained actor snapshot loaded: {a.actor_snapshot}')
            actor_feature_type = ChunkFeature
        else:
            actor_feature_type = None

        # Token and actor initialization must not shift the base policy's noise stream.
        torch.set_rng_state(policy_cpu_rng)
        if policy_cuda_rng is not None:
            torch.cuda.set_rng_state_all(policy_cuda_rng)

        # Actor exploration has its own reproducible stream and cannot shift
        # the SmolVLA noise stream used by the next chunk.
        actor_cpu_rng = torch.Generator(device='cpu').manual_seed(300).get_state()
        actor_cuda_rng = (
            torch.Generator(device=torch.device(a.device)).manual_seed(300).get_state()
            if a.device == 'cuda' else None
        )

        def refine_without_advancing_policy_rng(feature):
            nonlocal actor_cpu_rng, actor_cuda_rng
            policy_cpu = torch.get_rng_state()
            policy_cuda = torch.cuda.get_rng_state_all() if a.device == 'cuda' else None
            torch.set_rng_state(actor_cpu_rng)
            if actor_cuda_rng is not None:
                torch.cuda.set_rng_state(actor_cuda_rng, device=torch.device(a.device))
            try:
                result, _source = actor_runtime.refine(feature, normalizer, use_actor=True)
                version = actor_runtime.version
                actor_cpu_rng = torch.get_rng_state()
                if actor_cuda_rng is not None:
                    actor_cuda_rng = torch.cuda.get_rng_state(device=torch.device(a.device))
            finally:
                torch.set_rng_state(policy_cpu)
                if policy_cuda is not None:
                    torch.cuda.set_rng_state_all(policy_cuda)
            return result, version
    print(json.dumps(dict(ready=True, checkpoint_step=policy.step, contract=policy.contract)), file=protocol, flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get('stop'):
            break
        try:
            with contextlib.redirect_stdout(sys.stderr), np.load(request['input'], allow_pickle=False) as d:
                prediction = policy.predict(d['images'], d['left_wrist'], d['right_wrist'], d['state'], request['language'])
                output = {'predicted_chunk': prediction}
                if token is not None:
                    from rlt.model import prefix_and_actions, select_prefix
                    from rlt.serve import observations_to_batch
                    observation = {
                        'observation.state': d['state'],
                        'observation.images.camera1': d['images'],
                        'observation.images.camera2': d['left_wrist'],
                        'observation.images.camera3': d['right_wrist'],
                        'task': request['language'],
                    }
                    batch = observations_to_batch([observation], token_config.data.language, torch.device(a.device))
                    with torch.inference_mode(), torch.autocast(
                        'cuda', dtype=torch.bfloat16, enabled=a.device == 'cuda'
                    ):
                        prefix = prefix_and_actions(policy, batch, with_actions=False)
                        selected, mask = select_prefix(
                            prefix.prefix, prefix.mask, prefix.image_tokens, token_config.token.z_source
                        )
                        z = token.encode(selected.float(), mask)[0].float().cpu().numpy()
                    output['z_rl'] = z
                    if actor_runtime is not None:
                        actor_runtime.maybe_reload()
                        feature = actor_feature_type(
                            z=z,
                            proprio=np.asarray(d['state'], dtype=np.float32),
                            reference=prediction[:token_config.data.chunk_len].copy(),
                        )
                        refined, actor_version = refine_without_advancing_policy_rng(feature)
                        executed_chunk = prediction.copy()
                        executed_chunk[:token_config.data.chunk_len] = refined
                        output['actor_chunk'] = executed_chunk
                        output['actor_version'] = np.asarray(actor_version, dtype=np.int64)
                np.savez(request['output'], **output)
            print(json.dumps(dict(ok=True, shape=list(prediction.shape), z_dim=token.z_dim if token else None,
                                  actor_version=actor_runtime.version if actor_runtime is not None else None)),
                  file=protocol, flush=True)
        except Exception as exc:
            print(json.dumps(dict(ok=False, error=repr(exc))), file=protocol, flush=True)


if __name__ == '__main__':
    main()
