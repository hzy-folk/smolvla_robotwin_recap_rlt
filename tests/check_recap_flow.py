"""CPU integration check: actual adapters/CLI stages, synthetic episodes, no VLM."""
import json
import argparse
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'recap'), str(ROOT/'collection')]
from initial_sft_data import InitialSFTDataset
from final_policy_data import FinalPolicyDataset, make_final_batch
from robotwin_gripper_action import inward_targets, TIMING_RULE
from recap_config import RecapConfig
from recap_rewards import reward_sequence
from label_recap_advantage import advantages
from run_pipeline import commands, verify_stage, STAGES


def run(script, *args):
    subprocess.run([sys.executable, str(ROOT/'recap'/script), *map(str, args)], check=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--with-policy', action='store_true', help='run bounded real SmolVLA updates')
    parser.add_argument('--policy-device', choices=('cpu','cuda'), default='cpu')
    args = parser.parse_args()
    with TemporaryDirectory(prefix='recap-flow-') as folder:
        base = Path(folder)
        success, failure, rollout = (base/x for x in ('success', 'failure', 'rollout'))
        frames = 4
        cameras = ('images', 'left_wrist', 'right_wrist')
        for root, ok in ((success, True), (failure, False)):
            root.mkdir()
            entries = []
            for i, split in enumerate(('train', 'validation')):
                state = np.full((frames, 16), 0.025, dtype=np.float32)
                state[:, 0] = np.arange(frames)*0.01
                command = np.zeros((frames, 14), dtype=np.float32)
                action, _, _ = inward_targets(state, command)
                images = {key: np.full((frames, 256, 256, 3), 30+i*40+c*15, np.uint8)
                          for c, key in enumerate(cameras)}
                name = f'episode_{i:03d}'
                np.savez(root/f'{name}.npz', state=state, action=action, diagnostic_command14=command,
                         success=ok, official_success=np.array([False]*(frames-1)+[ok]),
                         sample_hz=25, timestamp=np.arange(frames)/25, task='stack the blocks',
                         joint_names=np.array([f'joint_{j}' for j in range(16)]),
                         joint_units=np.array(['rad']*16), **images)
                entries.append(dict(id=name, npz=f'{name}.npz', frames=frames, split=split,
                                    task_success=ok, intentional_failure=not ok))
            (root/'manifest.json').write_text(json.dumps(dict(episodes=entries,
                derivation={'timing_rule': TIMING_RULE})))
        initial = InitialSFTDataset(success, base/'initial_cache')
        assert len(initial.indices('train')) == len(initial.indices('validation')) == frames
        assert initial.language.endswith('Advantage: positive')
        rows = []
        for i, ok in enumerate((True, False)):
            episode = rollout/'rollouts'/f'episode_{i:03d}'
            episode.mkdir(parents=True)
            applied = np.full((frames-1, 16), 0.1+i*0.1, dtype=np.float32)
            np.savez(episode/'rollout.npz', state=state, applied_targets=applied)
            for camera in ('head', 'left_wrist', 'right_wrist'):
                writer = cv2.VideoWriter(str(episode/f'{camera}.mp4'),
                    cv2.VideoWriter_fourcc(*'mp4v'), 25, (256, 256))
                assert writer.isOpened()
                for t in range(frames):
                    writer.write(np.full((256, 256, 3), 70+t*10, np.uint8))
                writer.release()
            rows.append(dict(id=episode.name, official_success=ok, completed=True))
        (rollout/'summary.json').write_text(json.dumps(dict(status='completed', finished=2,
                                                           total=2, episodes=rows)))
        prepared, value = base/'value_data', base/'value'
        value.mkdir()
        run('prepare_recap_value_dataset.py', '--success', success, '--failure', failure,
            '--rollout', rollout, '--output', prepared)
        meta = json.loads((prepared/'manifest.json').read_text())
        assert len(meta['episodes']) == 6 and meta['samples'] == 24
        offsets = np.load(prepared/'image_offsets.npy')
        with (prepared/'images.jpgbin').open('rb') as stream:
            for offset, length in offsets.reshape(-1, 2):
                stream.seek(int(offset))
                image = cv2.imdecode(np.frombuffer(stream.read(int(length)), np.uint8), cv2.IMREAD_COLOR)
                assert image is not None and image.shape == (224, 224, 3)
        # Deliberately use toy predictions: this checks joins/TD targets, not PiStar training.
        predictions = {f"{e['source']}__{e['id']}": [-0.4, -0.3, -0.2, -0.1]
                       for e in meta['episodes']}
        (value/'predicted_values.json').write_text(json.dumps(predictions))
        run('label_recap_advantage.py', '--value-data', prepared, '--values', value/'predicted_values.json',
            '--output', value/'advantages.json')
        data = FinalPolicyDataset(value, base/'final_cache', value_data=prepared)
        assert len(data.entries) == 6 and len(data.indices('train')) == 16
        ids = [e['start'] for e in data.entries]
        batch = make_final_batch(data, ids, torch, 'cpu')
        assert batch['action'].shape == (6, 16, 16)
        for key in ('observation.images.camera1', 'observation.images.camera2', 'observation.images.camera3'):
            assert batch[key].shape == (6, 3, 256, 256)
        assert all('Advantage: ' in task for task in batch['task'])
        for e in data.entries:
            indices = data.target_indices(np.array([e['start']]))[0]
            assert indices.max() < e['stop']
            if e['source'].startswith('rollout'):
                with np.load(e['npz']) as original:
                    np.testing.assert_array_equal(data.action[indices[0]], original['applied_targets'][0])
                    np.testing.assert_array_equal(data.action[indices[-1]], original['applied_targets'][-1])
        restored = FinalPolicyDataset(value, base/'final_cache', value_data=prepared)
        np.testing.assert_array_equal(restored.action, data.action)
        # Terminal bootstrap must be zero; validation frames must not enter training statistics.
        assert advantages([1.0], [0.25], 8) == [0.75]
        train = np.asarray(data.state[data.indices('train')])
        np.testing.assert_allclose(data.statistics(torch)['observation.state']['mean'], train.mean(0))
        for ds in (data, restored):
            ds.stream.close()
        config = json.loads((ROOT/'recap/configs/pipeline.example.json').read_text())
        plan = commands(config)
        assert tuple(plan) == STAGES and '--value-data' in plan['final_sft']
        assert plan['rollout'][plan['rollout'].index('--split')+1] == 'train'
        assert plan['evaluate'][plan['evaluate'].index('--split')+1] == 'validation'
        initialized = commands({**config, 'value_init_checkpoint': '/existing/value/step_00040000'})
        assert '--init-checkpoint' in initialized['train_value']
        assert '--siglip-checkpoint' not in initialized['train_value']
        if args.with_policy:
            common = ['--device', args.policy_device, '--steps','2','--batch-size','1','--micro-batch','1',
                      '--warmup-steps','0','--validation-windows','1',
                      '--validation-every','1','--save-every','2','--report-every','1']
            run('train_recap_initial_sft.py', '--data', success, '--cache', base/'initial_cache',
                '--output', base/'initial_train', '--pause-at', '1', *common)
            saved = torch.load(base/'initial_train/checkpoint_000001.pt', map_location='cpu', weights_only=False)
            assert saved['step'] == 1 and saved['optimizer'] and saved['sampler']
            del saved
            try:
                verify_stage('initial_sft', {'initial_sft':['--output',str(base/'initial_train'),'--steps','2']})
            except RuntimeError:
                pass
            else:
                raise AssertionError('paused training was accepted as complete')
            run('train_recap_initial_sft.py', '--data', success, '--cache', base/'initial_cache',
                '--output', base/'initial_train', '--resume', *common)
            verify_stage('initial_sft', {'initial_sft':['--output',str(base/'initial_train'),'--steps','2']})
            run('train_recap_policy.py', '--data', value, '--value-data', prepared,
                '--cache', base/'final_cache', '--output', base/'final_train', *common)
            verify_stage('final_sft', {'final_sft':['--output',str(base/'final_train'),'--steps','2']})
        # A newly collected dataset may have recorded actions without the old derivation marker.
        for root in (success, failure):
            manifest = json.loads((root/'manifest.json').read_text())
            manifest.pop('derivation')
            for entry in manifest['episodes']:
                entry.pop('intentional_failure', None)
            (root/'manifest.json').write_text(json.dumps(manifest))
        raw_initial = InitialSFTDataset(success, base/'raw_initial_cache')
        raw_final = FinalPolicyDataset(value, base/'raw_final_cache', value_data=prepared)
        assert raw_initial.contract['version'].endswith('recorded_action_v1')
        assert raw_final.contract == raw_initial.contract
        with np.load(success/'episode_000.npz') as original:
            np.testing.assert_array_equal(raw_initial.action[:frames], original['action'])
            np.testing.assert_array_equal(raw_final.action[:frames], original['action'])
        raw_final.stream.close()
        # Recalculate all retained historical labels as an independent recovery check.
        historical = ROOT/'recap_runs/value_round2_pistar_batch8_40000'
        if (historical/'predicted_values.json').exists() and (historical/'advantages.json').exists():
            predicted = json.loads((historical/'predicted_values.json').read_text())
            labels = json.loads((historical/'advantages.json').read_text())
            for row in labels['rows']:
                values = predicted[row['key']]
                expected = advantages(reward_sequence(len(values), row['source'].endswith('_success'),
                                                       RecapConfig()), values, 8)
                np.testing.assert_allclose(expected, row['advantage'], atol=1e-12, rtol=0)
                assert ['positive' if x >= labels['threshold'] else 'negative' for x in expected] == row['adv_ind']
        print(json.dumps({'status': 'passed', 'episodes': 6, 'frames': 24,
                          'value_images': 'JPEG decode tested', 'policy_batch': [6, 16, 16],
                          'rollout_alignment': 'passed', 'historical_labels': 'passed if present',
                          'real_smolvla_training': args.with_policy, 'real_pistar_training': False, 'real_simulator_rollout': False}))


if __name__ == '__main__':
    main()
