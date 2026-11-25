"""Measured Piper qpos16: shared train/inference contract, independent of LIBERO."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from dataset import DATA_ROOT
from robotwin_gripper_action import inward_targets, TIMING_RULE

DEFAULT_DATA = DATA_ROOT / 'robotwin_stack_qpos25/dataset_qpos16_25hz_action_inward3mm_v3'
CAMERAS = ('images', 'left_wrist', 'right_wrist')
FEATURES = tuple(f'observation.images.camera{i}' for i in (1, 2, 3))
LEGACY_CONTRACT = dict(version='robotwin_piper_qpos16_future1_v1',
                state='left_arm6_rad_fingers2_m_right_arm6_rad_fingers2_m',
                target_offsets=list(range(1, 17)), chunk_size=16, n_action_steps=8,
                fps=25, physics_dt=.004, physics_steps_per_sample=10,
                std_floor=1e-3, tail='repeat_last_without_mask',
                loss='native_flow_velocity_mse_mean_32_no_dimension_mask',
                cameras=list(CAMERAS), camera_features=list(FEATURES),
                adapt_to_pi_aloha=False, use_delta_joint_actions_aloha=False)
V2_CONTRACT = dict(LEGACY_CONTRACT, version='robotwin_piper_qpos16_future_action_inward3mm_v2',
    action='absolute_joint_targets; each finger 0.003m inward on its own closed plateau',
    target_source='action', normalization='separate_train_only_state_and_action_statistics')
RECORDED_ACTION_CONTRACT = dict(LEGACY_CONTRACT,
    version='robotwin_piper_qpos16_recorded_action_v1',
    action='use absolute joint targets exactly as stored in each episode NPZ',
    target_source='action', normalization='separate_train_only_state_and_action_statistics')
CONTRACT = dict(V2_CONTRACT, version='robotwin_piper_qpos16_future_action_inward3mm_v3',
    action='absolute_joint_targets; each finger 0.003m inward in measured <=28mm grasp window',
    timing_rule=TIMING_RULE)


def future_action_chunk(action, start):
    """Action at t+1 ... t+16; repeat the final action without a time mask."""
    assert 0 <= start < len(action)
    return action[np.minimum(np.arange(start + 1, start + 17), len(action) - 1)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


class RobotwinDataset:
    def __init__(self, source=DEFAULT_DATA, cache=None):
        self.source = Path(source).resolve()
        manifest_path = self.source / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        assert manifest['status'] == 'complete_audited'
        derivation = manifest.get('derivation')
        if derivation:
            rule = derivation.get('timing_rule')
            assert rule in (None, TIMING_RULE), 'unknown action timing rule'
            self.contract = CONTRACT if rule == TIMING_RULE else V2_CONTRACT
        else:
            self.contract = RECORDED_ACTION_CONTRACT
        entries = manifest['episodes']
        assert len(entries) == 100
        assert len({e['id'] for e in entries}) == len({e['seed'] for e in entries}) == 100
        assert sum(e['split'] == 'train' for e in entries) == 80
        assert sum(e['split'] == 'validation' for e in entries) == 20
        for split in ('train', 'validation'):
            rows = json.loads((self.source / 'splits' / f'{split}.json').read_text())
            # Split files use a dictionary containing an episodes list.
            rows = rows['episodes'] if isinstance(rows, dict) else rows
            assert {e['id'] for e in rows} == {e['id'] for e in entries if e['split'] == split}
        audit = json.loads((self.source / 'correctness_audit.json').read_text())
        assert audit['all_pass'] and audit['total_frames'] == sum(e['frames'] for e in entries)
        audited_hashes = {row['id']: row['npz_sha256'] for row in audit['rows']}
        # Hash raw sources even on cache reuse. A changed source cannot silently
        # reuse a cache belonging to an older successful audit.
        raw_hashes = {}
        for e in entries:
            path = (self.source / e['npz']).resolve()
            assert path.is_relative_to(self.source)
            raw_hashes[e['id']] = sha(path)
            assert raw_hashes[e['id']] == audited_hashes[e['id']], e['id']
        self.fingerprint = hashlib.sha256(json.dumps(dict(
            manifest=sha(manifest_path), raw=raw_hashes, contract=self.contract), sort_keys=True).encode()).hexdigest()
        self.folder = Path(cache).resolve() if cache else self.source.with_name(self.source.name + '_smol_cache')
        self.folder.mkdir(parents=True, exist_ok=True)
        marker = self.folder / 'complete.json'
        required_cache = ['state.npy', 'action.npy', *[f'{key}.npy' for key in CAMERAS]]
        valid = (marker.exists()
                 and json.loads(marker.read_text()).get('fingerprint') == self.fingerprint
                 and all((self.folder / name).is_file() for name in required_cache))
        if not valid:
            marker.unlink(missing_ok=True)
            count = sum(e['frames'] for e in entries)
            original = None
            if derivation:
                assert derivation['offset_per_finger_m'] == .003
                assert derivation['state_unchanged'] and derivation['image_arrays_unchanged']
                assert all(r['state_and_non_action_members_byte_identical'] for r in audit['rows'])
                # The derived NPZ files contain their own images and measured
                # states. Historical source cleanup must not break this dataset.
                if (Path(derivation['source']) / 'manifest.json').is_file():
                    original = RobotwinDataset(derivation['source'])
                    assert original.source != self.source
                    source_audit = json.loads((original.source / 'correctness_audit.json').read_text())
                    source_hashes = {r['id']: r['npz_sha256'] for r in source_audit['rows']}
                    assert all(r['source_npz_sha256'] == source_hashes[r['id']] for r in audit['rows'])
                    assert [e['id'] for e in entries] == [e['id'] for e in original.entries]
            required = 0 if original else count * 256 * 256 * 3 * 3
            assert shutil.disk_usage(self.folder).free > required + 5 * 1024**3
            dest = {}
            for key in CAMERAS:
                path = self.folder / f'{key}.npy'
                if original:
                    assert not path.exists() or path.is_symlink(), 'refuse to replace an existing image cache'
                    path.unlink(missing_ok=True)
                    path.symlink_to(original.folder / f'{key}.npy')
                else:
                    if path.is_symlink():
                        # Remove the link itself; never write into its target.
                        path.unlink()
                    dest[key] = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8,
                        shape=(count, 256, 256, 3))
            states, actions, cached, cursor = [], [], [], 0
            language, names, units = None, None, None
            for i, e in enumerate(entries):
                with np.load(self.source / e['npz'], allow_pickle=False) as d:
                    s = d['state']
                    assert s.dtype == np.float32 and s.shape == (e['frames'], 16)
                    action = d['action']
                    assert np.isfinite(s).all() and action.dtype == np.float32 and action.shape == s.shape
                    assert np.isfinite(action).all()
                    if derivation:
                        if self.contract == CONTRACT:
                            expected, _, _ = inward_targets(s, d['diagnostic_command14'])
                        else:
                            expected = s.copy()
                            for col, dims in ((6, [6, 7]), (13, [14, 15])):
                                closed = np.isclose(d['diagnostic_command14'][:, col], 0, atol=1e-7, rtol=0)
                                expected[np.ix_(closed, dims)] -= np.float32(.003)
                        np.testing.assert_array_equal(action, expected)
                    assert bool(d['success']) and int(d['sample_hz']) == 25
                    np.testing.assert_allclose(d['timestamp'], np.arange(len(s)) / 25, atol=1e-9)
                    np.testing.assert_array_equal(d['tick'], np.arange(len(s)) * 10)
                    lang, ns, us = str(d['task']), d['joint_names'].tolist(), d['joint_units'].tolist()
                    if language is None:
                        language, names, units = lang, ns, us
                    assert (lang, ns, us) == (language, names, units)
                    assert us == (['rad'] * 6 + ['m'] * 2) * 2
                    for key in dest:
                        x = d[key]
                        assert x.shape == (len(s), 256, 256, 3) and x.dtype == np.uint8
                        dest[key][cursor:cursor + len(s)] = x
                    states.append(s.copy())
                    actions.append(action.copy())
                    cached.append(dict(e, start=cursor, stop=cursor + len(s), episode=i))
                    cursor += len(s)
                if i % 20 == 0 or i == 99:
                    print(f'RoboTwin cache {i + 1}/100', flush=True)
            for x in dest.values():
                x.flush()
            np.save(self.folder / 'state.npy', np.concatenate(states))
            np.save(self.folder / 'action.npy', np.concatenate(actions))
            marker.write_text(json.dumps(dict(fingerprint=self.fingerprint, entries=cached,
                language=language, joint_names=names, joint_units=units), ensure_ascii=False))
        meta = json.loads(marker.read_text())
        self.entries, self.language = meta['entries'], meta['language']
        self.joint_names, self.joint_units = meta['joint_names'], meta['joint_units']
        self.state = np.load(self.folder / 'state.npy', mmap_mode='r')
        self.action = np.load(self.folder / 'action.npy', mmap_mode='r')
        self.images = {key: np.load(self.folder / f'{key}.npy', mmap_mode='r') for key in CAMERAS}
        self.ends = np.empty(len(self.state), dtype=np.int64)
        self.split = np.empty(len(self.state), dtype=np.int8)
        for e in self.entries:
            self.ends[e['start']:e['stop']] = e['stop']
            self.split[e['start']:e['stop']] = 0 if e['split'] == 'train' else 1

    def indices(self, split):
        assert split in ('train', 'val', 'validation')
        return np.flatnonzero(self.split == (0 if split == 'train' else 1))

    def target_indices(self, starts):
        starts = np.asarray(starts, dtype=np.int64)
        assert starts.ndim == 1 and np.all((starts >= 0) & (starts < len(self.state)))
        return np.minimum(starts[:, None] + np.arange(1, 17), self.ends[starts, None] - 1)

    def statistics(self, torch):
        stats = {}
        for key, array in (('observation.state', self.state), ('action', self.action)):
            values = np.asarray(array[self.indices('train')])
            mean, raw_std = values.mean(0), values.std(0)
            stats[key] = dict(mean=torch.tensor(mean.copy()),
                std=torch.tensor(np.maximum(raw_std, 1e-3)), raw_std=torch.tensor(raw_std.copy()))
        return stats


def observation_batch(images, left, right, state, language, torch, device):
    state = np.asarray(state)
    pictures = [np.asarray(x) for x in (images, left, right)]
    if state.ndim == 1:
        state, pictures = state[None], [x[None] for x in pictures]
    assert state.shape == (len(state), 16) and np.isfinite(state).all()
    batch = {'observation.state': torch.tensor(state.copy(), device=device, dtype=torch.float32),
             'task': [language] * len(state)}
    for key, x in zip(FEATURES, pictures):
        assert x.shape == (len(state), 256, 256, 3) and x.dtype == np.uint8
        batch[key] = torch.tensor(x.transpose(0, 3, 1, 2).copy(), device=device, dtype=torch.float32) / 255
    return batch


def make_batch(data, starts, torch, device):
    starts = np.asarray(starts, dtype=np.int64)
    batch = observation_batch(*(data.images[k][starts] for k in CAMERAS),
                              data.state[starts], data.language, torch, device)
    batch['action'] = torch.tensor(np.array(data.action[data.target_indices(starts)]), device=device)
    return batch
