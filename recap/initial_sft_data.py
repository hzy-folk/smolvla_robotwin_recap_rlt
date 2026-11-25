"""Success-only RECAP dataset; preserve recorded episode splits and actions."""
import hashlib
import json
import shutil
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "collection"))
import numpy as np
from robotwin_gripper_action import inward_targets, TIMING_RULE
from robotwin_smol_data import RobotwinDataset, LEGACY_CONTRACT, CONTRACT, RECORDED_ACTION_CONTRACT, CAMERAS, sha


class InitialSFTDataset(RobotwinDataset):
    def __init__(self, source, cache=None):
        self.source = Path(source).resolve()
        manifest = self.source / 'manifest.json'
        manifest_data = json.loads(manifest.read_text())
        entries = manifest_data['episodes']
        assert entries and len({e['id'] for e in entries}) == len(entries)
        assert {e['split'] for e in entries} == {'train', 'validation'}
        assert all(e.get('task_success', True) and not e.get('intentional_failure', False) for e in entries)
        derivation = manifest_data.get('derivation') or {}
        corrected_actions = derivation.get('timing_rule') == TIMING_RULE
        self.contract = CONTRACT if corrected_actions else RECORDED_ACTION_CONTRACT
        hashes = {e['id']: sha(self.source / e['npz']) for e in entries}
        self.fingerprint = hashlib.sha256(json.dumps(dict(manifest=sha(manifest), raw=hashes,
            adapter=sha(Path(__file__)), conditioning='Advantage: positive'), sort_keys=True).encode()).hexdigest()
        self.folder = Path(cache or self.source / '.recap_sft_cache').resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
        marker = self.folder / 'complete.json'
        valid = marker.exists() and json.loads(marker.read_text())['fingerprint'] == self.fingerprint
        if not valid:
            marker.unlink(missing_ok=True)
            count = sum(e['frames'] for e in entries)
            reuse_images = all((self.folder / f'{k}.npy').is_symlink() for k in CAMERAS)
            required_image_bytes = 0 if reuse_images else count * 256*256*3*3
            assert shutil.disk_usage(self.folder).free > required_image_bytes + 3*1024**3
            # The corrected-action view has the same images as the raw success
            # cache. Pre-linked image arrays are reused to avoid another ~16 GB
            # uncompressed copy; state/action are still rebuilt from this source.
            dest = {}
            for k in CAMERAS:
                path = self.folder / f'{k}.npy'
                if path.is_symlink():
                    probe = np.load(path, mmap_mode='r')
                    assert probe.shape == (count, 256, 256, 3) and probe.dtype == np.uint8
                else:
                    dest[k] = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8,
                        shape=(count,256,256,3))
            states, actions, cached, cursor = [], [], [], 0
            language = names = units = None
            for i, e in enumerate(entries):
                with np.load(self.source / e['npz'], allow_pickle=False) as d:
                    s, a = d['state'], d['action']
                    assert s.shape == a.shape == (e['frames'],16)
                    assert s.dtype == a.dtype == np.float32 and np.isfinite(s).all() and np.isfinite(a).all()
                    if corrected_actions:
                        expected, _, _ = inward_targets(s, d['diagnostic_command14'])
                        mask = expected != s
                        np.testing.assert_array_equal(a, expected)
                        assert np.array_equal(a[~mask], s[~mask])
                    assert bool(d['success']) and int(d['sample_hz']) == 25
                    if 'official_success' in d:
                        assert bool(d['official_success'][-1])
                    np.testing.assert_allclose(d['timestamp'], np.arange(len(s))/25, atol=1e-9)
                    current = (str(d['task'].item()), d['joint_names'].tolist(), d['joint_units'].tolist())
                    if language is None: language, names, units = current
                    assert current == (language, names, units)
                    for k in CAMERAS:
                        x = d[k]
                        assert x.shape == (len(s),256,256,3) and x.dtype == np.uint8
                        if k in dest:
                            dest[k][cursor:cursor+len(s)] = x
                    states.append(s); actions.append(a)
                    cached.append(dict(e, start=cursor, stop=cursor+len(s), episode=i))
                    cursor += len(s)
                if (i+1)%10 == 0: print(f'RECAP success cache {i+1}/{len(entries)}', flush=True)
            for x in dest.values(): x.flush()
            np.save(self.folder/'state.npy', np.concatenate(states))
            np.save(self.folder/'action.npy', np.concatenate(actions))
            marker.write_text(json.dumps(dict(fingerprint=self.fingerprint, entries=cached,
                language=language, joint_names=names, joint_units=units)))
        meta = json.loads(marker.read_text())
        self.entries = meta['entries']
        self.language = meta['language'] + '. Advantage: positive'
        self.joint_names, self.joint_units = meta['joint_names'], meta['joint_units']
        self.state = np.load(self.folder/'state.npy', mmap_mode='r')
        self.action = np.load(self.folder/'action.npy', mmap_mode='r')
        self.images = {k: np.load(self.folder/f'{k}.npy', mmap_mode='r') for k in CAMERAS}
        self.ends = np.empty(len(self.state), dtype=np.int64)
        self.split = np.empty(len(self.state), dtype=np.int8)
        for e in self.entries:
            self.ends[e['start']:e['stop']] = e['stop']
            self.split[e['start']:e['stop']] = 0 if e['split']=='train' else 1
