"""Final RECAP dataset: lossless image cache and per-observation conditions."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from robotwin_smol_data import RobotwinDataset, CONTRACT, RECORDED_ACTION_CONTRACT, CAMERAS, sha, observation_batch

PROJECT = Path(__file__).resolve().parents[2]
VALUE = PROJECT / 'recap_runs/value_round2_pistar_batch8_40000'
ROLLOUT = PROJECT / 'recap_runs/rollout50_checkpoint020000_adv_positive'


class ImageIndex:
    def __init__(self, dataset, camera):
        self.dataset, self.camera = dataset, camera

    def __getitem__(self, ids):
        frames = []
        for i in np.asarray(ids):
            offset, size = self.dataset.offsets[i, self.camera]
            self.dataset.stream.seek(int(offset))
            image = cv2.imdecode(np.frombuffer(self.dataset.stream.read(int(size)), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f'Corrupt cache image {i}/{self.camera}')
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return np.stack(frames)


class FinalPolicyDataset(RobotwinDataset):
    def __init__(self, source=VALUE, cache=None, value_data=None):
        self.source = Path(source).resolve()
        label_path = self.source / 'advantages.json'
        meta_path = (Path(value_data).resolve() / 'manifest.json' if value_data else
                     PROJECT / 'recap_runs/value_round1_pistar/data/manifest.json')
        meta = json.loads(meta_path.read_text())
        labels = json.loads(label_path.read_text())
        assert labels['threshold_percentile'] == 50
        label_map = {r['key']: r for r in labels['rows']}
        sources = {}
        corrected_sources = set()
        source_manifests = {}
        for kind, name in [('original_success','robotwin_success_80_inward3mm_v3'),
                           ('original_failure','robotwin_failure_30_gpu_final_inward3mm')]:
            root = Path(meta.get('source_roots', {}).get(kind, PROJECT / 'data' / name))
            source_manifests[kind] = sha(root / 'manifest.json')
            manifest = json.loads((root / 'manifest.json').read_text())
            from robotwin_gripper_action import TIMING_RULE
            if (manifest.get('derivation') or {}).get('timing_rule') == TIMING_RULE:
                corrected_sources.add(kind)
            for e in manifest['episodes']:
                sources[f"{kind}__{e['id']}"] = root / e['npz']
        for e in meta['episodes']:
            key = f"{e['source']}__{e['id']}"
            if key not in sources:
                sources[key] = Path(meta.get('source_roots', {}).get('rollout', ROLLOUT)) / 'rollouts' / e['id'] / 'rollout.npz'
        self.fingerprint = hashlib.sha256(json.dumps({'labels':sha(label_path),'manifest':sha(meta_path),
            'adapter':sha(Path(__file__)), 'source_manifests':source_manifests, 'sources':{k:sha(v) for k,v in sources.items()}},sort_keys=True).encode()).hexdigest()
        self.folder = Path(cache or self.source / '.final_cache').resolve()
        self.folder.mkdir(parents=True,exist_ok=True)
        marker = self.folder / 'complete.json'
        valid = marker.exists() and json.loads(marker.read_text())['fingerprint']==self.fingerprint
        if not valid:
            marker.unlink(missing_ok=True)
            offsets = np.zeros((meta['samples'],3,2),dtype=np.int64)
            states, actions, conditions, entries = [], [], [], []
            cursor = 0
            with (self.folder/'images.pngbin').open('wb') as stream:
                for e in meta['episodes']:
                    key=f"{e['source']}__{e['id']}"; path=sources[key]
                    with np.load(path,allow_pickle=False) as data:
                        s=data['state']; n=len(s)
                        assert n==e['stop']-e['start']
                        if e['source'].startswith('original'):
                            a=data['action']
                            from robotwin_gripper_action import inward_targets
                            if e['source'] in corrected_sources:
                                expected,_,_=inward_targets(s,data['diagnostic_command14'])
                                np.testing.assert_array_equal(a,expected)
                            self.joint_names=data['joint_names'].tolist();self.joint_units=data['joint_units'].tolist()
                        else:
                            applied=data['applied_targets']
                            assert len(applied)==n-1
                            # applied[t] drives state[t] -> state[t+1]. The shared
                            # future+1 contract must retrieve applied[t], not t+1.
                            a=np.concatenate([applied[:1],applied],axis=0)
                            np.testing.assert_array_equal(a[1:],applied)
                        assert s.shape==a.shape==(n,16) and np.isfinite(s).all() and np.isfinite(a).all()
                        states.append(s.copy());actions.append(a.copy())
                        row=label_map[key]
                        assert len(row['adv_ind'])==n
                        np.testing.assert_array_equal(np.asarray(row['advantage'])>=labels['threshold'],np.array(row['adv_ind'])=='positive')
                        conditions.extend(row['adv_ind'])
                        for c,k in enumerate(CAMERAS):
                            if e['source'].startswith('original'):
                                frames=data[k]; assert frames.shape==(n,256,256,3)
                                iterator=(cv2.cvtColor(x,cv2.COLOR_RGB2BGR) for x in frames)
                                cap=None
                            else:
                                name=('head','left_wrist','right_wrist')[c]
                                cap=cv2.VideoCapture(str(path.parent/f'{name}.mp4'))
                                def video_frames():
                                    while True:
                                        ok,x=cap.read()
                                        if not ok:break
                                        yield x
                                iterator=video_frames()
                            count=0
                            for t,x in enumerate(iterator):
                                if t>=n: raise ValueError('Extra video frames')
                                assert x.shape==(256,256,3)
                                ok,encoded=cv2.imencode('.png',x,[cv2.IMWRITE_PNG_COMPRESSION,1]);assert ok
                                offsets[cursor+t,c]=[stream.tell(),len(encoded)]
                                stream.write(encoded.tobytes());count+=1
                            if cap:cap.release()
                            assert count==n,(key,k,count,n)
                    entries.append(dict(e,start=cursor,stop=cursor+n,episode=len(entries),npz=str(path)))
                    cursor+=n
                    print(f'final cache {len(entries)}/{len(meta["episodes"])} {key}',flush=True)
            assert cursor==meta['samples']
            np.save(self.folder/'state.npy',np.concatenate(states));np.save(self.folder/'action.npy',np.concatenate(actions))
            np.save(self.folder/'offsets.npy',offsets)
            marker.write_text(json.dumps(dict(fingerprint=self.fingerprint,entries=entries,conditions=conditions,
                joint_names=self.joint_names,joint_units=self.joint_units)))
        saved=json.loads(marker.read_text());self.entries=saved['entries'];self.conditions=np.array(saved['conditions'])
        self.joint_names=saved['joint_names'];self.joint_units=saved['joint_units']
        self.state=np.load(self.folder/'state.npy',mmap_mode='r');self.action=np.load(self.folder/'action.npy',mmap_mode='r')
        self.offsets=np.load(self.folder/'offsets.npy',mmap_mode='r');self.stream=(self.folder/'images.pngbin').open('rb')
        self.images={k:ImageIndex(self,i) for i,k in enumerate(CAMERAS)}
        self.ends=np.empty(len(self.state),dtype=np.int64);self.split=np.empty(len(self.state),dtype=np.int8)
        self.tasks=np.empty(len(self.state),dtype=object)
        for e in self.entries:
            sl=slice(e['start'],e['stop']);self.ends[sl]=e['stop'];self.split[sl]=0 if e['split']=='train' else 1
            self.tasks[sl]=e['task']
        self.language='stack the red block and then the green block'
        self.contract=CONTRACT if len(corrected_sources)==2 else RECORDED_ACTION_CONTRACT
        assert {e['split'] for e in self.entries} == {'train', 'validation'}


def make_final_batch(data,starts,torch,device):
    starts=np.asarray(starts,dtype=np.int64)
    batch=observation_batch(*(data.images[k][starts] for k in CAMERAS),data.state[starts],data.language,torch,device)
    batch['task']=[f'{data.tasks[t]}. Advantage: {data.conditions[t]}' for t in starts]
    batch['action']=torch.tensor(np.array(data.action[data.target_indices(starts)]),device=device)
    return batch
