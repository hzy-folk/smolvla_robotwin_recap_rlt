"""Task-local three-camera factory; official SmolVLA and LIBERO remain unchanged."""
import json
import numpy as np

from dataset import AUDIT
from robotwin_smol_data import (CONTRACT, V2_CONTRACT, LEGACY_CONTRACT,
                                RECORDED_ACTION_CONTRACT, FEATURES, observation_batch, sha)
from smol_model import (torch, load_file, SmolVLAConfig, SmolVLAPolicy,
                        PolicyFeature, FeatureType, NormalizationMode)

BASE = AUDIT / 'followup_runs/models/smolvla/model.safetensors'


def make(stats):
    raw = json.loads((BASE.parent / 'config.json').read_text())
    raw.pop('type')
    raw['input_features'] = {'observation.state': PolicyFeature(FeatureType.STATE, (16,))}
    raw['input_features'].update({key: PolicyFeature(FeatureType.VISUAL, (3, 256, 256)) for key in FEATURES})
    raw['output_features'] = {'action': PolicyFeature(FeatureType.ACTION, (16,))}
    raw['normalization_mapping'] = {k: NormalizationMode(v) for k, v in raw['normalization_mapping'].items()}
    raw.update(vlm_model_name=str(AUDIT / 'followup_runs/models/smolvlm'), load_vlm_weights=False,
        resize_imgs_with_padding=(256, 256), chunk_size=16, n_action_steps=8,
        tokenizer_max_length=64, num_steps=4, adapt_to_pi_aloha=False, use_delta_joint_actions_aloha=False)
    model = SmolVLAPolicy(SmolVLAConfig(**raw), dataset_stats=stats)
    own = model.state_dict()
    weights = load_file(str(BASE))
    weights = {k.replace('model._orig_mod.', 'model.'): v for k, v in weights.items()}
    weights = {k: v for k, v in weights.items() if k.startswith('model.') and k in own}
    missing, unexpected = model.load_state_dict(weights, strict=False)
    pointers = {own[k].data_ptr() for k in weights}
    assert not unexpected
    assert not [k for k in missing if k.startswith('model.') and own[k].data_ptr() not in pointers]
    assert model.config.max_action_dim == model.config.max_state_dim == 32
    return model


class Policy:
    def __init__(self, checkpoint, device='cuda', flow_steps=4):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert saved['platform'] == 'robotwin' and saved['state_dim'] == saved['action_dim'] == 16
        self.contract = saved['config']['contract']
        assert self.contract in (CONTRACT, V2_CONTRACT, LEGACY_CONTRACT,
                                 RECORDED_ACTION_CONTRACT), 'incompatible dataset/model contract'
        assert saved['config']['base_model_sha256'] == sha(BASE), 'pretrained base differs from training'
        self.model = make(saved['stats'])
        trainable = {n for n, p in self.model.named_parameters() if p.requires_grad}
        assert set(saved['trainable']) == trainable, 'incomplete or extra checkpoint parameters'
        missing, unexpected = self.model.load_state_dict(saved['trainable'], strict=False)
        assert not unexpected and not trainable.intersection(missing)
        assert flow_steps > 0
        self.model.config.num_steps = flow_steps
        self.model.to(device).eval()
        self.device, self.step = device, saved['step']

    def predict(self, head, left, right, state, language):
        batch = observation_batch(head, left, right, state, language, torch, self.device)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.device == 'cuda'):
            result = self.model.predict_action_chunk(batch)[0].float().cpu().numpy()
        assert result.shape == (16, 16) and np.isfinite(result).all()
        return result  # Native policy already unnormalizes to radians/metres.
