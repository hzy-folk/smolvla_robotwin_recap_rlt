"""官方 SmolVLA；共享预训练主干，平台各自训练动作专家并保存增量权重。"""
import os
import sys
import json
from dataset import AUDIT
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.extend([str(AUDIT/'venv/lib/python3.10/site-packages'),
                 '/home/hzy/venvs/legged-model/lib/python3.10/site-packages'])
sys.path.insert(0,str(AUDIT/'lerobot_stable_src'))
import torch
from safetensors.torch import load_file
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.types import PolicyFeature,FeatureType,NormalizationMode


def make(state_dim,action_dim,stats,cameras=1):
    assert cameras in (1,2)
    raw=json.loads((AUDIT/'followup_runs/models/smolvla/config.json').read_text());raw.pop('type')
    raw['input_features']={'observation.state':PolicyFeature(FeatureType.STATE,(state_dim,)),
                           'observation.images.camera1':PolicyFeature(FeatureType.VISUAL,(3,256,256))}
    if cameras == 2:
        raw['input_features']['observation.images.camera2']=PolicyFeature(FeatureType.VISUAL,(3,256,256))
    raw['output_features']={'action':PolicyFeature(FeatureType.ACTION,(action_dim,))}
    raw['normalization_mapping']={k:NormalizationMode(v) for k,v in raw['normalization_mapping'].items()}
    raw.update(vlm_model_name=str(AUDIT/'followup_runs/models/smolvlm'),load_vlm_weights=False,
               resize_imgs_with_padding=(256,256),chunk_size=16,n_action_steps=5,tokenizer_max_length=64,num_steps=10)
    model=SmolVLAPolicy(SmolVLAConfig(**raw),dataset_stats=stats)
    weights=load_file(str(AUDIT/'followup_runs/models/smolvla/model.safetensors'))
    own=model.state_dict()
    weights={k.replace('model._orig_mod.','model.'):v for k,v in weights.items()}
    weights={k:v for k,v in weights.items() if k.startswith('model.') and k in own}
    missing,unexpected=model.load_state_dict(weights,strict=False)
    pointers={own[k].data_ptr() for k in weights}
    assert not unexpected
    assert not [k for k in missing if k.startswith('model.') and own[k].data_ptr() not in pointers]
    return model
