"""Native LIBERO OSC7 inference, with the same checkpoint and wire contract."""
import numpy as np
from smol_model import make, torch


class LiberoPolicy:
    def __init__(self, checkpoint, device):
        self.device = device
        self.model = make(8, 7, checkpoint['stats'], cameras=2)
        expected = {name for name, p in self.model.named_parameters() if p.requires_grad}
        if set(checkpoint['trainable']) != expected:
            raise ValueError('Incomplete or incompatible LIBERO task checkpoint')
        self.model.load_state_dict(checkpoint['trainable'], strict=False)
        self.model.to(device).eval()

    def predict(self, obs):
        batch = {'task': ['pick up the salad dressing'],
                 'observation.state': torch.as_tensor(obs['states'], device=self.device, dtype=torch.float32)}
        for i, name in enumerate(('main_images', 'wrist_images'), start=1):
            image = np.asarray(obs[name])
            if image.shape != (1, 256, 256, 3) or image.dtype != np.uint8:
                raise ValueError(f'Expected uint8 single-environment RGB, got {image.shape}/{image.dtype}')
            batch[f'observation.images.camera{i}'] = torch.as_tensor(
                np.ascontiguousarray(image.transpose(0, 3, 1, 2)),
                device=self.device, dtype=torch.float32) / 255.0
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16,
                                                   enabled=self.device == 'cuda'):
            chunk = self.model.predict_action_chunk(batch).float().cpu().numpy()
        if chunk.shape != (1, 16, 7) or not np.isfinite(chunk).all():
            raise ValueError('Invalid LIBERO model output')
        return np.ascontiguousarray(np.clip(chunk[:, :5], -1.0, 1.0))

    def close(self):
        del self.model
        if self.device == 'cuda':
            torch.cuda.empty_cache()
