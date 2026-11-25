"""Model-only subprocess: JSON protocol, current RGB/qpos/language inputs only."""
import argparse
import contextlib
import json
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--flow-steps', type=int, default=4)
    a = p.parse_args()
    protocol = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        from robotwin_smol_model import Policy, torch
        torch.set_num_threads(4); torch.manual_seed(300)
        policy = Policy(a.checkpoint, a.device, a.flow_steps)
    print(json.dumps(dict(ready=True, checkpoint_step=policy.step, contract=policy.contract)), file=protocol, flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get('stop'):
            break
        try:
            with contextlib.redirect_stdout(sys.stderr), np.load(request['input'], allow_pickle=False) as d:
                prediction = policy.predict(d['images'], d['left_wrist'], d['right_wrist'], d['state'], request['language'])
                np.savez(request['output'], predicted_chunk=prediction)
            print(json.dumps(dict(ok=True, shape=list(prediction.shape))), file=protocol, flush=True)
        except Exception as exc:
            print(json.dumps(dict(ok=False, error=repr(exc))), file=protocol, flush=True)


if __name__ == '__main__':
    main()
