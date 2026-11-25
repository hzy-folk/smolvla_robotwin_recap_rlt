"""RoboTwin SmolVLA: raw inference, native qpos replay, or actual closed-loop control."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from dataset import AUDIT
from robotwin_smol_data import CONTRACT, CAMERAS, future_action_chunk


class Worker:
    def __init__(self, checkpoint, device, flow_steps, output):
        self.output = output
        self.log = (output / 'model_worker.log').open('w')
        command = [str(AUDIT / 'followup_vlm_venv/bin/python'), str(Path(__file__).with_name('robotwin_smol_worker.py')),
                   '--checkpoint', str(checkpoint.resolve()), '--device', device, '--flow-steps', str(flow_steps)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, text=True, bufsize=1)
        try:
            self.ready = self.read()
            assert self.ready.get('ready'), self.ready
        except BaseException:
            self.close()
            raise

    def read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Model worker exited; see model_worker.log')
        return json.loads(line)

    def predict(self, observation, language):
        request, result = self.output / '_current_observation.npz', self.output / '_current_prediction.npz'
        np.savez(request, **dict(zip((*CAMERAS, 'state'), observation)))
        self.process.stdin.write(json.dumps(dict(input=str(request), output=str(result), language=language)) + '\n')
        self.process.stdin.flush()
        response = self.read()
        if not response.get('ok'):
            raise RuntimeError(response)
        with np.load(result, allow_pickle=False) as d:
            chunk = d['predicted_chunk'].copy()
        assert chunk.shape == (16, 16) and np.isfinite(chunk).all()
        return chunk

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"stop":true}\n'); self.process.stdin.flush()
                self.process.wait(timeout=10)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.terminate()
                self.process.wait(timeout=10)
        self.log.close()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--mode', choices=('raw', 'reference', 'rollout'), default='raw')
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frames', type=int, nargs='+', default=[0])
    p.add_argument('--rollout-steps', type=int, default=376, help='25 Hz steps; default 15.04 simulated seconds')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--flow-steps', type=int, default=4)
    p.add_argument('--task-prompt')
    p.add_argument('--advantage', choices=('positive', 'negative'))
    p.add_argument('--no-video', action='store_true', help='skip camera video encoding during rollout')
    p.add_argument('--check-success-every-step', action='store_true',
                   help='check RoboTwin success after each action and stop on first success; skip settling tail')
    a = p.parse_args()
    assert a.flow_steps > 0 and a.rollout_steps > 0
    assert a.mode == 'reference' or a.checkpoint is not None
    raw, output = a.raw.resolve(), a.output.resolve()
    assert not output.exists(), 'use a new output directory'
    output.mkdir(parents=True)
    report = dict(mode=a.mode, contract=CONTRACT, raw=str(raw), flow_steps=a.flow_steps,
        control_interval_seconds=.04, simulation_time_pauses_during_model_inference=True,
        uses_future_truth_for_control=a.mode == 'reference')
    worker, env = None, None
    try:
        with np.load(raw, allow_pickle=False) as d:
            language = a.task_prompt or str(d['task'])
        if a.advantage:
            language += f'. Advantage: {a.advantage}'
        report['policy_language'] = language
        report['advantage_condition'] = a.advantage
        if a.mode == 'raw':
            worker = Worker(a.checkpoint, a.device, a.flow_steps, output)
            report.update(checkpoint_step=worker.ready['checkpoint_step'], checkpoint=str(a.checkpoint.resolve()),
                          contract=worker.ready['contract'])
            rows = []
            with np.load(raw, allow_pickle=False) as d:
                for t in a.frames:
                    assert 0 <= t < len(d['state'])
                    observation = tuple(d[key][t] for key in (*CAMERAS, 'state'))
                    started = time.monotonic()
                    pred = worker.predict(observation, language)
                    np.savez(output / f'frame_{t:04d}.npz', predicted_chunk=pred, executed_targets=pred[:8], state=observation[-1])
                    rows.append(dict(frame=t, shape=list(pred.shape), finite=True, seconds=time.monotonic() - started))
            report.update(frames=rows, limitation='Interface test only; not closed-loop task success.')
        else:
            import cv2
            from robotwin_smol_control import restore
            env, control, alignment = restore(raw)
            report.update(alignment)
            reference = None
            reference_state = None
            if a.mode == 'reference':
                with np.load(raw, allow_pickle=False) as d:
                    reference = d['action'].copy()
                    reference_state = d['state'].copy()
                report['reference_source'] = 'action[t+1:t+17]; final action repeated, no extra inward correction'
            else:
                # Initialize SAPIEN before the GPU inference worker. On this
                # machine, creating the Vulkan renderer after a separate CUDA
                # model process starts can crash the simulator during startup.
                worker = Worker(a.checkpoint, a.device, a.flow_steps, output)
                report.update(checkpoint_step=worker.ready['checkpoint_step'],
                              checkpoint=str(a.checkpoint.resolve()),
                              contract=worker.ready['contract'])
            observation = control.observe()
            states, targets, applied, replans, chunks = [observation[-1]], [], [], [], []
            writers = None if a.no_video else [
                cv2.VideoWriter(str(output / f'{name}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                25, (256, 256)) for name in ('head', 'left_wrist', 'right_wrist')]
            assert writers is None or all(w.isOpened() for w in writers)
            def write_frame(obs):
                if writers is not None:
                    for w, rgb in zip(writers, obs[:3]):
                        w.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            write_frame(observation)
            limit = len(reference) - 1 if reference is not None else a.rollout_steps
            started = time.monotonic()
            success_step = None
            try:
                while len(targets) < limit and success_step is None:
                    t = len(targets)
                    replans.append(t)
                    if reference is None:
                        # Model receives only the current three views + actual state.
                        chunk = worker.predict(observation, language)
                    else:
                        chunk = future_action_chunk(reference, t)
                    # Identical tracking for model and reference modes: start
                    # each newly planned eight-step segment at measured qpos.
                    control.reanchor()
                    chunks.append(chunk)
                    for target in chunk[:min(8, limit - t)]:
                        targets.append(target)
                        applied.append(control.execute(target))
                        observation = control.observe()
                        states.append(observation[-1]); write_frame(observation)
                        if a.check_success_every_step and env.check_success():
                            success_step = len(targets)
                            break
            finally:
                if writers is not None:
                    for w in writers:
                        w.release()
            if a.check_success_every_step:
                report['official_success'] = success_step is not None
                report['success_step'] = success_step
                report['success_check_policy'] = 'RoboTwin check_success after every low-level action; stop at first success; no settling tail'
            else:
                report.update(control.finish())
            actual = np.asarray(states)
            report.update(steps=len(targets), simulated_seconds=len(targets) / 25,
                wall_seconds=time.monotonic() - started, replan_steps=replans,
                tracker='ten-step linear interpolation plus drive velocity feedforward and gravity compensation')
            if reference is not None:
                report.update(reference_max_error=float(np.abs(actual - reference_state).max()),
                    tracking_error_reference='original measured state; intentionally different from inward action target',
                    arm_mae_rad=float(np.abs(actual - reference_state)[:, [0,1,2,3,4,5,8,9,10,11,12,13]].mean()),
                    gripper_mae_m=float(np.abs(actual - reference_state)[:, [6,7,14,15]].mean()))
            np.savez_compressed(output / 'rollout.npz', state=actual, predicted_targets=np.asarray(targets),
                                applied_targets=np.asarray(applied), predicted_chunks=np.asarray(chunks))
        report['completed'] = True
    except BaseException as exc:
        report.update(completed=False, error=repr(exc))
        raise
    finally:
        if env is not None:
            env.close_env()
        if worker is not None:
            worker.close()
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
