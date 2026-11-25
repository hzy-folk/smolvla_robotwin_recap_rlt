"""PiStar value-model loader used by the SmolVLA RECAP pipeline.

The architecture and implementation are imported from the checked-out PiStar
repository. This module intentionally contains no fallback network: training
must use PiStar's SigLIP + Gemma3 VLM and its 201-bin value head.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path


DEFAULT_PISTAR_ROOT = Path("/home/hzy/git/learn/vla/pi0.6*/pistar")


def load_value_checkpoint(trainer, checkpoint_path, state):
    """Restore weights onto the current device, including GPU-to-CPU checks."""
    import jax
    import orbax.checkpoint as ocp

    checkpoint_path = Path(checkpoint_path).resolve()
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    with ocp.PyTreeCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(str(checkpoint_path))
        restore_args = jax.tree.map(
            lambda leaf: ocp.ArrayRestoreArgs(sharding=sharding)
            if isinstance(leaf, ocp.metadata.ArrayMetadata) else ocp.RestoreArgs(),
            metadata,
        )
        restored = checkpointer.restore(str(checkpoint_path), restore_args=restore_args)
    state.params.replace_by_pure_dict(restored['params'])
    if state.ema_params is not None and 'ema_params' in restored:
        state.ema_params.replace_by_pure_dict(restored['ema_params'])
    return trainer.TrainState(int(restored['step']), state.params, state.model_def,
                              state.opt_state, state.ema_params)


def load_pistar_value_components(pistar_root: Path | None = None):
    root = Path(pistar_root or os.environ.get("PISTAR_ROOT", DEFAULT_PISTAR_ROOT)).expanduser().resolve()
    if not (root / "src/openpi/models/value_model.py").is_file():
        raise FileNotFoundError(f"PiStar checkout not found at {root}")
    for path in (root, root / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    # This PiStar checkout's training entrypoint imports two UI-only helpers
    # that are absent from its openpi/shared package. Keep those imports local
    # to the adapter; they do not participate in model or optimizer logic.
    import openpi.shared

    for helper_name in ("console", "progress", "hf_datasets_compat"):
        qualified_name = f"openpi.shared.{helper_name}"
        if qualified_name in sys.modules:
            continue
        try:
            __import__(qualified_name)
        except ModuleNotFoundError as exc:
            if exc.name != qualified_name:
                raise
            helper = types.ModuleType(qualified_name)
            if helper_name == "console":
                for method in ("info", "ok", "warn", "error"):
                    setattr(helper, method, lambda message, *_args, **_kwargs: str(message))
            elif helper_name == "progress":
                helper.sync_pbar_color = lambda _progress_bar: None
            setattr(openpi.shared, helper_name, helper)
            sys.modules[qualified_name] = helper

    script_path = root / "scripts/train_value.py"
    spec = importlib.util.spec_from_file_location("pistar_recap_train_value", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load PiStar value trainer at {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from openpi.models import model as openpi_model
    from openpi.models.value_model_config import ValueModelConfig

    return root, module, openpi_model, ValueModelConfig
