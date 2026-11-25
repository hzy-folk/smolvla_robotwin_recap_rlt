"""File-based cooperative pause requests, safe at completed optimizer steps."""
import json
import os
from pathlib import Path


class PauseTraining(Exception):
    """A requested pause completed successfully (not a training error)."""


def request_pause(checkpoint_dir: Path) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    request = checkpoint_dir / "pause_requested"
    request.touch()
    return request


def pause_if_requested(checkpoint_dir: Path, step: int, save, emit) -> None:
    request = checkpoint_dir / "pause_requested"
    if not request.exists():
        return
    # If saving fails, retain the request and never claim the pause succeeded.
    checkpoint = save(step)
    result = {"phase": "paused", "step": step, "checkpoint": checkpoint.name,
              "resume_checkpoint": str(checkpoint.resolve())}
    temporary = checkpoint_dir / "pause_state.json.tmp"
    with temporary.open("w") as stream:
        json.dump(result, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(checkpoint_dir / "pause_state.json")
    request.unlink(missing_ok=True)
    emit(result)
    raise PauseTraining()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ask a running value trainer to save and pause")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    print(request_pause(args.checkpoint_dir.resolve()))
