"""Project-local paths and portable runtime configuration."""
from pathlib import Path
import sys
SMOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SMOL_ROOT))
from runtime_config import *  # noqa: F403
PROJECT_ROOT = SMOL_ROOT.parent
DATA_ROOT = SMOL_ROOT / "data"
RUNS_ROOT = SMOL_ROOT / "runs"
