"""Stable paths shared by the task-specific SmolVLA scripts."""
from pathlib import Path

SMOL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SMOL_ROOT.parent
AUDIT = PROJECT_ROOT.parent / 'feasibility_audit'

# Keep ROOT as the local SmolVLA root for existing task-specific imports.
ROOT = SMOL_ROOT
DATA_ROOT = SMOL_ROOT / 'data'
RUNS_ROOT = SMOL_ROOT / 'runs'
