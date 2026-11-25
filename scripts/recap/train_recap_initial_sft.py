#!/usr/bin/env python3
"""Real positive-conditioned SFT using the existing trainer and isolated dataset adapter.

Initializes from original pretrained SmolVLA, not an old task checkpoint.
All observations, including validation, receive Advantage: positive.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from initial_sft_data import InitialSFTDataset
import initial_sft_trainer as trainer

if __name__ == '__main__':
    trainer.RobotwinDataset = InitialSFTDataset
    trainer.main()
