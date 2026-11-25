"""Action contract helper retained for SmolVLA inference imports."""
import numpy as np

TIMING_RULE = 'measured_mean_le_28mm_connected_to_command_zero_v3'


def inward_targets(state, diagnostic_command14):
    state = np.asarray(state, dtype=np.float32)
    command = np.asarray(diagnostic_command14, dtype=np.float32)
    out = state.copy()
    # The paired finger coordinates move inward by 3 mm only during the
    # measured closed-gripper interval; this is the dataset action contract.
    for command_col, finger_cols in ((6, (6, 7)), (13, (14, 15))):
        closed = np.isclose(command[:, command_col], 0, atol=1e-7, rtol=0)
        closed &= np.mean(state[:, finger_cols], axis=1) <= 0.028
        out[np.ix_(closed, finger_cols)] -= np.float32(0.003)
    return out, np.zeros_like(out, dtype=bool), np.zeros(len(out), dtype=np.int64)
