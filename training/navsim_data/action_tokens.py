from __future__ import annotations

"""Helpers to convert NavSim continuous values into <bev_*> tokens."""

from typing import List, Sequence, Tuple, Union

import numpy as np
import torch

_BEV_MIN, _BEV_MAX = -100.0, 100.0
_BEV_RES = 0.3
_BEV_BINS = np.arange(_BEV_MIN, _BEV_MAX + 1e-6, _BEV_RES, dtype=np.float32)

_DIRECTIONS = ("go left", "go straight", "go right", "unknown")


def bev_tokens() -> List[str]:
    return [f"<bev_{i}>" for i in range(len(_BEV_BINS))]


def _value_to_bev_token(value: float) -> str:
    clipped = float(np.clip(value, _BEV_BINS[0], _BEV_BINS[-1]))
    idx = int(np.argmin(np.abs(_BEV_BINS - clipped)))
    return f"<bev_{idx}>"


def split_status_tensor(
    status: Union[torch.Tensor, Sequence[float], np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(status, torch.Tensor):
        status_np = status.detach().cpu().numpy()
    else:
        status_np = np.asarray(status)
    if status_np.shape[-1] < 8:
        raise ValueError(f"Expected status tensor of length >= 8, got {status_np.shape}")
    command = status_np[..., :4]
    velocity = status_np[..., 4:6]
    acceleration = status_np[..., 6:8]
    return command, velocity, acceleration


def driving_command_to_text(command: Union[np.ndarray, Sequence[float]]) -> str:
    idx = int(np.argmax(command))
    idx = max(0, min(len(_DIRECTIONS) - 1, idx))
    return _DIRECTIONS[idx]


def velocity_to_bev_tokens(velocity: Union[np.ndarray, Sequence[float]]) -> List[str]:
    vx, vy = map(float, velocity[:2])
    return [_value_to_bev_token(vx), _value_to_bev_token(vy)]


def acceleration_to_bev_tokens(acceleration: Union[np.ndarray, Sequence[float]]) -> List[str]:
    ax, ay = map(float, acceleration[:2])
    return [_value_to_bev_token(ax), _value_to_bev_token(ay)]


def status_to_bev_tokens(status: Union[torch.Tensor, Sequence[float], np.ndarray]) -> List[str]:
    _, velocity, acceleration = split_status_tensor(status)
    return velocity_to_bev_tokens(velocity) + acceleration_to_bev_tokens(acceleration)


def trajectory_to_bev_tokens(trajectory: Union[torch.Tensor, np.ndarray]) -> List[str]:
    """
    Convert trajectory waypoints to BEV tokens.
    :param trajectory: trajectory tensor of shape [T, 3] or [T, 2+] where first 2 dims are (x, y)
    :return: list of BEV tokens for each waypoint [x0, y0, x1, y1, ...]
    """
    if isinstance(trajectory, torch.Tensor):
        traj_np = trajectory.detach().cpu().numpy()
    else:
        traj_np = np.asarray(trajectory)
    
    # Extract x, y coordinates from trajectory (shape: [T, 3] or [T, 2+])
    tokens = []
    for waypoint in traj_np:
        x, y = float(waypoint[0]), float(waypoint[1])
        tokens.append(_value_to_bev_token(x))
        tokens.append(_value_to_bev_token(y))
    
    return tokens


def action_token_vocab() -> List[str]:
    """Return BEV token vocabulary used for action regression."""
    return bev_tokens()
