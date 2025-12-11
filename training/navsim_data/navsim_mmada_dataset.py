from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from .action_tokens import status_to_bev_tokens, trajectory_to_bev_tokens
from .navsim_online_dataset import NavsimOnlineDataset


@dataclass
class NavsimSampleTransforms:
    include_stitched: bool = True
    include_front: bool = True


def _default_navsim_image_transform() -> Callable[[Image.Image], torch.Tensor]:
    """Default image transform with normalization to [-1, 1] range."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def _format_velocity(vec: torch.Tensor) -> str:
    vx, vy = float(vec[0].item()), float(vec[1].item())
    speed = math.sqrt(vx ** 2 + vy ** 2)
    heading_deg = math.degrees(math.atan2(vy, vx))
    return f"speed={speed:.2f} m/s, heading={heading_deg:.1f} deg"


def _format_point(x: float, y: float) -> str:
    return f"({x:.2f}, {y:.2f})"


def _format_trajectory(traj: torch.Tensor) -> str:
    # traj shape: [T, 3] (x, y, heading)
    coords = [_format_point(float(p[0].item()), float(p[1].item())) for p in traj]
    return "[" + ", ".join(coords) + "]"


class NavsimMMaDADataset(Dataset):
    """Wrap NavSim stream to expose history context + discrete action tokens for MMaDA training."""

    def __init__(
        self,
        *,
        json_path: str,
        navsim_log_path: str,
        sensor_blobs_path: str,
        transforms: Optional[NavsimSampleTransforms] = None,
        num_history_frames: int = 4,
        num_future_frames: int = 1,
        target_future_seconds: float = 0.5,
        stitched_image_size: Tuple[int, int] = (1024, 256),
        front_image_size: Tuple[int, int] = (256, 128),
        history_image_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        future_image_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        predict_future_image: bool = True,
    ) -> None:
        super().__init__()
        tfm = transforms or NavsimSampleTransforms()

        # Use provided transforms or default normalized transforms
        if history_image_transform is None:
            history_image_transform = _default_navsim_image_transform()
        if future_image_transform is None:
            future_image_transform = _default_navsim_image_transform()

        self.predict_future_image = bool(predict_future_image)
        self.dataset = NavsimOnlineDataset(
            json_path=json_path,
            navsim_log_path=navsim_log_path,
            sensor_blobs_path=sensor_blobs_path,
            include_front=tfm.include_front,
            include_stitched=tfm.include_stitched,
            num_history_frames=num_history_frames,
            num_future_frames=num_future_frames,
            target_future_seconds=target_future_seconds,
            stitched_image_size=stitched_image_size,
            front_image_size=front_image_size,
            history_image_transform=history_image_transform,
            future_image_transform=future_image_transform,
        )
        self.target_future_seconds = target_future_seconds

    def __len__(self) -> int:
        return len(self.dataset)

    def _build_prompt(self, sample: Dict[str, Any]) -> str:
        history_status: torch.Tensor = sample["history_status"]
        history_traj: torch.Tensor = sample["history_trajectory"]
        history_times: torch.Tensor = sample["history_timestamps"]
        num_frames = history_status.shape[0]

        last_status = history_status[-1]
        command_vec = last_status[:4]
        velocity = last_status[4:6]
        acceleration = last_status[6:8]

        dt = float(self.target_future_seconds)
        prompt = [
            f"The ego vehicle history spans {num_frames} frames. Predict the driving action and front camera view after {dt:.2f} seconds.",
            f"Current velocity: {_format_velocity(velocity)}.",
            f"Current acceleration: ax={float(acceleration[0].item()):.2f} m/s^2, ay={float(acceleration[1].item()):.2f} m/s^2.",
        ]

        directions = ["go left", "go straight", "go right", "unknown"]
        dir_idx = int(torch.argmax(command_vec).item())
        dir_text = directions[max(0, min(len(directions) - 1, dir_idx))]
        prompt.append(f"You plan to take the following action: {dir_text}.")

        # Include sampled trajectory as BEV tokens for exact reproducibility
        history_traj_tokens = trajectory_to_bev_tokens(history_traj)
        prompt.append(f"Recent trajectory (ego frame) tokens: {''.join(history_traj_tokens)}.")

        return " ".join(prompt)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.dataset[idx]
        future_trajectory: torch.Tensor = sample["future_trajectory"]  # Shape: [T, 3] (x, y, heading)
        action_tokens = trajectory_to_bev_tokens(future_trajectory)

        record: Dict[str, Any] = {
            "history_front_images": sample.get("history_front_images"),
            "history_stitched_images": sample.get("history_stitched_images"),
            "history_status": sample["history_status"],
            "history_trajectory": sample["history_trajectory"],
            "history_timestamps": sample["history_timestamps"],
            "action_tokens": action_tokens,
            "prompt_text": self._build_prompt(sample),
            "target_future_seconds": torch.tensor(self.target_future_seconds, dtype=torch.float32),
            "metadata": sample["metadata"],
        }
        if self.predict_future_image and "future_front_image" in sample:
            record["future_front_image"] = sample["future_front_image"]
        return record


def create_navsim_mmada_dataloader(
    *,
    json_path: str,
    navsim_log_path: str,
    sensor_blobs_path: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    drop_last: bool = False,
    **dataset_kwargs: Any,
) -> DataLoader:
    dataset = NavsimMMaDADataset(
        json_path=json_path,
        navsim_log_path=navsim_log_path,
        sensor_blobs_path=sensor_blobs_path,
        **dataset_kwargs,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )
