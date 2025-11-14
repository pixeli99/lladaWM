from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import NAVSIM_INTERVAL_LENGTH, SceneFilter, SensorConfig

logger = logging.getLogger(__name__)


def _load_sample_list(json_path: str) -> List[str]:
    """Load sample relative paths from JSON array or newline separated text."""
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [ln.strip() for ln in f if ln.strip()]
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {json_path}, got {type(data)}")
    return data


def _default_transform() -> Callable[[Image.Image], torch.Tensor]:
    return transforms.Compose([transforms.ToTensor()])


class NavsimOnlineDataset(Dataset):
    """Stream NavSim scenes on-the-fly and expose history / future tensors for MMaDA."""

    def __init__(
        self,
        json_path: str,
        navsim_log_path: str,
        sensor_blobs_path: str,
        *,
        num_history_frames: int = 4,
        num_future_frames: int = 8,
        target_future_steps: Optional[int] = None,
        target_future_seconds: Optional[float] = 4.0,
        history_image_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        future_image_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        stitched_image_size: Tuple[int, int] = (1024, 256),  # (width, height)
        front_image_size: Tuple[int, int] = (512, 256),
        include_front: bool = True,
        include_stitched: bool = True,
        drop_missing: bool = True,
    ) -> None:
        super().__init__()

        if not include_front and not include_stitched:
            raise ValueError("At least one of include_front / include_stitched must be True.")

        self.sample_rel = _load_sample_list(json_path)
        random.shuffle(self.sample_rel)
        logger.info("[NavSim] Loaded %d sample entries from %s", len(self.sample_rel), json_path)

        self.num_history_frames = int(num_history_frames)
        if self.num_history_frames < 1:
            raise ValueError("num_history_frames must be >= 1")

        self._dt = NAVSIM_INTERVAL_LENGTH
        if target_future_steps is None:
            if target_future_seconds is None:
                target_future_steps = num_future_frames
            else:
                target_future_steps = max(1, int(round(target_future_seconds / self._dt)))
        self.target_future_steps = int(target_future_steps)
        self.future_delta_seconds = self.target_future_steps * self._dt

        self.num_future_frames = max(int(num_future_frames), self.target_future_steps)
        self.target_future_index = self.num_history_frames - 1 + self.target_future_steps

        if self.target_future_steps > self.num_future_frames:
            raise ValueError(
                f"target_future_steps ({self.target_future_steps}) must be <= num_future_frames ({self.num_future_frames})"
            )

        self.include_front = include_front
        self.include_stitched = include_stitched
        self.drop_missing = drop_missing

        self.history_transform = history_image_transform or _default_transform()
        self.future_transform = future_image_transform or self.history_transform

        self.stitched_size = stitched_image_size
        self.front_size = front_image_size

        history_indices = list(range(self.num_history_frames))
        required_indices = sorted(set(history_indices + [self.target_future_index]))

        self.sensor_config = SensorConfig(
            cam_f0=required_indices,
            cam_l0=required_indices if include_stitched else False,
            cam_l1=False,
            cam_l2=False,
            cam_r0=required_indices if include_stitched else False,
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=False,
        )

        data_root = Path(navsim_log_path)
        candidate_roots = [data_root, data_root / "trainval", data_root / "train", data_root / "val"]
        valid_roots: List[Path] = []
        for root in candidate_roots:
            try:
                if root.exists() and any((p.is_file() and p.suffix == ".pkl") for p in root.iterdir()):
                    valid_roots.append(root)
            except Exception:
                continue
        if not valid_roots:
            valid_roots = [data_root]

        scene_filter = SceneFilter(
            num_history_frames=self.num_history_frames,
            num_future_frames=self.num_future_frames,
            frame_interval=1,
            tokens=None,
            has_route=True,
        )

        self._loaders: List[SceneLoader] = []
        self._token_to_loader: Dict[str, int] = {}

        sensor_blobs = Path(sensor_blobs_path)
        for root in valid_roots:
            loader = SceneLoader(
                sensor_blobs_path=sensor_blobs,
                data_path=root,
                scene_filter=scene_filter,
                sensor_config=self.sensor_config,
            )
            if len(loader) == 0:
                continue
            loader_id = len(self._loaders)
            self._loaders.append(loader)
            for token in loader.tokens:
                if token not in self._token_to_loader:
                    self._token_to_loader[token] = loader_id

        self._index: List[Dict[str, Any]] = []
        dropped = 0
        for rel in self.sample_rel:
            rel = rel.strip("/")
            if not rel:
                continue
            token = rel.split("/")[-1]
            lid = self._token_to_loader.get(token)
            if lid is None:
                dropped += 1
                if not self.drop_missing:
                    raise KeyError(f"Token {token} not found in NavSim logs")
                continue
            self._index.append({"rel": rel, "token": token, "loader_id": lid})
        if dropped > 0:
            logger.warning("[NavSim] Dropped %d samples not present in provided logs.", dropped)
        logger.info("[NavSim] %d samples remain after filtering.", len(self._index))

    def __len__(self) -> int:
        return len(self._index)

    @staticmethod
    def _status_tensor(ego_status) -> torch.Tensor:
        driving = torch.as_tensor(np.asarray(ego_status.driving_command), dtype=torch.float32)
        velocity = torch.as_tensor(np.asarray(ego_status.ego_velocity), dtype=torch.float32)
        acceleration = torch.as_tensor(np.asarray(ego_status.ego_acceleration), dtype=torch.float32)
        return torch.cat([driving, velocity, acceleration], dim=0)

    def _resize_front(self, image: np.ndarray) -> np.ndarray:
        width, height = self.front_size
        return cv2.resize(image, (width, height))

    def _resize_stitched(self, image: np.ndarray) -> np.ndarray:
        width, height = self.stitched_size
        return cv2.resize(image, (width, height))

    @staticmethod
    def _ensure_image(name: str, image: Optional[np.ndarray]) -> np.ndarray:
        if image is None:
            raise ValueError(f"Camera image {name} is missing (check SensorConfig).")
        return image

    def _build_front_tensor(self, frame) -> torch.Tensor:
        cam = self._ensure_image("cam_f0", frame.cameras.cam_f0.image)
        resized = self._resize_front(cam)
        pil_image = Image.fromarray(resized.astype(np.uint8))
        return self.history_transform(pil_image)

    def _build_future_front_tensor(self, frame) -> torch.Tensor:
        cam = self._ensure_image("cam_f0", frame.cameras.cam_f0.image)
        resized = self._resize_front(cam)
        pil_image = Image.fromarray(resized.astype(np.uint8))
        return self.future_transform(pil_image)

    def _build_stitched_tensor(self, frame, *, future: bool = False) -> torch.Tensor:
        cams = frame.cameras
        l0 = self._ensure_image("cam_l0", cams.cam_l0.image)
        f0 = self._ensure_image("cam_f0", cams.cam_f0.image)
        r0 = self._ensure_image("cam_r0", cams.cam_r0.image)

        l0_crop = l0[28:-28, 416:-416]
        f0_crop = f0[28:-28]
        r0_crop = r0[28:-28, 416:-416]
        stitched = np.concatenate([l0_crop, f0_crop, r0_crop], axis=1)
        resized = self._resize_stitched(stitched)
        pil_image = Image.fromarray(resized.astype(np.uint8))
        transform = self.future_transform if future else self.history_transform
        return transform(pil_image)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self._index[idx]
        loader = self._loaders[rec["loader_id"]]
        scene = loader.get_scene_from_token(rec["token"])

        history_front: List[torch.Tensor] = []
        history_stitched: List[torch.Tensor] = []
        history_status: List[torch.Tensor] = []
        history_timestamps: List[int] = []

        for frame_idx in range(self.num_history_frames):
            frame = scene.frames[frame_idx]
            if self.include_front:
                history_front.append(self._build_front_tensor(frame))
            if self.include_stitched:
                history_stitched.append(self._build_stitched_tensor(frame))
            history_status.append(self._status_tensor(frame.ego_status))
            history_timestamps.append(int(frame.timestamp))

        if self.include_front:
            history_front_tensor = torch.stack(history_front, dim=0)
        if self.include_stitched:
            history_stitched_tensor = torch.stack(history_stitched, dim=0)

        history_status_tensor = torch.stack(history_status, dim=0)
        history_timestamp_tensor = torch.as_tensor(history_timestamps, dtype=torch.long)

        history_traj = torch.from_numpy(
            scene.get_history_trajectory(self.num_history_frames).poses.astype(np.float32)
        )
        future_traj = torch.from_numpy(
            scene.get_future_trajectory(self.num_future_frames).poses.astype(np.float32)
        )

        future_frame = scene.frames[self.target_future_index]
        future_action = torch.as_tensor(np.asarray(future_frame.ego_status.driving_command), dtype=torch.float32)
        future_status = self._status_tensor(future_frame.ego_status)
        future_timestamp = torch.tensor(int(future_frame.timestamp), dtype=torch.long)

        sample: Dict[str, Any] = {
            "token": rec["token"],
            "rel_path": rec["rel"],
            "history_status": history_status_tensor,
            "history_timestamps": history_timestamp_tensor,
            "history_trajectory": history_traj,
            "future_trajectory": future_traj,
            "future_action": future_action,
            "future_status": future_status,
            "future_timestamp": future_timestamp,
            "future_delta_seconds": torch.tensor(self.future_delta_seconds, dtype=torch.float32),
            "metadata": {
                "log_name": scene.scene_metadata.log_name,
                "scene_token": scene.scene_metadata.scene_token,
                "map_name": scene.scene_metadata.map_name,
            },
        }

        if self.include_front:
            sample["history_front_images"] = history_front_tensor
            future_front = self._build_future_front_tensor(future_frame)
            sample["future_front_image"] = future_front

        if self.include_stitched:
            sample["history_stitched_images"] = history_stitched_tensor
            future_stitched = self._build_stitched_tensor(future_frame, future=True)
            sample["future_stitched_image"] = future_stitched

        return sample


class NavsimFuturePredictionDataset(Dataset):
    """Lightweight wrapper that reshapes NavSim samples into (inputs, targets)."""

    def __init__(
        self,
        json_path: str,
        navsim_log_path: str,
        sensor_blobs_path: str,
        *,
        include_stitched: bool = False,
        include_front: bool = True,
        **dataset_kwargs: Any,
    ) -> None:
        base_kwargs = dict(dataset_kwargs)
        base_kwargs["include_front"] = include_front
        base_kwargs["include_stitched"] = include_stitched
        if not base_kwargs["include_front"]:
            raise ValueError("Future prediction dataset requires include_front=True to expose target images.")

        self.dataset = NavsimOnlineDataset(
            json_path=json_path,
            navsim_log_path=navsim_log_path,
            sensor_blobs_path=sensor_blobs_path,
            **base_kwargs,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.dataset[idx]

        inputs: Dict[str, Any] = {
            "front_images": sample["history_front_images"],
            "status": sample["history_status"],
            "trajectory": sample["history_trajectory"],
            "timestamps": sample["history_timestamps"],
        }
        if "history_stitched_images" in sample:
            inputs["stitched_images"] = sample["history_stitched_images"]

        targets: Dict[str, Any] = {
            "action": sample["future_action"],
            "status": sample["future_status"],
            "trajectory": sample["future_trajectory"],
            "timestamp": sample["future_timestamp"],
            "delta_seconds": sample["future_delta_seconds"],
        }
        if "future_front_image" in sample:
            targets["front_image"] = sample["future_front_image"]
        if "future_stitched_image" in sample:
            targets["stitched_image"] = sample["future_stitched_image"]

        metadata = {
            "token": sample["token"],
            "rel_path": sample["rel_path"],
        }
        metadata.update(sample["metadata"])

        return {"inputs": inputs, "targets": targets, "metadata": metadata}


def create_navsim_dataloader(
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
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
    **dataset_kwargs: Any,
) -> DataLoader:
    """Factory helper that instantiates ``NavsimOnlineDataset`` and wraps it in a dataloader."""
    dataset = NavsimOnlineDataset(
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
        collate_fn=collate_fn,
    )


def create_navsim_future_prediction_dataloader(
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
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
    **dataset_kwargs: Any,
) -> DataLoader:
    """Create dataloader that yields dicts with ``inputs`` and ``targets`` keys for future prediction."""
    dataset = NavsimFuturePredictionDataset(
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
        collate_fn=collate_fn,
    )
