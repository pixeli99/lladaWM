#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pprint

import torch

from training.navsim_data.navsim_mmada_dataset import create_navsim_mmada_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect NavSim -> MMaDA dataloader outputs")
    parser.add_argument("json_path", type=str, help="Path to JSON/JSONL listing NavSim relative samples")
    parser.add_argument("navsim_log_path", type=str, help="Root folder containing NavSim log pickles")
    parser.add_argument("sensor_blobs_path", type=str, help="Root folder containing sensor blob files")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--history_frames", type=int, default=4)
    parser.add_argument("--future_frames", type=int, default=8)
    parser.add_argument("--future_seconds", type=float, default=4.0)
    parser.add_argument("--max_batches", type=int, default=1)
    return parser.parse_args()


def describe_batch(batch: dict[str, object]) -> None:
    history_front: torch.Tensor = batch["history_front_images"]
    future_front: torch.Tensor = batch["future_front_image"]
    history_status: torch.Tensor = batch["history_status"]
    history_traj: torch.Tensor = batch["history_trajectory"]
    action_tokens = batch["action_tokens"]
    prompt_text = batch["prompt_text"]
    metadata = batch["metadata"]

    batch_size = history_front.shape[0]
    for idx in range(batch_size):
        print(f"Sample {idx}")
        print("  history_front_images:", tuple(history_front[idx].shape))
        print("  future_front_image:", tuple(future_front[idx].shape))
        print("  history_status:", tuple(history_status[idx].shape))
        print("  history_trajectory:", tuple(history_traj[idx].shape))
        print("  action_tokens:", action_tokens[idx])
        print("  prompt_text:", prompt_text[idx])
        print("  metadata:")
        pprint.pprint(metadata[idx])
        print("  " + "-" * 32)
    print("=" * 80)


def main() -> None:
    args = parse_args()

    dataloader = create_navsim_mmada_dataloader(
        json_path=args.json_path,
        navsim_log_path=args.navsim_log_path,
        sensor_blobs_path=args.sensor_blobs_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        num_history_frames=args.history_frames,
        num_future_frames=args.future_frames,
        target_future_seconds=args.future_seconds,
        persistent_workers=False,
    )

    for idx, batch in enumerate(dataloader):
        describe_batch(batch)
        if idx + 1 >= args.max_batches:
            break


if __name__ == "__main__":
    main()
