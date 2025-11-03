#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from typing import List

import torch
from transformers import AutoTokenizer

from training.prompting_utils import UniversalPrompting
from training.navsim.action_tokens import action_token_vocab, status_to_bev_tokens


def _make_status_samples() -> torch.Tensor:
    samples = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0, 5.2, -0.3, 0.8, -1.1],
            [0.0, 0.0, 1.0, 0.0, -4.1, 2.7, -0.4, 0.2],
        ],
        dtype=torch.float32,
    )
    return samples


def test_action_tokens() -> None:
    print("# Action token sanity")
    status_batch = _make_status_samples()
    for idx, status in enumerate(status_batch):
        tokens = status_to_bev_tokens(status)
        print(f" sample {idx}: {tokens}")
    print()


def _build_history_sequences(
    uni_prompting: UniversalPrompting,
    history_codes: torch.Tensor,
) -> torch.Tensor:
    batch, num_hist, seq_len = history_codes.shape
    sep_token = uni_prompting.sptids_dict["<nav_hist_sep>"].to(history_codes.device).long().view(1)
    pad_value = torch.tensor([uni_prompting.pad_id], device=history_codes.device, dtype=torch.long)

    sequences: List[torch.Tensor] = []
    for i in range(batch):
        pieces: List[torch.Tensor] = []
        for t in range(num_hist):
            pieces.append(history_codes[i, t])
            if t < num_hist - 1:
                pieces.append(sep_token)
        sequences.append(torch.cat(pieces, dim=0))

    max_len = max(seq.shape[0] for seq in sequences)
    padded = []
    for seq in sequences:
        if seq.shape[0] < max_len:
            pad = pad_value.repeat(max_len - seq.shape[0])
            seq = torch.cat((pad, seq), dim=0)
        padded.append(seq)
    return torch.stack(padded, dim=0)


def test_navsim_prompt(tokenizer_name: str) -> None:
    print("# NavSim prompt construction with dummy tokens")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")
    uni = UniversalPrompting(
        tokenizer,
        max_text_len=128,
        ignore_id=-100,
        cond_dropout_prob=0.0,
        use_reserved_token=True,
    )
    navsim_tokens = ["<|navsim|>", "<nav_hist_sep>", "<nav_action_sep>", "<nav_future_sep>", *action_token_vocab()]
    uni.register_tokens(navsim_tokens)

    batch = 2
    num_history = 4
    vq_tokens = 16
    token_offset = len(uni.text_tokenizer)

    history_codes = torch.randint(
        low=token_offset,
        high=token_offset + 512,
        size=(batch, num_history, vq_tokens),
        dtype=torch.long,
    )
    history_tensor = _build_history_sequences(uni, history_codes)

    future_tokens = torch.randint(
        low=token_offset,
        high=token_offset + 512,
        size=(batch, vq_tokens),
        dtype=torch.long,
    )

    status_batch = _make_status_samples()
    action_token_list = [status_to_bev_tokens(status) for status in status_batch]
    action_ids = torch.tensor(
        [[uni.text_tokenizer.convert_tokens_to_ids(tok) for tok in seq] for seq in action_token_list],
        dtype=torch.long,
    )

    prompts = [
        f"Dummy prompt {i} with seed {random.randint(0, 999)}"
        for i in range(batch)
    ]

    input_ids, prompt_masks, labels = uni(
        (history_tensor, prompts, action_ids, future_tokens),
        "navsim",
    )

    print(f" input_ids shape: {tuple(input_ids.shape)}")
    print(f" prompt_masks shape: {tuple(prompt_masks.shape)}")
    print(f" labels shape: {tuple(labels.shape)}")
    print(f" sample label prefix (first 16 positions): {labels[0, :16].tolist()}")
    print("# done\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dummy checks for NavSim integration.")
    parser.add_argument(
        "--tokenizer",
        default="hf-internal-testing/llama-tokenizer",
        help="Tokenizer checkpoint to instantiate UniversalPrompting.",
    )
    args = parser.parse_args()

    test_action_tokens()
    test_navsim_prompt(args.tokenizer)


if __name__ == "__main__":
    main()
