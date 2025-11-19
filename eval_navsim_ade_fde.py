#!/usr/bin/env python3
# Copyright 2025 MMaDA Team
# NavSim Evaluation Script for ADE/FDE

import os
import sys
import math
import torch
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from models import MAGVITv2, MMadaModelLM
from training.prompting_utils import UniversalPrompting
from training.navsim_data.navsim_mmada_dataset import create_navsim_mmada_dataloader
from training.navsim_data.action_tokens import action_token_vocab, _BEV_BINS

# Import load_model_and_tokenizer from inference_navsim.py
try:
    from inference_navsim import load_model_and_tokenizer
except ImportError:
    print("Could not import load_model_and_tokenizer from inference_navsim.py")
    sys.exit(1)

def decode_bev_token(token_str):
    """Convert <bev_idx> token string to float value."""
    if not isinstance(token_str, str):
        return None
    if not token_str.startswith("<bev_") or not token_str.endswith(">"):
        return None
    try:
        idx = int(token_str[5:-1])
        if 0 <= idx < len(_BEV_BINS):
            return float(_BEV_BINS[idx])
    except ValueError:
        pass
    return None

def decode_trajectory_from_tokens(tokens):
    """
    Decode list of token strings to trajectory (T, 2).
    tokens: list of strings like ["<bev_100>", "<bev_101>", ...]
    """
    coords = []
    current_point = []
    
    for token in tokens:
        val = decode_bev_token(token)
        if val is not None:
            current_point.append(val)
            if len(current_point) == 2:
                coords.append(current_point)
                current_point = []
    
    if not coords:
        return np.zeros((0, 2))
        
    return np.array(coords)

def calculate_ade_fde(pred_traj, gt_traj):
    """
    Calculate ADE and FDE.
    pred_traj: (T, 2)
    gt_traj: (T, 2)
    """
    min_len = min(len(pred_traj), len(gt_traj))
    if min_len == 0:
        return None, None
    
    # Truncate to same length
    p = pred_traj[:min_len]
    g = gt_traj[:min_len]
    
    errors = np.sqrt(np.sum((p - g)**2, axis=1))
    ade = np.mean(errors)
    fde = errors[-1]
    
    return ade, fde

@torch.no_grad()
def predict_trajectory(
    model,
    vq_model,
    uni_prompting,
    sample,
    config,
    device="cuda",
    temperature=0.0,
    cfg_scale=0.0,
):
    """Predict trajectory for a single sample."""
    
    # 1. Prepare History
    history_front = sample["history_front_images"].unsqueeze(0).to(device)  # [1, num_hist, C, H, W]
    bsz, num_hist, channels, height, width = history_front.shape
    history_flat = history_front.reshape(bsz * num_hist, channels, height, width)
    token_offset = len(uni_prompting.text_tokenizer)

    # Encode history frames
    history_codes = vq_model.get_code(history_flat).long() + token_offset
    history_codes = history_codes.view(bsz, num_hist, -1)
    
    # 2. Build History Sequence
    hist_sep_token = uni_prompting.sptids_dict['<nav_hist_sep>'].to(device).long().view(1)
    pad_value = torch.tensor([uni_prompting.pad_id], device=device, dtype=torch.long)
    history_sequences = []
    for i in range(bsz):
        pieces = []
        for t in range(num_hist):
            pieces.append(history_codes[i, t])
            if t < num_hist - 1:
                pieces.append(hist_sep_token)
        seq = torch.cat(pieces, dim=0)
        history_sequences.append(seq)
    
    # Pad history (though batch size is 1 here)
    max_history_len = max(seq.shape[0] for seq in history_sequences)
    padded_histories = []
    for seq in history_sequences:
        if seq.shape[0] < max_history_len:
            pad = pad_value.repeat(max_history_len - seq.shape[0])
            seq = torch.cat((pad, seq), dim=0)
        padded_histories.append(seq)
    history_tensor = torch.stack(padded_histories, dim=0)  # [1, seq_len]
    
    # 3. Build Input
    navsim_token = uni_prompting.sptids_dict['<|navsim|>'].to(device).long().view(1, 1)
    soi_token = uni_prompting.sptids_dict['<|soi|>'].to(device).long().view(1, 1)
    eoi_token = uni_prompting.sptids_dict['<|eoi|>'].to(device).long().view(1, 1)
    action_sep_token = uni_prompting.sptids_dict['<nav_action_sep>'].to(device).long().view(1, 1)
    
    # Encode prompt
    prompt_text = sample["prompt_text"]
    prompt_token_ids = uni_prompting.text_tokenizer([prompt_text], truncation=True)['input_ids'][0]
    if len(prompt_token_ids) == 0:
        prompt_token_ids = [uni_prompting.text_tokenizer.bos_token_id]
    elif prompt_token_ids[0] != uni_prompting.text_tokenizer.bos_token_id:
        prompt_token_ids = [uni_prompting.text_tokenizer.bos_token_id] + prompt_token_ids
    prompt_token_ids = prompt_token_ids + [uni_prompting.text_tokenizer.eos_token_id]
    
    max_text_len = max(1, uni_prompting.max_text_len - 1)
    if max_text_len >= len(prompt_token_ids):
        prompt_token_ids = prompt_token_ids + [uni_prompting.text_tokenizer.eos_token_id] * (max_text_len - len(prompt_token_ids))
    else:
        prompt_token_ids = prompt_token_ids[:max_text_len - 1] + [uni_prompting.text_tokenizer.eos_token_id]
    prompt_ids = torch.tensor(prompt_token_ids, device=device, dtype=torch.long).unsqueeze(0)
    
    # Construct initial input
    input_ids = torch.cat([
        navsim_token,
        soi_token,
        history_tensor,
        eoi_token,
        prompt_ids,
        action_sep_token,
    ], dim=1)
    
    # 4. Generate Actions
    mask_token_id = getattr(model.config, "mask_token_id", 126336)
    
    action_token_ids = [
        uni_prompting.text_tokenizer.convert_tokens_to_ids(token)
        for token in action_token_vocab()
    ]
    
    # Check if contiguous
    action_token_ids_sorted = sorted(action_token_ids)
    if len(action_token_ids) > 1:
        if action_token_ids_sorted[-1] - action_token_ids_sorted[0] + 1 != len(action_token_ids):
            print(f"Warning: Action token IDs are not contiguous! Range: {action_token_ids_sorted[0]}-{action_token_ids_sorted[-1]}, Count: {len(action_token_ids)}")

    action_id_min = min(action_token_ids)
    action_id_max = max(action_token_ids)
    
    # We assume 16 tokens for 8 waypoints (x, y)
    num_action_tokens = 16
    
    action_start = input_ids.shape[1]
    action_clamp_ranges = [
        (
            action_start,
            action_start + num_action_tokens,
            action_id_min,
            action_id_max,
        )
    ]
    
    # Generate
    actions_full = model.mmu_generate(
        idx=input_ids,
        max_new_tokens=num_action_tokens, 
        steps=num_action_tokens, 
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking="low_confidence",
        mask_id=mask_token_id,
        clamp_ranges=action_clamp_ranges,
    )
    
    generated_actions_tensor = actions_full[:, action_start:action_start + num_action_tokens]
    generated_action_ids = generated_actions_tensor[0].tolist()
    
    # Convert IDs to tokens
    generated_tokens = []
    for tid in generated_action_ids:
        token = uni_prompting.text_tokenizer.convert_ids_to_tokens(tid)
        generated_tokens.append(token)
        
    # Decode to trajectory
    pred_traj = decode_trajectory_from_tokens(generated_tokens)
    
    # Get GT trajectory
    gt_action_tokens = sample["action_tokens"]
    gt_traj = decode_trajectory_from_tokens(gt_action_tokens)
    
    return pred_traj, gt_traj, generated_tokens, gt_action_tokens

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="NavSim ADE/FDE Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config_path", type=str, default=None, help="Path to config")
    parser.add_argument("--output", type=str, default="navsim_eval_results.txt", help="Output file")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--cfg_scale", type=float, default=0.0, help="CFG Scale")
    
    args = parser.parse_args()
    
    # Load model
    model, vq_model, uni_prompting, config = load_model_and_tokenizer(
        args.checkpoint,
        config_path=args.config_path,
        device=args.device
    )
    
    # Load data
    print("\nLoading NavSim dataset...")
    navsim_cfg = config.dataset.params.navsim
    navsim_params = OmegaConf.to_container(navsim_cfg, resolve=True)
    
    navsim_loader = create_navsim_mmada_dataloader(
        json_path=navsim_params["json_path"],
        navsim_log_path=navsim_params["navsim_log_path"],
        sensor_blobs_path=navsim_params["sensor_blobs_path"],
        batch_size=1,
        num_workers=4,
        shuffle=False,
        pin_memory=False,
        persistent_workers=False,
        drop_last=False,
        num_history_frames=navsim_params.get("num_history_frames", 4),
        num_future_frames=navsim_params.get("num_future_frames", 8),
        target_future_seconds=navsim_params.get("target_future_seconds", 4.0),
    )
    
    dataset = navsim_loader.dataset
    num_samples = len(dataset)
    if args.max_samples is not None:
        num_samples = min(num_samples, args.max_samples)
    
    print(f"Evaluating on {num_samples} samples...")
    
    ade_list = []
    fde_list = []
    
    with open(args.output, "w") as f:
        f.write("Index\tADE\tFDE\tGT_Tokens\tPred_Tokens\n")
        
        for i in tqdm(range(num_samples)):
            sample = dataset[i]
            
            pred_traj, gt_traj, pred_tokens, gt_tokens = predict_trajectory(
                model, vq_model, uni_prompting, sample, config,
                device=args.device,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale
            )
            
            ade, fde = calculate_ade_fde(pred_traj, gt_traj)
            
            if ade is not None:
                ade_list.append(ade)
                fde_list.append(fde)
                
                # Log to file
                gt_str = " ".join(gt_tokens)
                pred_str = " ".join(pred_tokens)
                f.write(f"{i}\t{ade:.4f}\t{fde:.4f}\t{gt_str}\t{pred_str}\n")
                
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    if ade_list:
        print(f"Mean ADE: {np.mean(ade_list):.4f}")
        print(f"Mean FDE: {np.mean(fde_list):.4f}")
    else:
        print("No valid results.")
    print(f"Detailed results saved to {args.output}")

if __name__ == "__main__":
    main()

