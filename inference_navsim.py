#!/usr/bin/env python3
# Copyright 2025 MMaDA Team
# NavSim Inference Script for Overfitting Visualization

import os
import sys
import math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from pathlib import Path
import matplotlib.pyplot as plt

from transformers import AutoTokenizer
from models import MAGVITv2, MMadaModelLM, MMadaConfig
from training.prompting_utils import UniversalPrompting
from training.navsim_data.navsim_mmada_dataset import create_navsim_mmada_dataloader
from training.navsim_data.action_tokens import action_token_vocab


def load_model_and_tokenizer(checkpoint_path, config_path=None, device="cuda"):
    """加载训练好的模型"""
    print(f"Loading model from {checkpoint_path}...")
    
    # 加载配置
    if config_path is None:
        # 尝试从checkpoint目录查找
        ckpt_path = Path(checkpoint_path)
        if (ckpt_path / "config.yaml").exists():
            config_path = ckpt_path / "config.yaml"
        elif (ckpt_path.parent / "config.yaml").exists():
            config_path = ckpt_path.parent / "config.yaml"
        else:
            raise FileNotFoundError(f"config.yaml not found near {checkpoint_path}. Please specify --config_path")
    
    config = OmegaConf.load(config_path)
    print(f"Loaded config from {config_path}")
    
    # 加载tokenizer - 从配置文件指定的tokenizer路径加载
    tokenizer_path = config.model.mmada.tokenizer_path
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, 
        padding_side="left",
        trust_remote_code=True
    )
    
    # 初始化UniversalPrompting
    uni_prompting = UniversalPrompting(
        tokenizer, 
        max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=(
            "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
            "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
        ),
        ignore_id=-100, 
        cond_dropout_prob=0.0,  # 推理时不dropout
        use_reserved_token=True
    )
    
    # 注册NavSim特殊token
    navsim_special_tokens = [
        "<|navsim|>",
        "<nav_hist_sep>",
        "<nav_action_sep>",
        "<nav_future_sep>",
    ]
    navsim_token_list = navsim_special_tokens + action_token_vocab()
    uni_prompting.register_tokens(navsim_token_list)
    
    # 加载VQ模型
    print("Loading VQ model...")
    vq_model = MAGVITv2()
    if config.model.vq_model.get("pretrained_model_path", None):
        state_dict = torch.load(config.model.vq_model.pretrained_model_path)['model']
        vq_model.load_state_dict(state_dict)
    else:
        vq_model = MAGVITv2.from_pretrained(config.model.vq_model.vq_model_name)
    vq_model.eval()
    vq_model.requires_grad_(False)
    vq_model.to(device)
    
    # 加载MMaDA模型
    print("Loading MMaDA model...")
    total_vocab_size = config.model.mmada.new_vocab_size + len(navsim_token_list)
    
    model = MMadaModelLM.from_pretrained(
        config.model.mmada.pretrained_model_path,
        torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(total_vocab_size)
    model.config.vocab_size = total_vocab_size
    model.config.new_vocab_size = total_vocab_size
    model.config.embedding_size = total_vocab_size
    
    # 加载checkpoint权重
    ckpt_path = Path(checkpoint_path)
    if (ckpt_path / "unwrapped_model" / "pytorch_model.bin").exists():
        state_dict = torch.load(
            ckpt_path / "unwrapped_model" / "pytorch_model.bin", 
            map_location="cpu"
        )
        model.load_state_dict(state_dict, strict=True)
        del state_dict
    elif (ckpt_path / "unwrapped_model" / "model.safetensors.index.json").exists():
        from transformers.modeling_utils import load_sharded_checkpoint
        load_sharded_checkpoint(model, str(ckpt_path / "unwrapped_model/"))
    else:
        raise FileNotFoundError(f"No model weights found in {checkpoint_path}")
    
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    
    print("Model loaded successfully!")
    return model, vq_model, uni_prompting, config


@torch.no_grad()
def inference_navsim_sample(
    model,
    vq_model,
    uni_prompting,
    sample,
    config,
    device="cuda",
    decoding_steps=128,
    block_length=None,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
):
    """推理单个NavSim样本"""
    
    # 1. 准备历史帧
    history_front = sample["history_front_images"].unsqueeze(0).to(device)  # [1, num_hist, C, H, W]
    bsz, num_hist, channels, height, width = history_front.shape
    history_flat = history_front.reshape(bsz * num_hist, channels, height, width)
    token_offset = len(uni_prompting.text_tokenizer)
    
    # 编码历史帧为tokens
    history_codes = vq_model.get_code(history_flat).long() + token_offset
    history_codes = history_codes.view(bsz, num_hist, -1)
    
    # 2. 构建历史序列（用分隔符连接，并与训练相同地进行pad）
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
    max_history_len = max(seq.shape[0] for seq in history_sequences)
    padded_histories = []
    for seq in history_sequences:
        if seq.shape[0] < max_history_len:
            pad = pad_value.repeat(max_history_len - seq.shape[0])
            seq = torch.cat((pad, seq), dim=0)
        padded_histories.append(seq)
    history_tensor = torch.stack(padded_histories, dim=0)  # [1, seq_len]
    
    # 3. 构建输入（不包含action和future），与训练的数据流保持一致
    navsim_token = uni_prompting.sptids_dict['<|navsim|>'].to(device).long().view(1, 1)
    soi_token = uni_prompting.sptids_dict['<|soi|>'].to(device).long().view(1, 1)
    eoi_token = uni_prompting.sptids_dict['<|eoi|>'].to(device).long().view(1, 1)
    action_sep_token = uni_prompting.sptids_dict['<nav_action_sep>'].to(device).long().view(1, 1)
    future_sep_token = uni_prompting.sptids_dict['<nav_future_sep>'].to(device).long().view(1, 1)
    
    # 编码prompt text，遵循训练时的padding/truncation方式
    prompt_text = sample["prompt_text"]
    prompt_token_ids = uni_prompting.text_tokenizer([prompt_text], truncation=True)['input_ids'][0]
    if len(prompt_token_ids) == 0:
        prompt_token_ids = [uni_prompting.text_tokenizer.bos_token_id]
    elif prompt_token_ids[0] != uni_prompting.text_tokenizer.bos_token_id:
        prompt_token_ids = [uni_prompting.text_tokenizer.bos_token_id] + prompt_token_ids
    prompt_token_ids = prompt_token_ids + [uni_prompting.text_tokenizer.eos_token_id]
    
    # max_text_len = max(1, uni_prompting.max_text_len - 1)
    # if max_text_len >= len(prompt_token_ids):
    #     prompt_token_ids = prompt_token_ids + [uni_prompting.text_tokenizer.eos_token_id] * (max_text_len - len(prompt_token_ids))
    # else:
    #     prompt_token_ids = prompt_token_ids[:max_text_len - 1] + [uni_prompting.text_tokenizer.eos_token_id]
    prompt_ids = torch.tensor(prompt_token_ids, device=device, dtype=torch.long).unsqueeze(0)
    
    # 构建初始输入
    input_ids = torch.cat([
        navsim_token,
        soi_token,
        history_tensor,
        eoi_token,
        prompt_ids,
        action_sep_token,
    ], dim=1)
    
    print(f"\n{'='*80}")
    print(f"Input sequence length: {input_ids.shape[1]}")
    print(f"History frames: {num_hist}")
    print(f"Prompt: {prompt_text}")
    
    # 4. 使用掩码生成器（非自回归）解码后续tokens
    gt_action_tokens = sample["action_tokens"]
    gt_action_token_ids = [
        uni_prompting.text_tokenizer.convert_tokens_to_ids(token) for token in gt_action_tokens
    ]
    expected_action_tokens = len(gt_action_token_ids)
    num_vq_tokens = config.model.mmada.num_vq_tokens
    suffix_tokens = expected_action_tokens + 1 + 1 + num_vq_tokens + 1  # actions + <nav_future_sep> + <|soi|> + future + <|eoi|>
    
    print(f"Expected action tokens: {expected_action_tokens}")
    print(f"Total tokens to generate (actions + future + specials): {suffix_tokens}")
    
    if block_length is None or block_length <= 0 or suffix_tokens % block_length != 0:
        if block_length not in (None, 0):
            print(f"Adjusting block_length from {block_length} to {suffix_tokens} to fit suffix length.")
        block_length = suffix_tokens
    num_blocks = suffix_tokens // block_length
    if decoding_steps % num_blocks != 0:
        adjusted_steps = math.ceil(decoding_steps / num_blocks) * num_blocks
        print(f"Adjusting decoding steps from {decoding_steps} to {adjusted_steps} to align with {num_blocks} block(s).")
        decoding_steps = adjusted_steps
    
    mask_token_id = getattr(model.config, "mask_token_id", None)
    if mask_token_id is None:
        print("Warning: model.config.mask_token_id is None, defaulting to 126336.")
        mask_token_id = 126336
    
    generated_full = model.mmu_generate(
        idx=input_ids,
        max_new_tokens=suffix_tokens,
        steps=decoding_steps,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
        mask_id=mask_token_id,
    )
    
    generated_suffix = generated_full[:, input_ids.shape[1]:]
    generated_action_tokens = generated_suffix[0, :expected_action_tokens].tolist()
    nav_future_sep_pred = generated_suffix[0, expected_action_tokens].item()
    future_soi_pred = generated_suffix[0, expected_action_tokens + 1].item()
    future_token_slice = generated_suffix[0, expected_action_tokens + 2:expected_action_tokens + 2 + num_vq_tokens]
    future_eoi_pred = generated_suffix[0, expected_action_tokens + 2 + num_vq_tokens].item()
    
    nav_future_sep_id = uni_prompting.sptids_dict['<nav_future_sep>'].item()
    soi_id = uni_prompting.sptids_dict['<|soi|>'].item()
    eoi_id = uni_prompting.sptids_dict['<|eoi|>'].item()
    
    if nav_future_sep_pred != nav_future_sep_id:
        print(f"Warning: expected <nav_future_sep> (id={nav_future_sep_id}) but got id={nav_future_sep_pred}.")
    if future_soi_pred != soi_id:
        print(f"Warning: expected future <|soi|> (id={soi_id}) but got id={future_soi_pred}.")
    if future_eoi_pred != eoi_id:
        print(f"Warning: expected future <|eoi|> (id={eoi_id}) but got id={future_eoi_pred}.")
    
    generated_future_tokens = (future_token_slice - token_offset).clamp(min=0, max=config.model.mmada.codebook_size - 1).tolist()
    
    # 6. 解码图像
    generated_future_tokens_tensor = torch.tensor(
        [generated_future_tokens], 
        device=device, 
        dtype=torch.long
    )
    pred_future_image = vq_model.decode_code(generated_future_tokens_tensor)
    
    # 解码GT未来图像
    gt_future_front = sample["future_front_image"].unsqueeze(0).to(device)
    gt_future_tokens = vq_model.get_code(gt_future_front).long()
    recon_gt_future = vq_model.decode_code(gt_future_tokens)
    
    return {
        'generated_action_tokens': generated_action_tokens,
        'gt_action_tokens': sample["action_tokens"],
        'gt_action_token_ids': gt_action_token_ids,
        'pred_future_image': pred_future_image,
        'gt_future_image': gt_future_front,
        'recon_gt_future': recon_gt_future,
        'history_images': history_front,
        'prompt_text': prompt_text,
    }


def visualize_results(results, uni_prompting, save_path="navsim_inference_result.png"):
    """可视化推理结果"""
    
    # 1. 打印动作对比
    print(f"\n{'='*80}")
    print("ACTION COMPARISON:")
    print(f"{'='*80}")
    
    tokenizer = uni_prompting.text_tokenizer
    token_offset = len(tokenizer)
    gt_actions = results['gt_action_tokens']
    gt_action_ids = results.get('gt_action_token_ids', [])
    pred_action_ids = results['generated_action_tokens']
    pred_action_tokens = []
    for token_id in pred_action_ids:
        if 0 <= token_id < len(tokenizer):
            pred_action_tokens.append(tokenizer.convert_ids_to_tokens([token_id])[0])
        elif token_id >= token_offset:
            pred_action_tokens.append(f"<vq_{token_id - token_offset}>")
        else:
            pred_action_tokens.append(f"ID_{token_id}")
    
    print(f"\n{'Index':<8} {'GT Action':<30} {'Predicted Token':<25} {'Match'}")
    print("-" * 80)
    
    num_actions = min(len(gt_action_ids), len(pred_action_ids)) if gt_action_ids else min(len(gt_actions), len(pred_action_ids))
    matches = 0
    
    for i in range(max(len(gt_actions), len(pred_action_ids))):
        if i < len(gt_actions) and i < len(pred_action_ids):
            gt_token = gt_actions[i]
            pred_token = pred_action_tokens[i]
            if gt_action_ids:
                match = "✓" if gt_action_ids[i] == pred_action_ids[i] else "✗"
            else:
                match = "✓" if gt_token == pred_token else "✗"
            if match == "✓":
                matches += 1
            print(f"{i:<8} {gt_token:<30} {pred_token:<25} {match}")
        elif i < len(gt_actions):
            print(f"{i:<8} {gt_actions[i]:<30} {'MISSING':<25} ✗")
        else:
            pred_token = pred_action_tokens[i] if i < len(pred_action_tokens) else str(pred_action_ids[i])
            print(f"{i:<8} {'MISSING':<30} {pred_token:<25} ✗")
    
    accuracy = matches / num_actions * 100 if num_actions > 0 else 0
    print(f"\nAction Accuracy: {matches}/{num_actions} = {accuracy:.2f}%")
    
    # 2. 可视化图像
    print(f"\n{'='*80}")
    print("IMAGE VISUALIZATION:")
    print(f"{'='*80}")
    
    def tensor_to_image(tensor):
        """将tensor转换为numpy图像"""
        img = tensor.squeeze(0).cpu()
        img = torch.clamp((img + 1.0) / 2.0, 0.0, 1.0)
        img = img.permute(1, 2, 0).numpy()
        return (img * 255).astype(np.uint8)
    
    # 准备历史帧
    num_hist = results['history_images'].shape[1]
    history_imgs = [
        tensor_to_image(results['history_images'][0, i]) 
        for i in range(num_hist)
    ]
    
    # 准备未来帧
    gt_future_img = tensor_to_image(results['gt_future_image'])
    pred_future_img = tensor_to_image(results['pred_future_image'])
    recon_gt_img = tensor_to_image(results['recon_gt_future'])
    
    # 创建可视化
    fig, axes = plt.subplots(2, num_hist + 1, figsize=(4 * (num_hist + 1), 8))
    
    # 第一行：历史帧 + GT未来帧
    for i in range(num_hist):
        axes[0, i].imshow(history_imgs[i])
        axes[0, i].set_title(f'History Frame {i+1}')
        axes[0, i].axis('off')
    axes[0, num_hist].imshow(gt_future_img)
    axes[0, num_hist].set_title('GT Future Frame')
    axes[0, num_hist].axis('off')
    
    # 第二行：历史帧（重复） + 预测未来帧
    for i in range(num_hist):
        axes[1, i].imshow(history_imgs[i])
        axes[1, i].set_title(f'History Frame {i+1}')
        axes[1, i].axis('off')
    axes[1, num_hist].imshow(pred_future_img)
    axes[1, num_hist].set_title('Predicted Future Frame')
    axes[1, num_hist].axis('off')
    
    plt.suptitle(f"NavSim Inference Result\nPrompt: {results['prompt_text']}", 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    
    # 计算图像相似度（简单的MSE）
    mse = np.mean((gt_future_img.astype(float) - pred_future_img.astype(float)) ** 2)
    print(f"Future Frame MSE: {mse:.2f}")
    
    return fig


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="NavSim Inference Script")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to checkpoint directory")
    parser.add_argument("--config_path", type=str, default=None,
                       help="Path to training config (will auto-detect from checkpoint if not provided)")
    parser.add_argument("--sample_idx", type=int, default=0,
                       help="Index of sample to visualize")
    parser.add_argument("--output", type=str, default="navsim_inference_result.png",
                       help="Output image path")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--decoding_steps", type=int, default=128,
                       help="Mask decoding steps for NavSim suffix generation")
    parser.add_argument("--block_length", type=int, default=None,
                       help="Block length for NavSim generation (must divide generated length). Defaults to suffix length.")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature (0 = greedy)")
    parser.add_argument("--cfg_scale", type=float, default=0.0,
                       help="Classifier-free guidance scale")
    parser.add_argument("--remasking", type=str, default="low_confidence",
                       choices=["low_confidence", "random"],
                       help="Remasking strategy for iterative decoding")
    
    args = parser.parse_args()
    
    # 1. 加载模型
    model, vq_model, uni_prompting, config = load_model_and_tokenizer(
        args.checkpoint,
        config_path=args.config_path,
        device=args.device
    )
    
    # 2. 加载数据
    print("\nLoading NavSim dataset...")
    navsim_cfg = config.dataset.params.navsim
    navsim_params = OmegaConf.to_container(navsim_cfg, resolve=True)
    
    navsim_loader = create_navsim_mmada_dataloader(
        json_path=navsim_params["json_path"],
        navsim_log_path=navsim_params["navsim_log_path"],
        sensor_blobs_path=navsim_params["sensor_blobs_path"],
        batch_size=1,  # 单个样本
        num_workers=0,
        shuffle=False,
        pin_memory=False,
        persistent_workers=False,
        drop_last=False,
        num_history_frames=navsim_params.get("num_history_frames", 4),
        num_future_frames=navsim_params.get("num_future_frames", 8),
        target_future_seconds=navsim_params.get("target_future_seconds", 4.0),
    )
    
    # 3. 获取指定样本
    dataset = navsim_loader.dataset
    if args.sample_idx < 0 or args.sample_idx >= len(dataset):
        print(f"Sample {args.sample_idx} not found! Dataset size: {len(dataset)}")
        return
    print(f"Fetching sample {args.sample_idx} directly from dataset (size={len(dataset)})...")
    sample = dataset[args.sample_idx]
    
    # 4. 推理
    print(f"\n{'='*80}")
    print("RUNNING INFERENCE...")
    print(f"{'='*80}")
    
    results = inference_navsim_sample(
        model,
        vq_model,
        uni_prompting,
        sample,
        config,
        device=args.device,
        decoding_steps=args.decoding_steps,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        remasking=args.remasking,
    )
    
    # 5. 可视化
    visualize_results(results, uni_prompting, save_path=args.output)
    
    print(f"\n{'='*80}")
    print("INFERENCE COMPLETED!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
