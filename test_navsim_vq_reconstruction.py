#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 NavSim 图像在不同分辨率下的 VQ-VAE 重建效果
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf
from torchvision import transforms

from models import MAGVITv2
from training.navsim_data.navsim_mmada_dataset import create_navsim_mmada_dataloader
from vision_tokenizer import build_vision_tokenizer

MODEL_DISPLAY_NAMES = {
    "magvit": "MAGVITv2",
    "vision_tokenizer": "Vision Tokenizer",
}


def load_vq_model(vq_model_path, device="cuda"):
    """加载 VQ-VAE 模型"""
    print(f"Loading VQ model from {vq_model_path}...")
    vq_model = MAGVITv2.from_pretrained(vq_model_path).to(device)
    vq_model.eval()
    vq_model.requires_grad_(False)
    print("VQ model loaded successfully!")
    return vq_model


def resize_and_normalize(image_pil, resolution):
    """
    将 PIL 图像 resize 到指定分辨率并归一化
    Args:
        image_pil: PIL Image
        resolution: int, 目标分辨率的高度（宽度为高度的2倍，比例1:2）
    Returns:
        torch.Tensor: [1, 3, H, W], 范围 [-1, 1]
    """
    height = resolution
    width = resolution * 2
    transform = transforms.Compose([
        transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform(image_pil).unsqueeze(0)


def denormalize_image(tensor):
    """
    将归一化后的 tensor 转换回 [0, 255] 的 numpy 图像
    Args:
        tensor: torch.Tensor, [1, 3, H, W] 或 [3, H, W], 范围 [-1, 1]
    Returns:
        numpy.ndarray: [H, W, 3], uint8, 范围 [0, 255]
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # 从 [-1, 1] 转换到 [0, 1]
    img = torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)
    img = img.permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)


def reconstruct_at_resolution(
    vq_model,
    image_pil,
    resolution,
    device="cuda",
    vision_tokenizer=None,
    vision_tokenizer_device=None,
):
    """
    在指定分辨率下进行 VQ-VAE 重建
    Args:
        vq_model: VQ-VAE 模型
        image_pil: PIL Image (原始图像)
        resolution: int, 目标分辨率的高度（宽度为高度的2倍）
        device: str
        vision_tokenizer: optional vision tokenizer model
        vision_tokenizer_device: device for the vision tokenizer
    Returns:
        dict: {model_name: result_dict}
    """
    results = {}
    # Resize 和归一化（保持 CPU，用于多个模型）
    base_tensor = resize_and_normalize(image_pil, resolution)
    original_np = denormalize_image(base_tensor.cpu())
    
    # 编码（MAGVIT）
    img_tensor = base_tensor.to(device)
    with torch.no_grad():
        codes = vq_model.get_code(img_tensor)
        num_tokens = codes.shape[1]
        
        downsample_factor = 16
        latent_h = resolution // downsample_factor
        latent_w = (resolution * 2) // downsample_factor
        reconstructed = vq_model.decode_code(codes, shape=(latent_h, latent_w))
    
    results["magvit"] = {
        'original': original_np,
        'reconstructed': denormalize_image(reconstructed.cpu()),
        'codes': codes,
        'num_tokens': num_tokens
    }
    
    if vision_tokenizer is not None:
        vt_device = vision_tokenizer_device or device
        vt_tensor = base_tensor.to(vt_device)
        with torch.no_grad():
            quant, _, info = vision_tokenizer.encode(vt_tensor)
            vt_codes = info[-1].view(vt_tensor.shape[0], -1).long()
            latent_h, latent_w = quant.shape[2], quant.shape[3]
            vt_shape = (vt_tensor.shape[0], latent_h, latent_w, quant.shape[1])
            reconstructed_vt = vision_tokenizer.decode_code(vt_codes, shape=vt_shape)
        
        results["vision_tokenizer"] = {
            'original': original_np,
            'reconstructed': denormalize_image(reconstructed_vt.cpu()),
            'codes': vt_codes,
            'num_tokens': vt_codes.shape[1],
            'latent_shape': (latent_h, latent_w),
        }
    
    return results


def visualize_multi_resolution_reconstruction(results_dict, save_path, model_order):
    """
    可视化多个分辨率的重建结果
    Args:
        results_dict: dict, {resolution: result_dict}
        save_path: str, 保存路径
    """
    resolutions = sorted(results_dict.keys())
    num_res = len(resolutions)
    num_rows = 1 + len(model_order)
    
    fig, axes = plt.subplots(num_rows, num_res, figsize=(5 * num_res, 4 * num_rows))
    if num_res == 1:
        axes = np.array(axes).reshape(num_rows, 1)
    
    for col, res in enumerate(resolutions):
        width = res * 2
        first_model = results_dict[res][model_order[0]]
        axes[0, col].imshow(first_model['original'])
        axes[0, col].set_title(f'Original\n{res}x{width}', fontsize=14, fontweight='bold')
        axes[0, col].axis('off')
        
        for row_idx, model_name in enumerate(model_order, start=1):
            result = results_dict[res][model_name]
            axes[row_idx, col].imshow(result['reconstructed'])
            axes[row_idx, col].set_title(
                f'{MODEL_DISPLAY_NAMES.get(model_name, model_name)}\n{res}x{width}\n{result["num_tokens"]} tokens',
                fontsize=12,
                fontweight='bold'
            )
            axes[row_idx, col].axis('off')
    
    plt.suptitle('VQ Reconstruction at Different Resolutions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    
    return fig


def compute_metrics(original, reconstructed):
    """
    计算重建质量指标
    Args:
        original: numpy array [H, W, 3], uint8
        reconstructed: numpy array [H, W, 3], uint8
    Returns:
        dict: 包含各种指标
    """
    # MSE
    mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
    
    # PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # MAE
    mae = np.mean(np.abs(original.astype(float) - reconstructed.astype(float)))
    
    return {
        'MSE': mse,
        'PSNR': psnr,
        'MAE': mae
    }


def main():
    import argparse
    
    # 默认路径配置
    DEFAULT_VQ_MODEL_PATH = "/lpai/volumes/ad-vla-vol-ga/lipengxiang/hf_model/magvit2"
    DEFAULT_JSON_PATH = "/lpai/volumes/ad-vla-vol-ga/lipengxiang/code/github_code_backup/lladaWM/demo.jsonl"
    DEFAULT_NAVSIM_LOG_PATH = "/lpai/dataset/navsim-unzip/0-1-0/navsim_workspace/trainval_navsim_logs/trainval"
    DEFAULT_SENSOR_BLOBS_PATH = "/lpai/dataset/navsim-unzip/0-1-0/navsim_workspace/trainval_sensor_blobs/trainval"
    
    parser = argparse.ArgumentParser(description="Test NavSim VQ-VAE reconstruction at different resolutions")
    parser.add_argument("--vq_model_path", type=str, default=DEFAULT_VQ_MODEL_PATH,
                       help="Path to VQ-VAE model checkpoint")
    parser.add_argument("--json_path", type=str, default=DEFAULT_JSON_PATH,
                       help="Path to NavSim sample JSON")
    parser.add_argument("--navsim_log_path", type=str, default=DEFAULT_NAVSIM_LOG_PATH,
                       help="Path to NavSim log directory")
    parser.add_argument("--sensor_blobs_path", type=str, default=DEFAULT_SENSOR_BLOBS_PATH,
                       help="Path to sensor blobs directory")
    parser.add_argument("--sample_idx", type=int, default=0,
                       help="Index of sample to test")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[128, 256, 384, 512],
                       help="List of resolutions to test")
    parser.add_argument("--output", type=str, default="navsim_vq_reconstruction_test.png",
                       help="Output visualization path")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--vision_tokenizer_path", type=str, default=None,
                       help="Path to vision tokenizer directory (expects config.yaml & model.ckpt)")
    parser.add_argument("--vision_tokenizer_type", type=str, default="ibq",
                       help="Vision tokenizer type (default: ibq)")
    parser.add_argument("--vision_tokenizer_device", type=str, default=None,
                       help="Device for the vision tokenizer (defaults to --device)")
    
    args = parser.parse_args()
    
    device = args.device
    vision_tokenizer_device = args.vision_tokenizer_device or device
    
    vision_tokenizer = None
    if args.vision_tokenizer_path is not None:
        print(f"\nLoading vision tokenizer from {args.vision_tokenizer_path} ({args.vision_tokenizer_type})...")
        vision_tokenizer = build_vision_tokenizer(
            type=args.vision_tokenizer_type,
            model_path=args.vision_tokenizer_path,
            device=vision_tokenizer_device,
        )
        vision_tokenizer.eval()
        vision_tokenizer.requires_grad_(False)
        print("Vision tokenizer loaded successfully!")
    
    # 1. 加载 VQ-VAE 模型
    vq_model = load_vq_model(args.vq_model_path, device=device)
    
    # 2. 加载 NavSim 数据
    print("\nLoading NavSim dataset...")
    dataloader = create_navsim_mmada_dataloader(
        json_path=args.json_path,
        navsim_log_path=args.navsim_log_path,
        sensor_blobs_path=args.sensor_blobs_path,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
        persistent_workers=False,
        drop_last=False,
    )
    
    # 3. 获取指定样本
    print(f"Loading sample {args.sample_idx}...")
    for idx, sample in enumerate(dataloader):
        if idx == args.sample_idx:
            target_sample = sample
            break
    else:
        raise ValueError(f"Sample index {args.sample_idx} not found in dataset")
    
    # 获取第一帧历史图像（已经过 normalize，范围 [-1, 1]）
    history_images = target_sample["history_front_images"]  # [B, num_hist, C, H, W]
    first_image_tensor = history_images[0, 0]  # [C, H, W], 范围 [-1, 1]
    
    # 转换为 PIL Image 用于 resize
    # 先反归一化到 [0, 1]，再转为 PIL
    first_image_denorm = torch.clamp((first_image_tensor + 1.0) / 2.0, 0.0, 1.0)
    first_image_pil = transforms.ToPILImage()(first_image_denorm)
    
    print(f"Original image size: {first_image_pil.size}")
    print(f"Testing resolutions: {args.resolutions}")
    
    # 4. 测试不同分辨率
    results_dict = {}
    model_order = ["magvit"]
    if vision_tokenizer is not None:
        model_order.append("vision_tokenizer")
    
    print("\n" + "="*80)
    print("Testing reconstructions at different resolutions (1:2 aspect ratio):")
    print("="*80)
    
    for resolution in args.resolutions:
        width = resolution * 2
        print(f"\nResolution: {resolution}x{width}")
        per_model_results = reconstruct_at_resolution(
            vq_model,
            first_image_pil,
            resolution,
            device,
            vision_tokenizer=vision_tokenizer,
            vision_tokenizer_device=vision_tokenizer_device,
        )
        results_dict[resolution] = per_model_results
        
        # 计算指标
        for model_name in model_order:
            if model_name not in per_model_results:
                continue
            display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            metrics = compute_metrics(
                per_model_results[model_name]['original'],
                per_model_results[model_name]['reconstructed']
            )
            print(f"  {display_name}:")
            print(f"    Tokens: {per_model_results[model_name]['num_tokens']}")
            print(f"    MSE:  {metrics['MSE']:.2f}")
            print(f"    PSNR: {metrics['PSNR']:.2f} dB")
            print(f"    MAE:  {metrics['MAE']:.2f}")
    
    # 5. 可视化
    print("\n" + "="*80)
    print("Creating visualization...")
    visualize_multi_resolution_reconstruction(results_dict, args.output, model_order)
    
    # 6. 打印总结
    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    print(f"{'Resolution':<15} {'Model':<18} {'Tokens':<10} {'MSE':<12} {'PSNR (dB)':<12} {'MAE':<10}")
    print("-" * 80)
    for resolution in args.resolutions:
        width = resolution * 2
        for model_name in model_order:
            if model_name not in results_dict[resolution]:
                continue
            result = results_dict[resolution][model_name]
            metrics = compute_metrics(result['original'], result['reconstructed'])
            display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            print(f"{resolution}x{width:<8} {display_name:<18} {result['num_tokens']:<10} "
                  f"{metrics['MSE']:<12.2f} {metrics['PSNR']:<12.2f} {metrics['MAE']:<10.2f}")
    print("="*80)


if __name__ == "__main__":
    main()
