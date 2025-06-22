import os
import numpy as np
import matplotlib.pyplot as plt
from data_utils import ARC_CMAP, ARC_NORM, plot_puzzle

OUTPUT_VIS_DIR = "visualizations"


def save_initial_overview(puzzle_data, puzzle_id, solutions_file_train=None):
    plot_puzzle(puzzle_data, puzzle_id, solutions_file_train, output_vis_dir=OUTPUT_VIS_DIR)


def calculate_meaningful_accuracy(pred_crop, gt_crop, background_color=0):
    """Calculate accuracy metrics that are not dominated by background"""
    total_pixels = pred_crop.size
    correct_pixels = np.sum(pred_crop == gt_crop)
    overall_accuracy = correct_pixels / total_pixels
    
    bg_mask = (gt_crop == background_color)
    non_bg_mask = ~bg_mask
    
    bg_correct = np.sum((pred_crop == gt_crop) & bg_mask) if np.any(bg_mask) else 0
    bg_total = np.sum(bg_mask)
    bg_accuracy = bg_correct / bg_total if bg_total > 0 else 1.0
    
    pattern_correct = np.sum((pred_crop == gt_crop) & non_bg_mask) if np.any(non_bg_mask) else 0
    pattern_total = np.sum(non_bg_mask)
    pattern_accuracy = pattern_correct / pattern_total if pattern_total > 0 else 1.0
    
    return {
        'overall_correct': int(correct_pixels),
        'overall_total': int(total_pixels),
        'overall_accuracy': float(overall_accuracy),
        'bg_correct': int(bg_correct),
        'bg_total': int(bg_total),
        'bg_accuracy': float(bg_accuracy),
        'pattern_correct': int(pattern_correct),
        'pattern_total': int(pattern_total),
        'pattern_accuracy': float(pattern_accuracy)
    }


def save_training_visualization(input_grid, pred_grid, gt_grid, puzzle_id, step, vis_idx=0):
    puzzle_dir = os.path.join(OUTPUT_VIS_DIR, puzzle_id)
    os.makedirs(puzzle_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(input_grid, cmap=ARC_CMAP, norm=ARC_NORM)
    axes[0].set_title(f"Input\nShape: {input_grid.shape}")
    
    axes[1].imshow(pred_grid, cmap=ARC_CMAP, norm=ARC_NORM)
    
    if gt_grid is not None:
        min_h = min(pred_grid.shape[0], gt_grid.shape[0])
        min_w = min(pred_grid.shape[1], gt_grid.shape[1])
        
        if pred_grid.shape != gt_grid.shape:
            axes[1].set_title(f"Prediction\nShape: {pred_grid.shape}\nSize mismatch with GT: {gt_grid.shape}")
        else:
            pred_crop = pred_grid[:min_h, :min_w]
            gt_crop = gt_grid[:min_h, :min_w]
            
            non_zero_mask = (gt_crop != 0)
            if np.any(non_zero_mask):
                correct_pixels = np.sum((pred_crop == gt_crop) & non_zero_mask)
                total_pixels = np.sum(non_zero_mask)
                accuracy_percent = (correct_pixels / total_pixels) * 100
                axes[1].set_title(f"Prediction\nShape: {pred_grid.shape}\nCorrect: {correct_pixels}/{total_pixels} ({accuracy_percent:.1f}%)")
            else:
                axes[1].set_title(f"Prediction\nShape: {pred_grid.shape}\nNo non-zero pixels in GT")
        
        axes[2].imshow(gt_grid, cmap=ARC_CMAP, norm=ARC_NORM)
        axes[2].set_title(f"Ground Truth\nShape: {gt_grid.shape}")
    else:
        axes[1].set_title(f"Prediction\nShape: {pred_grid.shape}")
        h, w = input_grid.shape
        placeholder = np.full((h, w), 5, dtype=np.int64)
        axes[2].imshow(placeholder, cmap=ARC_CMAP, norm=ARC_NORM, alpha=0.3)
        axes[2].set_title("Ground Truth (Missing)")
    
    for ax in axes:
        ax.axis('off')
    save_path = os.path.join(puzzle_dir, f"train_vis_pair_{vis_idx}_step_{step}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close(fig)
    return save_path


def save_inference_visualization(input_grid, pred_grid, gt_grid, puzzle_id, idx):
    inf_dir = os.path.join(OUTPUT_VIS_DIR, puzzle_id, "inference_icl")
    os.makedirs(inf_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(input_grid, cmap=ARC_CMAP, norm=ARC_NORM)
    axes[0].set_title(f"Input\nShape: {input_grid.shape}")
    
    axes[1].imshow(pred_grid, cmap=ARC_CMAP, norm=ARC_NORM)
    
    if gt_grid is not None:
        min_h = min(pred_grid.shape[0], gt_grid.shape[0])
        min_w = min(pred_grid.shape[1], gt_grid.shape[1])
        
        if pred_grid.shape != gt_grid.shape:
            axes[1].set_title(f"Prediction\nShape: {pred_grid.shape}\nSize mismatch with GT: {gt_grid.shape}")
        else:
            pred_crop = pred_grid[:min_h, :min_w]
            gt_crop = gt_grid[:min_h, :min_w]
            
            non_zero_mask = (gt_crop != 0)
            if np.any(non_zero_mask):
                correct_pixels = np.sum((pred_crop == gt_crop) & non_zero_mask)
                total_pixels = np.sum(non_zero_mask)
                accuracy_percent = (correct_pixels / total_pixels) * 100
                axes[1].set_title(f"Prediction\nShape: {pred_grid.shape}\nCorrect: {correct_pixels}/{total_pixels} ({accuracy_percent:.1f}%)")
            else:
                axes[1].set_title(f"Prediction\nShape: {pred_grid.shape}\nNo non-zero pixels in GT")
        
        axes[2].imshow(gt_grid, cmap=ARC_CMAP, norm=ARC_NORM)
        axes[2].set_title(f"Ground Truth\nShape: {gt_grid.shape}")
    else:
        axes[1].set_title(f"Prediction\nShape: {pred_grid.shape}")
        h, w = input_grid.shape
        placeholder = np.full((h, w), 5, dtype=np.int64)
        axes[2].imshow(placeholder, cmap=ARC_CMAP, norm=ARC_NORM, alpha=0.3)
        axes[2].set_title("Ground Truth (Missing)")
    
    for ax in axes:
        ax.axis('off')
    save_path = os.path.join(inf_dir, f"final_inference_test_{idx}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close(fig)
    return save_path 