import torch
from data_utils import load_puzzle
from train import train_model, infer_arc_task_icl
from visualization_utils import save_initial_overview, calculate_meaningful_accuracy
import json
import time
from thop import profile
import os
import numpy as np

CHALLENGES_FILE_TRAIN =  'arc-agi_training_challenges.json'
SOLUTIONS_FILE_TRAIN = 'arc-agi_training_solutions.json'    
PUZZLE_ID = ['87ab05b8']

HYPERPARAMS = {
    'num_colors': 10,
    'embed_dim': 128,
    'patch_size': 3,
    'encoder_depth': 4,
    'encoder_heads': 4,
    'decoder_init_channels': 128,
    'decoder_channel_mults': (1, 2, 4),
    'cross_attention_heads': 4,
    'lr': 3e-4,
    'weight_decay': 1e-4,
    'epochs': 1500
}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def human_format(num):
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)

def save_results_incremental(result_entry, model_info, filename="results.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                all_data = json.load(f)
                if not isinstance(all_data, dict):
                    all_data = {"model_info": model_info, "tasks": []}
            except Exception:
                all_data = {"model_info": model_info, "tasks": []}
    else:
        all_data = {"model_info": model_info, "tasks": []}
    if "model_info" not in all_data:
        all_data["model_info"] = model_info
    if "tasks" not in all_data:
        all_data["tasks"] = []
    all_data["tasks"].append(result_entry)
    with open(filename, "w") as f:
        json.dump(all_data, f, indent=2)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_stats_added = False
    model_info = HYPERPARAMS.copy()
    for idx, puzzle_id in enumerate(PUZZLE_ID):
        start_time = time.time()
        puzzle = load_puzzle(puzzle_id, CHALLENGES_FILE_TRAIN, SOLUTIONS_FILE_TRAIN)
        save_initial_overview(puzzle, puzzle_id, SOLUTIONS_FILE_TRAIN)
        encoder, decoder = train_model(puzzle, device, HYPERPARAMS, puzzle_id, SOLUTIONS_FILE_TRAIN)

        if not model_stats_added:
            encoder_num_params = count_parameters(encoder)
            decoder_num_params = count_parameters(decoder)
            total_num_params = encoder_num_params + decoder_num_params
            encoder_num_params_fmt = human_format(encoder_num_params)
            decoder_num_params_fmt = human_format(decoder_num_params)
            total_num_params_fmt = human_format(total_num_params)
            grid_shape = puzzle['max_grid_shape_train']
            dummy_x0 = torch.randint(0, HYPERPARAMS['num_colors'], (1, grid_shape[0], grid_shape[1]), device=device)
            dummy_x1 = torch.randint(0, HYPERPARAMS['num_colors'], (1, grid_shape[0], grid_shape[1]), device=device)
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                encoder_flops, _ = profile(encoder, inputs=(dummy_x0, dummy_x1), verbose=False)
                context_emb = encoder(dummy_x0, dummy_x1).unsqueeze(0)  # (1, 1, D)
                decoder_flops, _ = profile(decoder, inputs=(dummy_x0, context_emb), verbose=False)
            total_flops = encoder_flops + decoder_flops
            encoder_flops_fmt = human_format(encoder_flops)
            decoder_flops_fmt = human_format(decoder_flops)
            total_flops_fmt = human_format(total_flops)
            model_info["encoder_num_parameters_fmt"] = encoder_num_params_fmt
            model_info["decoder_num_parameters_fmt"] = decoder_num_params_fmt
            model_info["total_num_parameters_fmt"] = total_num_params_fmt
            model_info["encoder_flops_fmt"] = encoder_flops_fmt
            model_info["decoder_flops_fmt"] = decoder_flops_fmt
            model_info["total_flops_fmt"] = total_flops_fmt
            model_stats_added = True

        results, infer_time = infer_arc_task_icl(puzzle_id, puzzle, encoder, decoder, device, SOLUTIONS_FILE_TRAIN)
        train_time = time.time() - start_time - infer_time

        per_test_stats = []
        for idx2, res in enumerate(results):
            gt = res['gt']
            pred = res['prediction']
            accuracy = None
            perfect = False
            correct_pixels = None
            total_pixels = None
            
            if gt is not None:
                try:
                    gt_arr = (gt if hasattr(gt, 'shape') else None)
                    pred_arr = (pred if hasattr(pred, 'shape') else None)
                    if gt_arr is not None and pred_arr is not None:
                        if pred_arr.shape != gt_arr.shape:
                            correct_pixels = None
                            total_pixels = None
                            accuracy = None
                            perfect = False
                        else:
                            non_zero_mask = (gt_arr != 0)
                            if np.any(non_zero_mask):
                                correct_pixels = int(np.sum((pred_arr == gt_arr) & non_zero_mask))
                                total_pixels = int(np.sum(non_zero_mask))
                                accuracy = correct_pixels / total_pixels
                                perfect = bool(correct_pixels == total_pixels)
                            else:
                                correct_pixels = 0
                                total_pixels = 0
                                accuracy = None
                                perfect = False
                except Exception:
                    pass
            
            per_test_stats.append({
                "test_case_index": idx2,
                "accuracy": float(accuracy) if accuracy is not None else None,
                "perfect": perfect,
                "correct_pixels": correct_pixels,
                "total_pixels": total_pixels,
                "accuracy_percent": float(accuracy * 100) if accuracy is not None else None,
                "prediction_shape": list(pred.shape) if hasattr(pred, 'shape') else None,
                "gt_shape": list(gt.shape) if hasattr(gt, 'shape') and gt is not None else None,
                "prediction_output": pred.tolist() if hasattr(pred, 'tolist') else pred
            })

        color_set = set()
        for pair in puzzle['train']:
            color_set.update(set(pair['input'].flatten()))
            color_set.update(set(pair['output'].flatten()))
        for pair in puzzle['test']:
            color_set.update(set(pair['input'].flatten()))
            if pair.get('output') is not None:
                color_set.update(set(pair['output'].flatten()))
        num_colors_in_task = len(color_set)

        result_entry = {
            "task_id": puzzle_id,
            "train_time_s": train_time,
            "infer_time_s": infer_time,
            "num_train": len(puzzle['train']),
            "num_test": len(puzzle['test']),
            "num_colors_in_task": num_colors_in_task,
            "per_test_stats": per_test_stats,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        save_results_incremental(result_entry, model_info, filename="results.json")
        print(f"Puzzle {puzzle_id}: Inference completed in {infer_time:.2f} seconds. {len(results)} test cases processed. Results saved.")

if __name__ == '__main__':
    main() 