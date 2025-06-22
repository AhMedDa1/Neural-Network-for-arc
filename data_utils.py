import numpy as np
import json
import os
from typing import Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib import colors

NUM_COLORS = 10
ARC_COLOR_MAP_LIST = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"
]
ARC_CMAP = colors.ListedColormap(ARC_COLOR_MAP_LIST)
ARC_NORM = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)


def pad_grid(grid: np.ndarray, target_shape: Tuple[int, int], pad_value: int = 0) -> np.ndarray:
    h, w = grid.shape
    th, tw = target_shape
    if h == th and w == tw:
        return grid
    if h > th or w > tw:
        crop_h = min(h, th)
        crop_w = min(w, tw)
        crop_h = max(0, crop_h)
        crop_w = max(0, crop_w)
        return grid[:crop_h, :crop_w]
    if h == 0 or w == 0:
        return np.full(target_shape, pad_value, dtype=grid.dtype)
    padded = np.full(target_shape, pad_value, dtype=grid.dtype)
    padded[:h, :w] = grid
    return padded

def load_puzzle(puzzle_id: str, challenges_file: str, solutions_file: Optional[str]) -> Optional[Dict[str, Any]]:
    with open(challenges_file, 'r') as f:
        challenges = json.load(f)
    puzzle_data = challenges[puzzle_id]
    solution_list_for_puzzle = []
    if solutions_file and os.path.exists(solutions_file):
        with open(solutions_file, 'r') as f:
            solutions = json.load(f)
        if solutions and puzzle_id in solutions:
            raw_solutions = solutions[puzzle_id]
            parsed_test_solutions = []
            if isinstance(raw_solutions, list):
                num_train_expected = len(puzzle_data.get('train', []))
                num_test_expected = len(puzzle_data.get('test', []))
                if len(raw_solutions) == num_train_expected + num_test_expected:
                    parsed_test_solutions = raw_solutions[num_train_expected:]
                elif len(raw_solutions) == num_test_expected:
                    parsed_test_solutions = raw_solutions
                temp_parsed = []
                for item in parsed_test_solutions:
                    if isinstance(item, list) and len(item) == 1 and isinstance(item[0], list):
                        temp_parsed.append(item[0])
                    elif isinstance(item, list) and all(isinstance(row, list) for row in item):
                        temp_parsed.append(item)
                solution_list_for_puzzle = temp_parsed
            elif isinstance(raw_solutions, dict):
                temp_parsed = []
                i = 1
                while f'solution_{i}' in raw_solutions:
                    temp_parsed.append(raw_solutions[f'solution_{i}'])
                    i += 1
                if not temp_parsed:
                    i = 1
                    while f'attempt_{i}' in raw_solutions:
                        temp_parsed.append(raw_solutions[f'attempt_{i}'])
                        i += 1
                num_train_expected = len(puzzle_data.get('train', []))
                num_test_expected = len(puzzle_data.get('test', []))
                if len(temp_parsed) == num_train_expected + num_test_expected:
                    solution_list_for_puzzle = temp_parsed[num_train_expected:]
                elif len(temp_parsed) == num_test_expected:
                    solution_list_for_puzzle = temp_parsed
                else:
                    solution_list_for_puzzle = []
    processed_puzzle = {'train': [], 'test': []}
    train_examples_in_challenge = puzzle_data.get('train', [])
    test_examples_in_challenge = puzzle_data.get('test', [])
    max_grid_shape_train = [0, 0]
    for i, example in enumerate(train_examples_in_challenge):
        if 'input' in example:
            in_arr = np.array(example['input'], dtype=np.int64)
            if in_arr.ndim == 2:
                h, w = in_arr.shape
                if h > 0 and w > 0:
                    max_grid_shape_train[0] = max(max_grid_shape_train[0], h)
                    max_grid_shape_train[1] = max(max_grid_shape_train[1], w)
        if 'output' in example:
            out_arr = np.array(example['output'], dtype=np.int64)
            if out_arr.ndim == 2:
                h_out, w_out = out_arr.shape
                if h_out > 0 and w_out > 0:
                    max_grid_shape_train[0] = max(max_grid_shape_train[0], h_out)
                    max_grid_shape_train[1] = max(max_grid_shape_train[1], w_out)
    max_grid_shape_train[0] = max(1, max_grid_shape_train[0])
    max_grid_shape_train[1] = max(1, max_grid_shape_train[1])
    processed_puzzle['max_grid_shape_train'] = tuple(max_grid_shape_train)
    for i, example in enumerate(train_examples_in_challenge):
        train_input = np.array(example['input'], dtype=np.int64)
        if train_input.ndim != 2: raise ValueError("Input not 2D")
        if train_input.shape[0] == 0 or train_input.shape[1] == 0: raise ValueError("Input is empty")
        train_output = None
        potential_output = np.array(example['output'], dtype=np.int64)
        if potential_output.ndim == 2:
            if potential_output.shape[0] == 0 or potential_output.shape[1] == 0: raise ValueError("Output is empty")
            train_output = potential_output
        if train_input is not None and train_output is not None:
            processed_puzzle['train'].append({'input': train_input, 'output': train_output})
    max_grid_shape_test = [0, 0]
    for i, example in enumerate(test_examples_in_challenge):
        if not isinstance(example, dict) or 'input' not in example:
            continue
        test_in_arr = np.array(example['input'], dtype=np.int64)
        if test_in_arr.ndim != 2:
            continue
        h, w = test_in_arr.shape
        if h == 0 or w == 0:
            continue
        max_grid_shape_test[0] = max(max_grid_shape_test[0], h)
        max_grid_shape_test[1] = max(max_grid_shape_test[1], w)
        processed_puzzle['test'].append({'input': test_in_arr, 'output': None})
    max_grid_shape_test[0] = max(1, max_grid_shape_test[0])
    max_grid_shape_test[1] = max(1, max_grid_shape_test[1])
    processed_puzzle['max_grid_shape_test'] = tuple(max_grid_shape_test)
    processed_puzzle['test_solutions_gt'] = [sol if isinstance(sol, (np.ndarray, list)) else None for sol in solution_list_for_puzzle]
    return processed_puzzle

def plot_puzzle(puzzle_data: Dict[str, Any], puzzle_id: str, solutions_file_train: Optional[str] = None, output_vis_dir: str = "./visualizations_icl_cls_weighted_782"):
    train_pairs = puzzle_data.get('train', [])
    test_pairs = puzzle_data.get('test', [])
    num_train = len(train_pairs)
    num_test = len(test_pairs)
    gt_solutions = []
    solutions_exist = solutions_file_train and os.path.exists(solutions_file_train)
    if solutions_exist:
        with open(solutions_file_train, 'r') as f:
            all_solutions_data = json.load(f)
        if puzzle_id in all_solutions_data:
            puzzle_solutions_raw = all_solutions_data[puzzle_id]
            parsed_test_sols_raw = []
            if isinstance(puzzle_solutions_raw, list):
                if len(puzzle_solutions_raw) >= num_train + num_test:
                    parsed_test_sols_raw = puzzle_solutions_raw[num_train : num_train + num_test]
                elif len(puzzle_solutions_raw) == num_test:
                    print(f"  [Plot Warning] Solutions list length {len(puzzle_solutions_raw)} matches num_test {num_test}, assuming direct test solutions.")
                    parsed_test_sols_raw = puzzle_solutions_raw
                else:
                    print(f"  [Plot Warning] Solutions list length mismatch (List: {len(puzzle_solutions_raw)}, Train: {num_train}, Test: {num_test}). Cannot reliably load test GTs.")
            elif isinstance(puzzle_solutions_raw, dict):
                sol_keys = sorted([k for k in puzzle_solutions_raw if k.startswith('solution_')])
                att_keys = sorted([k for k in puzzle_solutions_raw if k.startswith('attempt_')])
                potential_keys = []
                if len(sol_keys) >= num_train + num_test: potential_keys = sol_keys[num_train : num_train + num_test]
                elif len(att_keys) >= num_train + num_test: potential_keys = att_keys[num_train : num_train + num_test]
                elif len(sol_keys) == num_test: potential_keys = sol_keys
                elif len(att_keys) == num_test: potential_keys = att_keys
                if len(potential_keys) == num_test:
                    for key in potential_keys: parsed_test_sols_raw.append(puzzle_solutions_raw[key])
            for raw_sol in parsed_test_sols_raw:
                sol_np = None
                actual_raw_sol = None
                if isinstance(raw_sol, list) and len(raw_sol) == 1 and isinstance(raw_sol[0], list):
                    actual_raw_sol = raw_sol[0]
                elif isinstance(raw_sol, list) and all(isinstance(row, list) for row in raw_sol):
                    actual_raw_sol = raw_sol
                if actual_raw_sol is not None:
                    try:
                        temp_np = np.array(actual_raw_sol, dtype=np.int64)
                        if temp_np.ndim == 2: sol_np = temp_np
                    except: pass
                gt_solutions.append(sol_np)
    while len(gt_solutions) < num_test: gt_solutions.append(None)
    gt_solutions = gt_solutions[:num_test]
    num_rows = num_train + num_test
    fig, axes = plt.subplots(num_rows, 2, figsize=(6, max(3, 2.5 * num_rows)), squeeze=False)
    fig.suptitle(f"Puzzle: {puzzle_id}", fontsize=14, y=0.99)
    for i in range(num_train):
        if i >= len(train_pairs) or not train_pairs[i] or 'input' not in train_pairs[i] or 'output' not in train_pairs[i]: continue
        input_grid = train_pairs[i]['input']
        output_grid = train_pairs[i]['output']
        if input_grid is None or output_grid is None: continue
        ax_in, ax_out = axes[i, 0], axes[i, 1]
        ax_in.imshow(input_grid, cmap=ARC_CMAP, norm=ARC_NORM, interpolation='nearest')
        ax_in.set_title(f'Train {i}: Input ({input_grid.shape[0]}x{input_grid.shape[1]})')
        ax_in.axis('off')
        ax_out.imshow(output_grid, cmap=ARC_CMAP, norm=ARC_NORM, interpolation='nearest')
        ax_out.set_title(f'Train {i}: Output ({output_grid.shape[0]}x{output_grid.shape[1]})')
        ax_out.axis('off')
    for i in range(num_test):
        row_idx = num_train + i
        if i >= len(test_pairs) or not test_pairs[i] or 'input' not in test_pairs[i]: continue
        input_grid = test_pairs[i]['input']
        if input_grid is None: continue
        ax_in, ax_out = axes[row_idx, 0], axes[row_idx, 1]
        ax_in.imshow(input_grid, cmap=ARC_CMAP, norm=ARC_NORM, interpolation='nearest')
        ax_in.set_title(f'Test {i}: Input ({input_grid.shape[0]}x{input_grid.shape[1]})')
        ax_in.axis('off')
        gt_grid = gt_solutions[i]
        if gt_grid is not None:
            ax_out.imshow(gt_grid, cmap=ARC_CMAP, norm=ARC_NORM, interpolation='nearest')
            ax_out.set_title(f'Test {i}: Output (GT) ({gt_grid.shape[0]}x{gt_grid.shape[1]})')
        else:
            h, w = input_grid.shape
            placeholder_shape = (max(1,h), max(1,w))
            placeholder = np.full(placeholder_shape, 5, dtype=np.int64)
            ax_out.imshow(placeholder, cmap=ARC_CMAP, norm=ARC_NORM, interpolation='nearest', alpha=0.3)
            ax_out.set_title(f'Test {i}: Output (GT ?)')
        ax_out.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    init_plot_path = os.path.join(output_vis_dir, puzzle_id, f"{puzzle_id}_initial_overview.png")
    os.makedirs(os.path.dirname(init_plot_path), exist_ok=True)
    fig.savefig(init_plot_path, bbox_inches='tight', dpi=100)
    plt.close(fig) 