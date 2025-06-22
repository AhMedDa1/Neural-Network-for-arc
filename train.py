import torch
from encoder import Encoder_ViT
from decoder import DirectDecoder_UNet
from data_utils import pad_grid
from visualization_utils import save_training_visualization, save_inference_visualization
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from typing import Dict, Any, Tuple, Optional, List

VISUALIZATION_INTERVAL = 500
  
def train_model(puzzle: Dict[str, Any], device: torch.device, hyperparams: Dict[str, Any], puzzle_id: str, solutions_file_train: str = None, vis_interval: int = VISUALIZATION_INTERVAL):
    train_pairs = puzzle['train']
    grid_shape = puzzle['max_grid_shape_train']
    encoder = Encoder_ViT(grid_shape, hyperparams['patch_size'], hyperparams['num_colors'], hyperparams['embed_dim'], hyperparams['encoder_depth'], hyperparams['encoder_heads']).to(device)
    decoder = DirectDecoder_UNet(grid_shape, hyperparams['num_colors'], hyperparams['decoder_init_channels'], hyperparams['decoder_channel_mults'], hyperparams['embed_dim'], hyperparams['cross_attention_heads']).to(device)
    all_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(all_params, lr=hyperparams['lr'], weight_decay=hyperparams['weight_decay'])
    loss_fn = nn.CrossEntropyLoss()
    encoder.train()
    decoder.train()
    for epoch in range(hyperparams['epochs']):
        total_loss = 0.0
        for k, pair in enumerate(train_pairs):
            x0 = torch.tensor(pad_grid(pair['input'], grid_shape), dtype=torch.long, device=device)
            x1 = torch.tensor(pad_grid(pair['output'], grid_shape), dtype=torch.long, device=device)

            context_pairs = [train_pairs[j] for j in range(len(train_pairs)) if j != k]
            if not context_pairs:
                continue
            context_embeddings = []
            for context_pair in context_pairs:
                cx0 = torch.tensor(pad_grid(context_pair['input'], grid_shape), dtype=torch.long, device=device)
                cx1 = torch.tensor(pad_grid(context_pair['output'], grid_shape), dtype=torch.long, device=device)
                emb = encoder(cx0.unsqueeze(0), cx1.unsqueeze(0))
                context_embeddings.append(emb)
            context_embeddings = torch.cat(context_embeddings, dim=0).unsqueeze(0)  # (1, K-1, D)
            logits = decoder(x0.unsqueeze(0), context_embeddings)
            loss = loss_fn(logits, x1.unsqueeze(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            step = epoch * len(train_pairs) + k
            if (step % vis_interval == 0) or (step == 0) or (epoch == hyperparams['epochs']-1 and k == len(train_pairs)-1):
                pred = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
                input_grid = x0.detach().cpu().numpy()
                gt_grid = x1.detach().cpu().numpy()
                save_training_visualization(input_grid, pred, gt_grid, puzzle_id, step, vis_idx=k)
        print(f"Epoch {epoch+1}/{hyperparams['epochs']}, Loss: {total_loss/len(train_pairs):.4f}")
    return encoder, decoder

def evaluate_model(encoder, decoder, puzzle: Dict[str, Any], device: torch.device, puzzle_id: str, solutions_file_train: str = None):
    test_pairs = puzzle['test']
    train_grid_shape = puzzle['max_grid_shape_train']
    encoder.eval()
    decoder.eval()
    for i, pair in enumerate(test_pairs):
        x0 = torch.tensor(pad_grid(pair['input'], train_grid_shape), dtype=torch.long, device=device)

        context_embeddings = []
        for train_pair in puzzle['train']:
            cx0 = torch.tensor(pad_grid(train_pair['input'], train_grid_shape), dtype=torch.long, device=device)
            cx1 = torch.tensor(pad_grid(train_pair['output'], train_grid_shape), dtype=torch.long, device=device)
            emb = encoder(cx0.unsqueeze(0), cx1.unsqueeze(0))
            context_embeddings.append(emb)
        context_embeddings = torch.cat(context_embeddings, dim=0).unsqueeze(0)
        with torch.no_grad():
            logits = decoder(x0.unsqueeze(0), context_embeddings)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        gt_grid = pair.get('output', None)
        input_grid = pair['input']
        save_inference_visualization(input_grid, pred, gt_grid, puzzle_id, i)
        print(f"Test Example {i+1}: Prediction shape: {pred.shape} (visualization saved)")

def infer_arc_task_icl(
    puzzle_id: str,
    puzzle: Dict[str, Any],
    encoder: nn.Module,
    decoder: nn.Module,
    device: torch.device,
    solutions_file_train: Optional[str] = None,
    output_vis_dir: str = "visualizations"
) -> Tuple[List[Dict[str, Any]], float]:
    import time
    test_pairs = puzzle['test']
    test_grid_shape = puzzle['max_grid_shape_test']
    train_grid_shape = puzzle['max_grid_shape_train']
    
    unified_grid_shape = train_grid_shape
    
    encoder.eval()
    decoder.eval()

    gt_solutions = []
    num_train = len(puzzle.get('train', []))
    num_test = len(test_pairs)
    if solutions_file_train and os.path.exists(solutions_file_train):
        with open(solutions_file_train, 'r') as f:
            all_solutions_data = json.load(f)
        if puzzle_id in all_solutions_data:
            puzzle_solutions_raw = all_solutions_data[puzzle_id]
            parsed_test_sols_raw = []
            if isinstance(puzzle_solutions_raw, list):
                if len(puzzle_solutions_raw) >= num_train + num_test:
                    parsed_test_sols_raw = puzzle_solutions_raw[num_train : num_train + num_test]
                elif len(puzzle_solutions_raw) == num_test:
                    parsed_test_sols_raw = puzzle_solutions_raw
                else:
                    parsed_test_sols_raw = []
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

    train_pairs = puzzle['train']
    context_embeddings = []
    for train_pair in train_pairs:
        cx0 = torch.tensor(pad_grid(train_pair['input'], unified_grid_shape), dtype=torch.long, device=device)
        cx1 = torch.tensor(pad_grid(train_pair['output'], unified_grid_shape), dtype=torch.long, device=device)
        emb = encoder(cx0.unsqueeze(0), cx1.unsqueeze(0))
        context_embeddings.append(emb)
    context_embeddings = torch.cat(context_embeddings, dim=0).unsqueeze(0)

    results = []
    t0 = time.time()
    for i, pair in enumerate(test_pairs):
        x0 = torch.tensor(pad_grid(pair['input'], unified_grid_shape), dtype=torch.long, device=device)
        with torch.no_grad():
            logits = decoder(x0.unsqueeze(0), context_embeddings)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

        gt_grid = gt_solutions[i]
        if gt_grid is not None:
            h, w = gt_grid.shape
            pred_cropped = pred[:h, :w]
        else:
            pred_cropped = pred
        input_grid = pair['input']
        save_inference_visualization(input_grid, pred_cropped, gt_grid, puzzle_id, i)
        results.append({
            'input': input_grid,
            'prediction': pred_cropped,
            'gt': gt_grid
        })
    t1 = time.time()
    return results, t1-t0 