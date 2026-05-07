#!/usr/bin/env python3

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import torch
import numpy as np
import pickle
from pathlib import Path
import logging
import click
import re
from tqdm import tqdm
from collections import defaultdict

from BACInfer.analysis.activation_analysis import load_activations
from BACInfer.analysis.collect_activations import get_activations_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_block_l1_errors(original_activations, cached_activations=None):
    block_pattern = re.compile(r'.*decoder\.layers\.(\d+)\.dropout(\d+).*')
    block_activations = {}
    for module_name in original_activations.keys():
        match = block_pattern.match(module_name)
        if match:
            layer_num = int(match.group(1))
            dropout_type = int(match.group(2))
            
            block_types = ['sa_block', 'mha_block', 'ff_block']
            if 1 <= dropout_type <= 3:
                block_key = f"decoder.layers.{layer_num}_{block_types[dropout_type-1]}"
                block_activations[block_key] = original_activations[module_name]
    
    block_errors = {}
    for block_key, activations in block_activations.items():
        num_timesteps = len(activations)
        if num_timesteps <= 1:
            logger.warning(f"Block {block_key} has only {num_timesteps} timesteps, skipping")
            continue
        
        total_error = 0.0
        total_comparisons = 0
        
        if cached_activations is None:
            for i in range(num_timesteps):
                for j in range(i+1, num_timesteps):
                    act_i = activations[i]
                    act_j = activations[j]
                    
                    if act_i.shape != act_j.shape:
                        continue
                    error = torch.mean(torch.abs(act_i - act_j)).item()
                    total_error += error
                    total_comparisons += 1
        else:
            if block_key in cached_activations:
                cached_acts = cached_activations[block_key]
                min_timesteps = min(num_timesteps, len(cached_acts))
                
                for t in range(min_timesteps):
                    orig_act = activations[t]
                    cache_act = cached_acts[t]
                    
                    if orig_act.shape != cache_act.shape:
                        continue
                    error = torch.mean(torch.abs(orig_act - cache_act)).item()
                    total_error += error
                    total_comparisons += 1
        
        if total_comparisons > 0:
            block_errors[block_key] = total_error / total_comparisons
        else:
            logger.warning(f"Block {block_key} has no valid comparison pairs, skipping")
    
    return block_errors

def select_top_error_blocks(block_errors, num_blocks=5):
    sorted_errors = sorted(block_errors.items(), key=lambda x: x[1], reverse=True)
    return sorted_errors[:num_blocks]

def save_block_errors(block_errors, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(block_errors, f)
    logger.info(f"Block error data saved to: {output_path}")

def save_selected_blocks(selected_blocks, output_path):
    selected_dict = {block: error for block, error in selected_blocks}
    with open(output_path, 'wb') as f:
        pickle.dump(selected_dict, f)
    logger.info(f"Selected blocks saved to: {output_path}")

def analyze_block_errors(activations_path, output_dir, num_blocks=5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading activations: {activations_path}")
    activations = load_activations(activations_path)
    
    logger.info("Computing block errors...")
    block_errors = compute_block_l1_errors(activations)
    
    errors_path = output_dir / 'block_l1_errors.pkl'
    save_block_errors(block_errors, errors_path)
    
    logger.info(f"Selecting top {num_blocks} error blocks...")
    selected_blocks = select_top_error_blocks(block_errors, num_blocks)
    
    selected_path = output_dir / f'top_{num_blocks}_error_blocks.pkl'
    save_selected_blocks(selected_blocks, selected_path)
    logger.info(f"Top {num_blocks} error blocks:")
    for block, error in selected_blocks:
        logger.info(f"  {block}: {error:.6f}")
    
    return selected_blocks

def get_analysis_output_dir(output_base_dir, task_name, cache_mode, **kwargs):
    activations_dir = get_activations_path(output_base_dir, task_name, cache_mode, **kwargs).parent
    return activations_dir / 'bu_block_selection'

@click.command()
@click.option('-o', '--output_base_dir', required=True)
@click.option('-t', '--task_name', required=True)
@click.option('--cache_mode', default='original', 
              type=click.Choice(['original', 'threshold', 'optimal', 'random']))
@click.option('--cache_threshold', default=5, type=int)
@click.option('--num_caches', default=30, type=int)
@click.option('--metric', default='cosine')
@click.option('--num_blocks', default=5, type=int)
@click.option('--force', is_flag=True)
def main(output_base_dir, task_name, cache_mode, cache_threshold, 
         num_caches, metric, num_blocks, force):
    
    kwargs = {
        'cache_threshold': cache_threshold,
        'num_caches': num_caches,
        'metric': metric
    }
    
    activations_path = get_activations_path(output_base_dir, task_name, cache_mode, **kwargs)
    if not activations_path.exists():
        logger.error(f"Activations file does not exist: {activations_path}")
        logger.error(f"Please run collect_activations.py first with cache_mode={cache_mode}")
        return
    
    analysis_output_dir = get_analysis_output_dir(output_base_dir, task_name, cache_mode, **kwargs)
    if analysis_output_dir.exists() and not force:
        selected_path = analysis_output_dir / f'top_{num_blocks}_error_blocks.pkl'
        if selected_path.exists():
            logger.info(f"Analysis results already exist: {selected_path}")
            logger.info("Use --force to force recomputation")
            return
    
    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Analyzing block errors for {cache_mode} mode")
    analyze_block_errors(str(activations_path), str(analysis_output_dir), num_blocks)
    
    logger.info(f"Analysis completed, results saved to: {analysis_output_dir}")

if __name__ == '__main__':
    main() 