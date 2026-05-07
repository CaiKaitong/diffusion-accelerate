import torch
import numpy as np
import os
from pathlib import Path
import pickle
import logging
from typing import Dict, List, Optional, Any, Union, Tuple

logger = logging.getLogger(__name__)

def wasserstein_distance_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    x = torch.abs(x.flatten())
    y = torch.abs(y.flatten())
    
    x = x / (torch.sum(x) + 1e-8)
    y = y / (torch.sum(y) + 1e-8)
    
    x_cdf = torch.cumsum(x, dim=0)
    y_cdf = torch.cumsum(y, dim=0)
    
    cdf_diff = torch.abs(x_cdf - y_cdf)
    
    wasserstein_dist = torch.sum(cdf_diff).item()
    
    return wasserstein_dist

def compute_similarity_metrics(act_i: torch.Tensor, act_j: torch.Tensor) -> Dict[str, float]:
    act_i = act_i.flatten()
    act_j = act_j.flatten()
    
    mse = torch.mean((act_i - act_j) ** 2).item()
    cos_sim = torch.nn.functional.cosine_similarity(act_i.unsqueeze(0), act_j.unsqueeze(0)).item()
    l1_dist = torch.norm(act_i - act_j, p=1).item()
    l2_dist = torch.norm(act_i - act_j, p=2).item()
    wasserstein_dist = wasserstein_distance_torch(act_i, act_j)
    
    return {
        'mse': mse,
        'cosine': cos_sim,
        'l1': l1_dist,
        'l2': l2_dist,
        'wasserstein': wasserstein_dist
    }

def compute_temporal_metrics(current: torch.Tensor, previous: torch.Tensor) -> Dict[str, float]:
    current = current.flatten()
    previous = previous.flatten()
    
    mse = torch.mean((current - previous) ** 2).item()
    cos_sim = torch.nn.functional.cosine_similarity(
        current.unsqueeze(0), previous.unsqueeze(0)).item()
    
    l1_diff = torch.norm(current - previous, p=1).item()
    l2_diff = torch.norm(current - previous, p=2).item()
    wasserstein_diff = wasserstein_distance_torch(current, previous)
    
    return {
        'mse': mse,
        'cosine': cos_sim,
        'l1': l1_diff,
        'l2': l2_diff,
        'wasserstein': wasserstein_diff
    }

def compute_similarity_matrix(activations: List[torch.Tensor]) -> Dict[str, np.ndarray]:

    n_steps = len(activations)
    # Create matrices for each similarity metric
    matrices = {
        'mse': np.zeros((n_steps, n_steps)),
        'cosine': np.zeros((n_steps, n_steps)),
        'l1': np.zeros((n_steps, n_steps)),
        'l2': np.zeros((n_steps, n_steps)),
        'wasserstein': np.zeros((n_steps, n_steps))
    }
    
    for i in range(n_steps):
        for j in range(n_steps):
            similarities = compute_similarity_metrics(activations[i], activations[j])
      
            for metric_name in matrices.keys():
                if metric_name in similarities:
                    matrices[metric_name][i, j] = similarities[metric_name]
    
    return matrices

def compute_temporal_differences(activations: List[torch.Tensor]) -> Dict[str, List[float]]:
    differences = {
        'mse': [], 'cosine': [], 'l1': [], 'l2': [], 'wasserstein': []
    }
    for t in range(1, len(activations)):
        metrics = compute_temporal_metrics(
            activations[t], 
            activations[t-1]
        )
        for metric_name, value in metrics.items():
            differences[metric_name].append(value)
    
    return differences

def load_activations(activations_path: str) -> Dict[str, List[torch.Tensor]]:
    logger.info(f"Loading activations: {activations_path}")
    with open(activations_path, 'rb') as f:
        return pickle.load(f)

def save_similarity_matrices(similarity_matrices: Dict[str, Dict[str, np.ndarray]], 
                           output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(similarity_matrices, f)
    logger.info(f"Similarity matrices saved to: {output_path}")

def get_module_metrics_from_similarity_data(similarity_matrices_dir: str) -> Dict[str, List[str]]:
    similarity_path = Path(similarity_matrices_dir) / 'similarity_matrices.pkl'
    
    if not similarity_path.exists():
        logger.error(f"Similarity matrices file does not exist: {similarity_path}")
        return {}
    
    with open(similarity_path, 'rb') as f:
        similarity_data = pickle.load(f)
    
    module_metrics = {}
    for module_name, module_data in similarity_data.items():
        if isinstance(module_data, dict):
            module_metrics[module_name] = list(module_data.keys())
    
    return module_metrics

def get_similarity_matrix(similarity_matrices_dir: str, 
                         module_name: str, 
                         metric: str = 'cosine') -> Optional[np.ndarray]:
    similarity_path = Path(similarity_matrices_dir) / 'similarity_matrices.pkl'
    
    if not similarity_path.exists():
        logger.error(f"Similarity matrices file does not exist: {similarity_path}")
        return None
    
    with open(similarity_path, 'rb') as f:
        similarity_data = pickle.load(f)
    
    if module_name not in similarity_data:
        logger.error(f"Module {module_name} does not exist in similarity data")
        return None
    
    module_data = similarity_data[module_name]
    if metric not in module_data:
        logger.error(f"Metric {metric} does not exist in module {module_name} data")
        return None
    
    return module_data[metric]

# Functions moved from draw_cache_activation_error.py

def compute_activation_error(original_activations, cached_activations, module_name, metrics=None):

    if metrics is None:
        metrics = ['mse', 'l1', 'l2', 'cosine']
        
    # Get activations for this module
    if module_name not in original_activations or module_name not in cached_activations:
        return None
        
    orig_acts = original_activations[module_name]
    cache_acts = cached_activations[module_name]
    
    # Make sure we have the same number of timesteps
    min_timesteps = min(len(orig_acts), len(cache_acts))
    if min_timesteps == 0:
        return None
        
    # Compute errors for each timestep
    errors = {metric: [] for metric in metrics}
    
    for t in range(min_timesteps):
        orig_act = orig_acts[t]
        cache_act = cache_acts[t]
        
        # Skip if shapes don't match
        if orig_act.shape != cache_act.shape:
            continue
            
        # Compute different error metrics
        if 'mse' in metrics:
            mse = torch.mean((orig_act - cache_act) ** 2).item()
            errors['mse'].append(mse)
            
        if 'l1' in metrics:
            l1 = torch.mean(torch.abs(orig_act - cache_act)).item()
            errors['l1'].append(l1)
            
        if 'l2' in metrics:
            l2 = torch.sqrt(torch.mean((orig_act - cache_act) ** 2)).item()
            errors['l2'].append(l2)
            
        if 'cosine' in metrics:
            # Flatten tensors for cosine similarity
            orig_flat = orig_act.reshape(-1)
            cache_flat = cache_act.reshape(-1)
            
            # Compute cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                orig_flat.unsqueeze(0), cache_flat.unsqueeze(0)
            ).item()
            
            # Convert to error (1 - similarity)
            errors['cosine'].append(1.0 - cos_sim)
            
        if 'max_abs' in metrics:
            max_abs = torch.max(torch.abs(orig_act - cache_act)).item()
            errors['max_abs'].append(max_abs)
            
    return errors

def analyze_cached_modules_data(original_activations, cached_activations):

    # Find common modules
    common_modules = set(original_activations.keys()) & set(cached_activations.keys())
    
    # Classify modules by type: transformer/attention related vs others
    sa_mha_modules = []
    normal_modules = []
    
    for module_name in common_modules:
        if any(marker in module_name.lower() for marker in 
              ['self', 'mha', 'attention', 'blocks', 'transformer', 'decoder']):
            sa_mha_modules.append(module_name)
        else:
            normal_modules.append(module_name)
    
    # Sort modules by name
    sa_mha_modules.sort()
    normal_modules.sort()
    
    # Compute errors for all modules
    attn_module_errors = {}
    for module_name in sa_mha_modules:
        errors = compute_activation_error(original_activations, cached_activations, module_name)
        if errors:
            attn_module_errors[module_name] = errors
    
    normal_module_errors = {}
    for module_name in normal_modules:
        errors = compute_activation_error(original_activations, cached_activations, module_name)
        if errors:
            normal_module_errors[module_name] = errors
    
    # Compute summary statistics across all attention modules
    summary_data = create_summary_data(original_activations, cached_activations, sa_mha_modules)
    
    return {
        'attention_modules': sa_mha_modules,
        'normal_modules': normal_modules,
        'attention_module_errors': attn_module_errors,
        'normal_module_errors': normal_module_errors,
        'summary_data': summary_data
    }

def create_summary_data(original_activations, cached_activations, module_names):

    # Compute average errors across all modules by timestep
    metrics = ['mse', 'l1', 'cosine']
    all_errors = {metric: [] for metric in metrics}
    
    # Track the number of valid modules for each timestep
    valid_module_counts = []
    max_timesteps = 0
    
    # First pass: find maximum number of timesteps
    for module_name in module_names:
        if module_name in original_activations and module_name in cached_activations:
            orig_acts = original_activations[module_name]
            cache_acts = cached_activations[module_name]
            max_timesteps = max(max_timesteps, min(len(orig_acts), len(cache_acts)))
    
    # Initialize arrays for averaging
    for metric in metrics:
        all_errors[metric] = [[] for _ in range(max_timesteps)]
    valid_module_counts = [0] * max_timesteps
    
    # Collect errors for each module and timestep
    for module_name in module_names:
        errors = compute_activation_error(original_activations, cached_activations, module_name, metrics)
        if errors:
            for metric, metric_errors in errors.items():
                for t, error in enumerate(metric_errors):
                    if t < max_timesteps:
                        all_errors[metric][t].append(error)
                        if metric == metrics[0]:  # Only count once per timestep
                            valid_module_counts[t] += 1
    
    # Compute averages
    avg_errors = {metric: [] for metric in metrics}
    for metric in metrics:
        for t in range(max_timesteps):
            if valid_module_counts[t] > 0:
                avg_errors[metric].append(np.mean(all_errors[metric][t]))
            else:
                avg_errors[metric].append(0)
    
    return {
        'max_timesteps': max_timesteps,
        'avg_errors': avg_errors,
        'valid_module_counts': valid_module_counts
    }

# Functions moved from draw_activation_distribution.py

def compute_activation_statistics(activations, module_name, num_timesteps):

    if module_name not in activations:
        raise ValueError(f"Module {module_name} does not exist in activations.")
    
    # Get activations
    module_activations = activations[module_name]
    
    # Calculate statistics for each timestep
    means = []
    stds = []
    for t in range(min(num_timesteps, len(module_activations))):
        act = module_activations[t]
        means.append(act.mean().item())
        stds.append(act.std().item())
    
    # Prepare data for plotting individual histograms
    step_interval = 10
    timesteps_to_plot = range(0, min(num_timesteps, len(module_activations)), step_interval)
    
    # Calculate global min and max for consistent x-axis
    all_values = torch.cat([module_activations[t].flatten() for t in timesteps_to_plot])
    global_min = all_values.min().item()
    global_max = all_values.max().item()
    
    # Prepare data for overall distribution
    all_activations = torch.cat(module_activations[:num_timesteps], dim=0)
    
    return {
        'means': means,
        'stds': stds,
        'timesteps_to_plot': timesteps_to_plot,
        'global_min': global_min,
        'global_max': global_max,
        'all_activations': all_activations,
        'module_activations': module_activations
    }

def analyze_all_modules_statistics(activations_dict, num_timesteps=100):

    # Get all module names
    module_names = sorted(activations_dict.keys())
    
    # Compute statistics for each module
    all_module_stats = {}
    for module_name in module_names:
        try:
            stats = compute_activation_statistics(activations_dict, module_name, num_timesteps)
            all_module_stats[module_name] = stats
        except Exception as e:
            logger.error(f"Error processing module {module_name}: {e}")
    
    return all_module_stats 