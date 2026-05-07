#!/usr/bin/env python3

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

import torch
import logging
from typing import Dict, List, Set
import pickle
from pathlib import Path
import torch.nn as nn
import copy
import random
import numpy as np
import torch.backends.cudnn as cudnn

os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

DEFAULT_RANDOM_SEED = 11

def set_global_seed(seed=DEFAULT_RANDOM_SEED):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        cudnn.deterministic = True

from BACInfer.analysis.run_policy import run_policy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActivationCollector:
    def __init__(self):
        self.activations = {}
        self.hooks = []
        self.current_timestep = -1
        self.tracked_modules = set()
        self.modules_seen_this_step = set()
        self.last_activations = {}
        
    def _hook_fn(self, name):
        def hook(module, input, output):
            self.modules_seen_this_step.add(name)
            
            if isinstance(output, torch.Tensor):
                if name not in self.activations:
                    self.activations[name] = []
                self.activations[name].append(output.detach().cpu())
                self.last_activations[name] = output.detach().cpu()
            elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                if name not in self.activations:
                    self.activations[name] = []
                self.activations[name].append(output[0].detach().cpu())
                self.last_activations[name] = output[0].detach().cpu()
        return hook
    
    def register_hooks(self, model):
        logger.info("Starting to scan model structure...")
        
        hook_count = 0
        dropout_count = 0
        param_count = 0
        
        exclude_dropout_names = []
        
        for name, module in model.named_modules():
            should_register = False
            
            if len(list(module.parameters())) > 0:
                should_register = True
                param_count += 1
            
            elif isinstance(module, nn.Dropout) and name not in exclude_dropout_names:
                should_register = True
                dropout_count += 1
                logger.info(f"Found dropout module: {name}")
            
            if should_register:
                hook = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)
                hook_count += 1
                self.tracked_modules.add(name)
        
        logger.info(f"Registered {hook_count} hooks (parameter modules: {param_count}, dropout modules: {dropout_count})")
    
    def set_timestep(self, timestep):
        if self.current_timestep != -1 and timestep != self.current_timestep:
            self.handle_step_completion()
        
        self.current_timestep = timestep
        self.modules_seen_this_step = set()
    
    def handle_step_completion(self):
        missing_modules = self.tracked_modules - self.modules_seen_this_step
        
        if missing_modules:
            logger.debug(f"Timestep {self.current_timestep}: Missing activations for {len(missing_modules)} modules")
            
            for module_name in missing_modules:
                if module_name in self.last_activations:
                    if module_name not in self.activations:
                        self.activations[module_name] = []
                    
                    self.activations[module_name].append(self.last_activations[module_name])
                    logger.debug(f"  - Reused previous activation for {module_name}")
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
    def get_activations(self):
        return self.activations
    
    def save_activations(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.activations, f)
        logger.info(f"Saved activations for {len(self.activations)} modules to {save_path}")

def collect_activations(
    checkpoint, 
    output_base_dir, 
    task_name,
    device='cuda:0', 
    demo_idx=0, 
    force_recompute=False,
    cache_mode='original', 
    cache_threshold=5, 
    optimal_steps_dir=None, 
    num_caches=30, 
    metric='cosine',
    num_bu_blocks=3,
    edit_steps=None,
    interpolation_ratio=1.0,
    reference_activations_path=None,
    random_seed=DEFAULT_RANDOM_SEED
):
    if random_seed is not None:
        logger.info(f"Setting random seed to {random_seed}")
        set_global_seed(random_seed)
    
    output_base_dir = Path(output_base_dir)
    if cache_mode == 'original':
        cache_dir_name = 'original'
    elif cache_mode == 'threshold':
        cache_dir_name = f'threshold_{cache_threshold}'
    elif cache_mode == 'optimal':
        cache_dir_name = f'optimal_{metric}_caches_{num_caches}'
    elif cache_mode == 'fix':
        cache_dir_name = f'fix_{metric}_caches_{num_caches}'
        if optimal_steps_dir is None:
            logger.error("Fix mode requires optimal_steps_dir to be specified")
            raise ValueError("Fix mode requires optimal_steps_dir to be specified")
    elif cache_mode == 'propagate':
        cache_dir_name = 'propagate_mode'
    elif cache_mode == 'edit':
        steps_str = '_'.join(map(str, edit_steps)) if edit_steps else 'default'
        cache_dir_name = f'edit_steps_{steps_str}'
    else:
        cache_dir_name = 'error!!'
        assert(f"Unsupported cache mode: {cache_mode}")
    
    if interpolation_ratio < 1.0 and reference_activations_path:
        cache_dir_name = f"{cache_dir_name}_interp_{interpolation_ratio:.2f}"
    
    output_dir = output_base_dir / task_name / cache_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / 'activations.pkl'
    if save_path.exists() and not force_recompute:
        logger.info(f"Found existing activation file: {save_path}")
        try:
            with open(save_path, 'rb') as f:
                activations = pickle.load(f)
            logger.info("Successfully loaded existing activations")
            return activations
        except Exception as e:
            logger.warning(f"Error loading existing file: {e}")
            logger.info("Will recompute activations")
    
    logger.info(f"Computing activations using {cache_mode} mode...")
    logger.info(f"Cache parameters: threshold={cache_threshold}, num_caches={num_caches}, metric={metric}")
    if optimal_steps_dir:
        logger.info(f"  optimal_steps_dir={optimal_steps_dir}")
    if edit_steps:
        logger.info(f"  edit_steps={edit_steps}")
    if num_bu_blocks > 0:
        logger.info(f"  Using BU algorithm with {num_bu_blocks} blocks")
    if interpolation_ratio < 1.0:
        logger.info(f"  interpolation_ratio={interpolation_ratio}")
        if reference_activations_path:
            logger.info(f"  reference_activations_path={reference_activations_path}")
    
    policy, obs_dict = run_policy(
        checkpoint=checkpoint,
        output_dir=str(output_dir),
        device=device,
        demo_idx=demo_idx,
        cache_mode=cache_mode,
        cache_threshold=cache_threshold,
        optimal_steps_dir=optimal_steps_dir,
        num_caches=num_caches,
        metric=metric,
        num_bu_blocks=num_bu_blocks,
        edit_steps=edit_steps,
        interpolation_ratio=interpolation_ratio,
        reference_activations_path=reference_activations_path,
        random_seed=random_seed
    )
    
    if cache_mode != 'original':
        if not hasattr(policy, '_cache'):
            logger.error(f"ERROR: Cache mode {cache_mode} was not properly applied")
            raise RuntimeError(f"Cache mode {cache_mode} was not properly applied to the model")
        
        actual_mode = policy._cache.get('mode')
        if actual_mode != cache_mode:
            logger.error(f"ERROR: Requested cache mode {cache_mode} but actual mode is {actual_mode}")
            raise RuntimeError(f"Requested cache mode {cache_mode} but actual mode is {actual_mode}")
            
        logger.info(f"Verified cache mode {cache_mode} applied correctly")
        
        if cache_mode in ['fix', 'optimal', 'edit']:
            block_steps = policy._cache.get('block_steps', {})
            if not block_steps:
                logger.error(f"ERROR: {cache_mode} mode should have block_steps but none were loaded")
                raise RuntimeError(f"Cache mode {cache_mode} should have block_steps but none were loaded")
            logger.info(f"Loaded {len(block_steps)} module cache steps")
            for i, (key, steps) in enumerate(sorted(block_steps.items())):
                if i >= 3:
                    break
                logger.info(f"  {key}: {steps}")
    
    collector = ActivationCollector()
    if hasattr(policy, 'model'):
        target_model = policy.model
    elif hasattr(policy, 'base_policy') and hasattr(policy.base_policy, 'model'):
        target_model = policy.base_policy.model
    else:
        logger.warning("Could not find standard model structure, using entire policy object")
        target_model = policy
    
    if hasattr(policy, '_cache'):
        cache_info = {
            'mode': policy._cache.get('mode', None),
            'block_steps': policy._cache.get('block_steps', {}),
            'threshold': policy._cache.get('threshold', None),
            'num_bu_blocks': policy._cache.get('num_bu_blocks', 0),
            'interpolation_ratio': policy._cache.get('interpolation_ratio', 1.0)
        }
        logger.info(f"Policy cache info: {cache_info}")
    
    collector.register_hooks(target_model)
    
    original_model_forward = target_model.forward
    
    def patched_forward(self, sample, timestep, cond=None, **kwargs):
        collector.set_timestep(timestep.item() if hasattr(timestep, 'item') else timestep)
        return original_model_forward(sample, timestep, cond, **kwargs)
    
    import types
    target_model.forward = types.MethodType(patched_forward, target_model)
    
    with torch.no_grad():
        if cache_mode != 'original' and hasattr(policy, 'reset_cache'):
            policy.reset_cache()
        action_dict = policy.predict_action(obs_dict)
    
    collector.handle_step_completion()
    collector.save_activations(save_path)
    
    collector.remove_hooks()
    target_model.forward = original_model_forward
    
    logger.info(f"Activation collection complete, saved to: {save_path}")
    return collector.get_activations()

def get_activations_path(output_base_dir, task_name, cache_mode, **kwargs):
    output_base_dir = Path(output_base_dir)
    if cache_mode == 'original':
        cache_dir_name = 'original'
    elif cache_mode == 'threshold':
        cache_dir_name = f'threshold_{kwargs.get("cache_threshold", 5)}'
    elif cache_mode == 'optimal':
        cache_dir_name = f'optimal_{kwargs.get("metric", "cosine")}_caches_{kwargs.get("num_caches", 30)}'
    elif cache_mode == 'fix':
        cache_dir_name = f'fix_{kwargs.get("metric", "cosine")}_caches_{kwargs.get("num_caches", 5)}'
    elif cache_mode == 'propagate':
        cache_dir_name = 'propagate_mode'
    elif cache_mode == 'edit':
        edit_steps = kwargs.get("edit_steps", [])
        steps_str = '_'.join(map(str, edit_steps)) if edit_steps else 'default'
        cache_dir_name = f'edit_steps_{steps_str}'
    else:
        cache_dir_name = cache_mode
    
    interpolation_ratio = kwargs.get("interpolation_ratio", 1.0)
    if interpolation_ratio < 1.0 and kwargs.get("reference_activations_path"):
        cache_dir_name = f"{cache_dir_name}_interp_{interpolation_ratio:.2f}"
    
    return output_base_dir / task_name / cache_dir_name / 'activations.pkl'

if __name__ == '__main__':
    import click
    
    @click.command()
    @click.option('-c', '--checkpoint', required=True)
    @click.option('-o', '--output_base_dir', required=True)
    @click.option('-t', '--task_name', required=True)
    @click.option('-d', '--device', default='cuda:0')
    @click.option('--demo_idx', default=0, type=int)
    @click.option('--force', is_flag=True)
    @click.option('--cache_mode', default='original', 
                  type=click.Choice(['original', 'threshold', 'optimal', 'fix', 'propagate', 'edit']))
    @click.option('--cache_threshold', default=5, type=int)
    @click.option('--optimal_steps_dir', default=None)
    @click.option('--num_caches', default=5, type=int)
    @click.option('--metric', default='cosine')
    @click.option('--bu', is_flag=True)
    @click.option('--edit_steps', default=None)
    @click.option('--interpolation_ratio', default=1.0, type=float)
    @click.option('--reference_activations_path', default=None)
    @click.option('--random_seed', default=DEFAULT_RANDOM_SEED, type=int)
    def main(checkpoint, output_base_dir, task_name, device, demo_idx, force,
             cache_mode, cache_threshold, optimal_steps_dir, num_caches, metric,
             bu, edit_steps, interpolation_ratio, reference_activations_path, random_seed):
      
        
        set_global_seed(random_seed)
        processed_edit_steps = None
        if edit_steps is not None:
            processed_edit_steps = [int(x.strip()) for x in edit_steps.split(',')]
            logger.info(f"Parsed edit steps: {processed_edit_steps}")
        
        collect_activations(
            checkpoint=checkpoint, 
            output_base_dir=output_base_dir,
            task_name=task_name,
            device=device, 
            demo_idx=demo_idx, 
            force_recompute=force,
            cache_mode=cache_mode,
            cache_threshold=cache_threshold,
            optimal_steps_dir=optimal_steps_dir,
            num_caches=num_caches,
            metric=metric,
            num_bu_blocks=3 if bu else 0,
            edit_steps=processed_edit_steps,
            interpolation_ratio=interpolation_ratio,
            reference_activations_path=reference_activations_path,
            random_seed=random_seed
        )
    

    set_global_seed(DEFAULT_RANDOM_SEED)
    main() 