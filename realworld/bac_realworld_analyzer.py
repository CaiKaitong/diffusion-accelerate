import sys
import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import logging
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class BACRealworldAnalyzer:
    """BAC Realworld Analyzer - Collects activations and computes optimal steps"""
    
    def __init__(self, policy, output_dir: str, metric: str = 'cosine'):
        """Initialize analyzer"""
        self.policy = policy
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metric = metric
        
        self.activations = defaultdict(list)  # {layer_name: [activations]}
        self.hooks = []
        self.is_collecting = False
        
        logger.info(f"BACRealworldAnalyzer initialized, output: {self.output_dir}")
    
    def start_collecting(self):
        if self.is_collecting:
            logger.warning("Already collecting activations")
            return
        
        if not hasattr(self.policy, 'model'):
            logger.error("Policy does not have 'model' attribute")
            return
        
        hook_count = 0
        dropout_count = 0
        param_count = 0
        
        for name, module in self.policy.model.named_modules():
            should_register = False
            
            if len(list(module.parameters())) > 0:
                should_register = True
                param_count += 1
            
            elif isinstance(module, nn.Dropout):
                should_register = True
                dropout_count += 1
                logger.info(f"Found dropout module: {name}")
            
            if should_register:
                hook = self._register_forward_hook(module, name)
                self.hooks.append(hook)
                hook_count += 1
        
        self.is_collecting = True
    
    def _register_forward_hook(self, layer, layer_name: str):
        def hook_fn(module, input, output):
            if self.is_collecting:
                if isinstance(output, tuple):
                    activation = output[0].detach().cpu()
                else:
                    activation = output.detach().cpu()
                
                self.activations[layer_name].append(activation)
        
        return layer.register_forward_hook(hook_fn)
    
    def stop_collecting(self):
        if not self.is_collecting:
            return
        
        for hook in self.hooks:
            hook.remove()
        
        self.hooks.clear()
        self.is_collecting = False
    
    def save_activations(self, filename: str = "realworld_activations.pkl"):
        save_path = self.output_dir / filename
        with open(save_path, 'wb') as f:
            pickle.dump(self.activations, f)
        return save_path
    
    def compute_optimal_steps(self, num_caches_list: List[int] = [5, 8, 10]):
        if not self.activations:
            return {}
        
        from BACInfer.analysis.optimal_cache_scheduler import OptimalCacheScheduler
        from BACInfer.analysis.activation_analysis import compute_similarity_matrix
        layer_similarity_matrices = {}
        
        for layer_name, acts in self.activations.items():
            if len(acts) < 2:
                continue
            
            shapes = [a.shape for a in acts]
            if len(set(shapes)) > 1:
                continue
            
            try:
                similarity_matrices = compute_similarity_matrix(acts)
                layer_similarity_matrices[layer_name] = similarity_matrices[self.metric]
            except:
                continue
        
        optimal_steps_all = {}
        
        for num_caches in num_caches_list:
            optimal_steps = {}
            
            for layer_name, similarity_matrix in layer_similarity_matrices.items():
                try:
                    scheduler = OptimalCacheScheduler(similarity_matrix=similarity_matrix, metric=self.metric)
                    steps = scheduler.compute_optimal_steps(num_caches)
                    optimal_steps[layer_name] = steps
                except:
                    continue
            
            optimal_steps_all[num_caches] = optimal_steps
        return optimal_steps_all
    
    
    def save_optimal_steps(self, optimal_steps_all: Dict, filename_prefix: str = "optimal_steps"):
        steps_dir = self.output_dir / "optimal_steps" / self.metric
        steps_dir.mkdir(parents=True, exist_ok=True)
        
        first_num_caches = list(optimal_steps_all.keys())[0]
        layer_names = list(optimal_steps_all[first_num_caches].keys())
        
        for layer_name in layer_names:
            layer_dir = steps_dir / layer_name
            layer_dir.mkdir(parents=True, exist_ok=True)
            
            for num_caches, optimal_steps in optimal_steps_all.items():
                if layer_name in optimal_steps:
                    filename = f"{filename_prefix}_{layer_name}_{num_caches}_{self.metric}.pkl"
                    save_path = layer_dir / filename
                    with open(save_path, 'wb') as f:
                        pickle.dump(optimal_steps[layer_name], f)
        return steps_dir
    
    def analyze_episode(self, num_caches_list: List[int] = [5, 8, 10]) -> str:
        if not self.is_collecting:
            self.start_collecting()
        self.stop_collecting()
        self.save_activations()
        optimal_steps_all = self.compute_optimal_steps(num_caches_list)
        steps_dir = self.save_optimal_steps(optimal_steps_all)
        return str(steps_dir)
    
    def clear_activations(self):
        self.activations.clear()
