
__version__ = "1.0.0"

from BACInfer.core.diffusion_cache_wrapper import FastDiffusionPolicy
from BACInfer.analysis import (
    collect_activations,
    compute_similarity_matrix,
    load_activations,
    get_optimal_cache_update_steps,
    analyze_block_errors,
)

__all__ = [
    'FastDiffusionPolicy',
    'collect_activations',
    'compute_similarity_matrix',
    'load_activations',
    'get_optimal_cache_update_steps',
    'analyze_block_errors',
]

