import sys
import os
import pathlib
import re
import click
import hydra
import torch
import dill
import numpy as np
import time
import json
import logging
from omegaconf import OmegaConf
from copy import deepcopy

from thop import profile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DIFFUSION_POLICY_ROOT = REPO_ROOT / "diffusion_policy"
if str(DIFFUSION_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_POLICY_ROOT))
BACINFER_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if BACINFER_ROOT not in sys.path:
    sys.path.insert(0, BACINFER_ROOT)
from BACInfer.core.diffusion_cache_wrapper import FastDiffusionPolicy
from diffusion_policy.common.pytorch_util import dict_apply

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("eval_script")

DEFAULT_TMP_DIR = REPO_ROOT / "tmp"


def _safe_name(value):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(value)).strip('_') or "unknown"


def _resolve_output_dir(output_dir, cfg, cache_mode):
    if output_dir:
        return pathlib.Path(output_dir)

    task_name = _safe_name(getattr(cfg.task, 'name', 'task'))
    mode_name = _safe_name(cache_mode or "default")
    return DEFAULT_TMP_DIR / task_name / f"eval_{mode_name}"


def _build_workspace(cls, cfg, output_dir):
    try:
        return cls(cfg, output_dir=output_dir)
    except TypeError as exc:
        if "output_dir" not in str(exc):
            raise
        workspace = cls(cfg)
        workspace.__dict__["output_dir"] = output_dir
        return workspace

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', default=None)
@click.option('-d', '--device', default='cuda:0')
@click.option('--cache_mode', default='skip', type=click.Choice(['original', 'skip', 'adaptive', 'hybrid']))
@click.option('--skip_score_threshold', default=0.03, type=float)
@click.option('--skip_max_stale', default=4, type=int)
@click.option('--skip_min_progress', default=0.0, type=float)
@click.option('--online_block_gate_config', default=None)
@click.option('--skip_video', is_flag=True)

def main(checkpoint, output_dir, device, cache_mode, skip_score_threshold, skip_max_stale, skip_min_progress, online_block_gate_config, skip_video):
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    logger.info(f"Configuration loaded: {cfg._target_}")

    output_dir = _resolve_output_dir(output_dir, cfg, cache_mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = str(output_dir)
    logger.info(f"Using output directory: {output_dir}")

    cls = hydra.utils.get_class(cfg._target_)
    workspace = _build_workspace(cls, cfg, output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    logger.info("Workspace loaded successfully")

    original_policy = workspace.model
    original_policy.to(device)
    original_policy.eval()

    has_model = hasattr(original_policy, 'model')
    logger.info(f"Policy type: {type(original_policy).__name__}")
    if has_model:
        logger.info(f"Model type: {type(original_policy.model).__name__}")

    actions_per_inference = original_policy.n_action_steps
    logger.info(f"Number of action frames per inference: {actions_per_inference}")

    logger.info("Creating policy copy for skip experimentation...")
    fast_policy = deepcopy(original_policy)
    online_block_gate = None
    if online_block_gate_config:
        with open(online_block_gate_config, 'r', encoding='utf-8') as f:
            online_block_gate = json.load(f)
        logger.info(f"Loaded online block gate config from {online_block_gate_config}")

    if cache_mode in ('adaptive', 'hybrid'):
        fast_policy = FastDiffusionPolicy.apply_cache(
            policy=fast_policy,
            cache_mode='adaptive',
            online_block_gate=online_block_gate,
            enable_trace=False,
        )
        logger.info("Applied adaptive block-level gate inside model forward")

    if cache_mode in ('skip', 'hybrid'):
        fast_policy.kwargs = dict(getattr(fast_policy, 'kwargs', {}))
        fast_policy.kwargs['enable_skip'] = True
        fast_policy.kwargs['skip_score_threshold'] = skip_score_threshold
        fast_policy.kwargs['skip_max_stale'] = skip_max_stale
        fast_policy.kwargs['skip_min_progress'] = skip_min_progress
        fast_policy.kwargs['record_traces'] = False
        logger.info(
            "Applying AdaptiveDiffusion-style step skip, threshold=%.3f, max_skip_steps=%d, min_progress=%.2f",
            skip_score_threshold, skip_max_stale, skip_min_progress
        )
    else:
        logger.info("Using mode without outer step skip controller")
    
    B = 1
    To = cfg.n_obs_steps
    
    if hasattr(cfg, 'shape_meta'):
        logger.info("Detected image model configuration")
        obs_dict = {}
        for key, shape in cfg.shape_meta['obs'].items():
            tensor_shape = [B, To] + list(shape['shape'])
            obs_dict[key] = torch.zeros(tensor_shape, device=device, dtype=torch.float32)
    else:
        logger.info("Detected low-dimensional model configuration")
        obs_dim = cfg.obs_dim if hasattr(cfg, 'obs_dim') else None
        
        if obs_dim is None and hasattr(cfg, 'task'):
            task_cfg = OmegaConf.to_container(cfg.task, resolve=True)
            if 'obs_dim' in task_cfg:
                obs_dim = task_cfg['obs_dim']
                logger.info(f"Using obs_dim from configuration: {obs_dim}")
        
        obs_dict = {
            'obs': torch.zeros((B, To, obs_dim), device=device, dtype=torch.float32)
        }
        
        if hasattr(cfg, 'use_past_action') and cfg.use_past_action:
            action_dim = cfg.action_dim
            obs_dict['past_action'] = torch.zeros((B, To, action_dim), device=device, dtype=torch.float32)

    logger.info(f"Created input dictionary with keys: {list(obs_dict.keys())}")
    for key, tensor in obs_dict.items():
        logger.info(f"  {key}: shape={tensor.shape}")
    
    logger.info("Warming up...")
    with torch.inference_mode():
        original_policy.predict_action(obs_dict)
        fast_policy.predict_action(obs_dict)
    logger.info("Warmup complete")
    
    flops_value = 0.0

    logger.info("\nComputing policy FLOPs using thop...")
    obs_dict_copy = {}
    for key, value in obs_dict.items():
        obs_dict_copy[key] = value.clone()
        
    with torch.inference_mode():
        nobs = fast_policy.normalizer.normalize(obs_dict_copy)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        
        if hasattr(fast_policy, 'action_dim'):
            Da = fast_policy.action_dim
        elif hasattr(cfg, 'action_dim'):
            Da = cfg.action_dim
        else:
            Da = cfg.task.action_dim
        
        if hasattr(fast_policy, 'obs_feature_dim'):
            Do = fast_policy.obs_feature_dim
        else:
            if hasattr(cfg, 'obs_dim'):
                Do = cfg.obs_dim
            else:
                Do = cfg.task.obs_dim
        
        logger.info(f"Action dimension: {Da}, Observation feature dimension: {Do}")
        
        device = fast_policy.device
        dtype = torch.float32
        timestep = torch.zeros(B, dtype=torch.long, device=device)
        
        cond = None
        sample = None
        
        action_dim = fast_policy.action_dim
        obs_feature_dim = fast_policy.obs_dim if hasattr(fast_policy, 'obs_dim') else Do

        logger.info(f"Action dimension: {action_dim}, Observation feature dimension: {obs_feature_dim}")
        
        if hasattr(fast_policy, 'obs_encoder'):
            if fast_policy.obs_as_cond:
                this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
                nobs_features = fast_policy.obs_encoder(this_nobs)
                cond = nobs_features.reshape(B, To, -1)
                sample = torch.zeros(size=(B, fast_policy.horizon, action_dim), device=device, dtype=dtype)
            else:
                this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
                nobs_features = fast_policy.obs_encoder(this_nobs)
                nobs_features = nobs_features.reshape(B, To, -1)
                sample = torch.zeros(size=(B, fast_policy.horizon, action_dim+obs_feature_dim), device=device, dtype=dtype)
                sample[:,:To,action_dim:] = nobs_features
        else:
            if hasattr(fast_policy, 'obs_as_cond') and fast_policy.obs_as_cond:
                cond = obs_dict['obs'][:,:To]
                sample = torch.zeros(size=(B, fast_policy.horizon, action_dim), device=device, dtype=dtype)
                if hasattr(fast_policy, 'pred_action_steps_only') and fast_policy.pred_action_steps_only:
                    sample = torch.zeros(size=(B, fast_policy.n_action_steps, action_dim), device=device, dtype=dtype)
            else:
                sample = torch.zeros(size=(B, fast_policy.horizon, action_dim+obs_feature_dim), device=device, dtype=dtype)
                sample[:,:To,action_dim:] = obs_dict['obs'][:,:To]
        
        fast_policy.eval()
        macs, params = profile(fast_policy.model, inputs=(sample, timestep, cond), verbose=False)
        flops_value = macs * 2
        
        logger.info(f"Model parameter count: {params/1e6:.2f} M")
        logger.info(f"Model MACs: {macs/1e9:.4f} G")
        logger.info(f"Model FLOPs: {flops_value/1e9:.4f} G")

    logger.info(f"\n=== Performance Testing (action frames/inference: {actions_per_inference}) ===")
    
    num_trials = 10
    
    logger.info("\n----- Original Policy -----")
    original_durations = []
    
    for i in range(num_trials):
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.inference_mode():
            original_policy.predict_action(obs_dict)
            
        torch.cuda.synchronize()
        duration = time.time() - start_time
        original_durations.append(duration)
        logger.info(f"Run {i+1}/{num_trials}: {duration:.4f} seconds")
    
    avg_original = np.mean(original_durations)
    frequency_original = actions_per_inference / avg_original
    logger.info(f"Average time: {avg_original:.4f} seconds")
    logger.info(f"Action frequency: {frequency_original:.2f} actions/second")
    
    logger.info("\n----- Cached Policy -----")
    fast_durations = []
    
    for i in range(num_trials):
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.inference_mode():
            fast_policy.predict_action(obs_dict)
            
        torch.cuda.synchronize()
        duration = time.time() - start_time
        fast_durations.append(duration)
        logger.info(f"Run {i+1}/{num_trials}: {duration:.4f} seconds")
    
    avg_fast = np.mean(fast_durations)
    frequency_fast = actions_per_inference / avg_fast
    speedup_time = avg_original / avg_fast
    
    logger.info(f"Average time: {avg_fast:.4f} seconds")
    logger.info(f"Action frequency: {frequency_fast:.2f} actions/second")
    logger.info(f"Speedup: {speedup_time:.2f}x")
    
    cache_config = {
        "device": str(device),
        "mode": cache_mode,
        "actions_per_inference": int(actions_per_inference),
        "skip_score_threshold": float(skip_score_threshold),
        "skip_max_stale": int(skip_max_stale),
        "skip_min_progress": float(skip_min_progress),
        "online_block_gate_config": online_block_gate_config,
    }
    
    benchmark_results = {
        **cache_config,
        "original": {
            "avg_time": float(avg_original),
            "frequency": float(frequency_original),
        },
        "fast": {
            "avg_time": float(avg_fast),
            "frequency": float(frequency_fast),
        },
        "speedup": float(speedup_time),
        "flops": float(flops_value),
        "config": OmegaConf.to_container(cfg, resolve=True)
    }
    
    with open(os.path.join(output_dir, 'benchmark_results.json'), 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    logger.info(f"Benchmark results saved to {output_dir}/benchmark_results.json")
    
    env_runner_cfg = OmegaConf.to_container(cfg.task.env_runner, resolve=True)
    if skip_video:
        env_runner_cfg['n_train_vis'] = 0
        env_runner_cfg['n_test_vis'] = 0
        logger.info("Skip video rendering")
    
    env_runner = hydra.utils.instantiate(env_runner_cfg, output_dir=output_dir)
    
    logger.info("\nRunning environment evaluation with cached policy...")
    fast_runner_log = env_runner.run(fast_policy)
    
    test_mean_score = fast_runner_log.get('test/mean_score', 0.0)
    
    eval_results = {
        "mean_score": float(test_mean_score),
        "speedup": float(speedup_time),
        "flops": float(flops_value),
        "cache_mode": cache_mode,
        "skip_score_threshold": float(skip_score_threshold),
        "skip_max_stale": int(skip_max_stale),
    }
    if hasattr(fast_policy, '_skip_stats'):
        eval_results['skip_stats'] = fast_policy._skip_stats
    
    with open(os.path.join(output_dir, 'eval_results.json'), 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {output_dir}/eval_results.json")
    logger.info(f"Test success rate: {test_mean_score:.4f}, Speedup: {speedup_time:.2f}x")
    if flops_value > 0:
        logger.info(f"FLOPs: {flops_value/1e9:.4f} GFLOPs")
    
    json_log = {k: v._path if hasattr(v, '_path') else v for k, v in fast_runner_log.items()}
    out_path = os.path.join(output_dir, 'fast_eval_log.json')
    with open(out_path, 'w') as f:
        json.dump(json_log, f, indent=2, sort_keys=True)
    logger.info(f"Detailed evaluation logs saved to {out_path}")

if __name__ == "__main__":
    main()
