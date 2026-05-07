import logging
import types
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion

logger = logging.getLogger(__name__)


class FastDiffusionPolicy:
    @staticmethod
    def _trace_enabled(cache: Dict) -> bool:
        return bool(cache.get("enable_trace", False))

    @staticmethod
    def _get_progress(cache: Dict) -> float:
        current_step = int(cache.get("current_step", -1))
        num_steps = max(1, int(cache.get("num_steps", 1)))
        if current_step < 0:
            return 0.0
        return float(current_step) / max(1, num_steps - 1)

    @staticmethod
    def _get_online_block_gate(cache: Dict):
        return cache.get("online_block_gate", None)

    @staticmethod
    def _get_online_block_state(cache: Dict):
        if "online_block_state" not in cache:
            cache["online_block_state"] = {
                "last_update_steps": {},
                "ema_ages": {},
                "current_thresholds": {},
                "feature_refs": {},
                "prev_inputs": {},
            }
        return cache["online_block_state"]

    @staticmethod
    def _ensure_block_trace_step(cache: Dict):
        if not FastDiffusionPolicy._trace_enabled(cache):
            return None
        current_step = int(cache.get("current_step", -1))
        if current_step < 0:
            return None
        trace = cache.setdefault("block_trace", [])
        while len(trace) <= current_step:
            trace.append({
                "step_index": len(trace),
                "blocks": {},
            })
        return trace[current_step]

    @staticmethod
    def _register_block_key(cache: Dict, block_key: str):
        block_keys = cache.setdefault("block_keys", [])
        if block_key not in block_keys:
            block_keys.append(block_key)

    @staticmethod
    def _record_block_decision(
        cache: Dict,
        block_key: str,
        reused: bool,
        updated: bool,
        had_cache_before: bool,
        score: float = None,
        force_update_reason: str = None,
        ):
        if not FastDiffusionPolicy._trace_enabled(cache):
            return
        step_trace = FastDiffusionPolicy._ensure_block_trace_step(cache)
        if step_trace is None:
            return

        state = FastDiffusionPolicy._get_online_block_state(cache)
        last_update_step = state["last_update_steps"].get(block_key, None)
        current_step = int(cache.get("current_step", -1))
        cache_age = None if last_update_step is None else current_step - last_update_step
        current_threshold = state["current_thresholds"].get(block_key, None)

        step_trace["blocks"][block_key] = {
            "reused": bool(reused),
            "updated": bool(updated),
            "had_cache_before": bool(had_cache_before),
            "progress": float(FastDiffusionPolicy._get_progress(cache)),
            "cache_age": None if cache_age is None else int(cache_age),
            "threshold": None if current_threshold is None else float(current_threshold),
            "score": None if score is None else float(score),
            "force_update_reason": force_update_reason,
        }

    @staticmethod
    def _block_type_from_key(block_key: str) -> str:
        if block_key.endswith("_sa_block"):
            return "sa_block"
        if block_key.endswith("_mha_block"):
            return "mha_block"
        if block_key.endswith("_ff_block"):
            return "ff_block"
        return "unknown"

    @staticmethod
    def _get_adaptive_block_params(gate: Dict, block_key: str) -> Dict:
        block_type = FastDiffusionPolicy._block_type_from_key(block_key)
        params = gate.get("default_by_type", {}).get(block_type, {}).copy()
        params.update(gate.get("blocks", {}).get(block_key, {}))
        return params

    @staticmethod
    def _resolve_scheduled_threshold(cache: Dict, params: Dict, default_threshold: float) -> float:
        if "threshold_start" not in params and "threshold_end" not in params:
            return float(params.get("threshold", default_threshold))
        start = float(params.get("threshold_start", params.get("threshold", default_threshold)))
        end = float(params.get("threshold_end", params.get("threshold", default_threshold)))
        progress = FastDiffusionPolicy._get_progress(cache)
        return float(start + (end - start) * progress)

    @staticmethod
    def _get_current_adaptive_threshold(cache: Dict, block_key: str, gate: Dict) -> float:
        state = FastDiffusionPolicy._get_online_block_state(cache)
        if block_key in state["current_thresholds"]:
            return float(state["current_thresholds"][block_key])
        params = FastDiffusionPolicy._get_adaptive_block_params(gate, block_key)
        init_threshold = float(
            params.get(
                "init_threshold",
                params.get("min_threshold", params.get("threshold", 1.0)),
            )
        )
        state["current_thresholds"][block_key] = init_threshold
        return init_threshold

    @staticmethod
    def _maybe_update_adaptive_threshold(cache: Dict, block_key: str, cache_age):
        gate = FastDiffusionPolicy._get_online_block_gate(cache)
        if not gate or gate.get("mode", "") != "adaptive_cache_age":
            return
        if cache_age is None:
            return

        params = FastDiffusionPolicy._get_adaptive_block_params(gate, block_key)
        eta = float(params.get("ema_eta", 0.1))
        gamma = float(params.get("threshold_gamma", 0.8))
        min_threshold = float(params.get("min_threshold", 1.0))
        max_threshold = float(params.get("max_threshold", 99.0))

        state = FastDiffusionPolicy._get_online_block_state(cache)
        prev_ema = state["ema_ages"].get(block_key, None)
        age_value = float(cache_age)
        new_ema = age_value if prev_ema is None else (1.0 - eta) * float(prev_ema) + eta * age_value
        new_threshold = round(gamma * new_ema)
        new_threshold = max(min_threshold, min(max_threshold, float(new_threshold)))

        state["ema_ages"][block_key] = new_ema
        state["current_thresholds"][block_key] = new_threshold

    @staticmethod
    def _online_block_should_update(
        cache: Dict,
        block_key: str,
        has_cache_before: bool,
        current_input: torch.Tensor = None,
    ):
        gate = FastDiffusionPolicy._get_online_block_gate(cache)
        if not gate:
            return {
                "should_update": True,
                "score": None,
                "threshold": None,
                "force_update_reason": "no_gate",
            }
        if not has_cache_before:
            return {
                "should_update": True,
                "score": None,
                "threshold": None,
                "force_update_reason": "no_cache",
            }

        state = FastDiffusionPolicy._get_online_block_state(cache)
        last_update_step = state["last_update_steps"].get(block_key, None)
        current_step = int(cache.get("current_step", -1))
        cache_age = None if last_update_step is None else current_step - last_update_step
        params = FastDiffusionPolicy._get_adaptive_block_params(gate, block_key)
        mode = gate.get("mode", "")

        if mode == "feature_distance":
            max_age = params.get("max_age", None)
            if max_age is not None and cache_age is not None and cache_age >= int(max_age):
                return {
                    "should_update": True,
                    "score": None,
                    "threshold": float(params.get("threshold", 0.0)),
                    "force_update_reason": "max_age",
                }

            feature_refs = state.get("feature_refs", {})
            ref_input = feature_refs.get(block_key, None)
            prev_inputs = state.get("prev_inputs", {})
            prev_input = prev_inputs.get(block_key, None)
            score_formula = params.get("score_formula", "delta_last_update_l2_over_input_l2")
            if current_input is None:
                return {
                    "should_update": True,
                    "score": None,
                    "threshold": float(params.get("threshold", 0.0)),
                    "force_update_reason": "missing_current_input",
                }

            eps = float(params.get("eps", 1e-6))
            if score_formula == "delta_prev_l2_over_input_l2":
                if prev_input is None:
                    return {
                        "should_update": True,
                        "score": None,
                        "threshold": float(params.get("threshold", 0.0)),
                        "force_update_reason": "missing_prev_input",
                    }
                numerator = torch.linalg.vector_norm((current_input - prev_input).float().reshape(-1), ord=2)
                denominator = torch.linalg.vector_norm(current_input.float().reshape(-1), ord=2)
            elif score_formula == "delta_last_update_l2_over_input_l2":
                if ref_input is None:
                    return {
                        "should_update": True,
                        "score": None,
                        "threshold": float(params.get("threshold", 0.0)),
                        "force_update_reason": "missing_feature_ref",
                    }
                numerator = torch.linalg.vector_norm((current_input - ref_input).float().reshape(-1), ord=2)
                denominator = torch.linalg.vector_norm(current_input.float().reshape(-1), ord=2)
            else:
                raise RuntimeError(f"Unsupported feature_distance score_formula: {score_formula}")
            score = float((numerator / (denominator + eps)).item())
            threshold = float(params.get("threshold", 0.0))
            return {
                "should_update": bool(score >= threshold),
                "score": score,
                "threshold": threshold,
                "force_update_reason": None,
            }

        if mode == "cosine_similarity":
            threshold = FastDiffusionPolicy._resolve_scheduled_threshold(cache, params, 0.99)
            max_age = params.get("max_age", None)
            if max_age is not None and cache_age is not None and cache_age >= int(max_age):
                return {
                    "should_update": True,
                    "score": None,
                    "threshold": threshold,
                    "force_update_reason": "max_age",
                }

            if threshold > 1.0:
                return {
                    "should_update": True,
                    "score": None,
                    "threshold": threshold,
                    "force_update_reason": None,
                }
            if threshold <= -1.0:
                return {
                    "should_update": False,
                    "score": None,
                    "threshold": threshold,
                    "force_update_reason": None,
                }

            feature_refs = state.get("feature_refs", {})
            ref_input = feature_refs.get(block_key, None)
            if current_input is None:
                return {
                    "should_update": True,
                    "score": None,
                    "threshold": threshold,
                    "force_update_reason": "missing_current_input",
                }
            if ref_input is None:
                return {
                    "should_update": True,
                    "score": None,
                    "threshold": threshold,
                    "force_update_reason": "missing_feature_ref",
                }

            current_flat = current_input.float().flatten()
            ref_flat = ref_input.float().flatten()
            score = float(torch.nn.functional.cosine_similarity(
                current_flat, ref_flat, dim=0, eps=float(params.get("eps", 1e-6))
            ).item())
            return {
                "should_update": bool(score < threshold),
                "score": score,
                "threshold": threshold,
                "force_update_reason": None,
            }

        if mode == "adaptive_cache_age":
            current_threshold = FastDiffusionPolicy._get_current_adaptive_threshold(cache, block_key, gate)
            max_threshold = params.get("max_threshold", None)
            if max_threshold is not None and cache_age is not None and cache_age >= int(max_threshold):
                return {
                    "should_update": True,
                    "score": None,
                    "threshold": float(current_threshold),
                    "force_update_reason": "max_threshold",
                }
            if cache_age is None:
                return {
                    "should_update": True,
                    "score": None,
                    "threshold": float(current_threshold),
                    "force_update_reason": "unknown_age",
                }
            return {
                "should_update": bool(float(cache_age) >= float(current_threshold)),
                "score": None,
                "threshold": float(current_threshold),
                "force_update_reason": None,
            }

        if not params:
            return {
                "should_update": True,
                "score": None,
                "threshold": None,
                "force_update_reason": "missing_params",
            }
        max_age = params.get("max_age", None)
        if max_age is not None and cache_age is not None and cache_age >= int(max_age):
            return {
                "should_update": True,
                "score": None,
                "threshold": float(params.get("threshold", 0.0)),
                "force_update_reason": "max_age",
            }
        if cache_age is None:
            return {
                "should_update": True,
                "score": None,
                "threshold": float(params.get("threshold", 0.0)),
                "force_update_reason": "unknown_age",
            }
        threshold = float(params.get("threshold", 0.0))
        direction = params.get("direction", "high->update")
        if direction == "high->update":
            should_update = float(cache_age) >= threshold
        elif direction == "low->update":
            should_update = float(cache_age) <= threshold
        else:
            should_update = True
        return {
            "should_update": bool(should_update),
            "score": None,
            "threshold": float(threshold),
            "force_update_reason": None,
        }

    @staticmethod
    def _update_online_block_state(
        cache: Dict,
        block_key: str,
        did_update: bool,
        current_input: torch.Tensor = None,
    ):
        state = FastDiffusionPolicy._get_online_block_state(cache)
        gate = FastDiffusionPolicy._get_online_block_gate(cache)
        mode = "" if not gate else gate.get("mode", "")
        if current_input is not None and mode == "feature_distance":
            state["prev_inputs"][block_key] = current_input

        if not gate or not did_update:
            return
        previous_step = state["last_update_steps"].get(block_key, None)
        current_step = int(cache.get("current_step", -1))
        cache_age = None if previous_step is None else current_step - previous_step
        FastDiffusionPolicy._maybe_update_adaptive_threshold(cache, block_key, cache_age)
        state["last_update_steps"][block_key] = current_step
        if current_input is not None:
            state["feature_refs"][block_key] = current_input

    @staticmethod
    def _find_cacheable_layers(model) -> List[Tuple[str, Any]]:
        cacheable_layers = []
        if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
            for i, layer in enumerate(model.decoder.layers):
                if isinstance(layer, nn.TransformerDecoderLayer):
                    name = f"decoder.layers.{i}"
                    cacheable_layers.append((name, layer))
                    logger.info(f"Selected decoder layer for caching: {name}")
        if not cacheable_layers:
            for name, module in model.named_modules():
                if isinstance(module, nn.TransformerDecoderLayer):
                    cacheable_layers.append((name, module))
                    logger.info(f"Selected Transformer layer for caching: {name}")
        logger.info(f"Total {len(cacheable_layers)} cacheable Transformer layers found")
        return cacheable_layers

    @staticmethod
    def _add_cache_to_block(layer, layer_name: str, cache: Dict):
        if not isinstance(layer, nn.TransformerDecoderLayer):
            return

        def forward_with_cache(
            self,
            tgt,
            memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
            tgt_is_causal=None,
            memory_is_causal=False,
        ):
            sa_block_key = f"{layer_name}_sa_block"
            mha_block_key = f"{layer_name}_mha_block"
            ff_block_key = f"{layer_name}_ff_block"
            trace_enabled = FastDiffusionPolicy._trace_enabled(cache)
            if trace_enabled:
                FastDiffusionPolicy._ensure_block_trace_step(cache)
            block_cache = cache["block_cache"]

            x = tgt
            sa_input = self.norm1(x).detach()
            sa_decision = FastDiffusionPolicy._online_block_should_update(
                cache=cache,
                block_key=sa_block_key,
                has_cache_before=(sa_block_key in block_cache),
                current_input=sa_input,
            )
            should_update_sa = bool(sa_decision["should_update"])
            has_sa_cache = sa_block_key in block_cache and not should_update_sa

            if has_sa_cache:
                if trace_enabled:
                    FastDiffusionPolicy._record_block_decision(
                        cache=cache,
                        block_key=sa_block_key,
                        reused=True,
                        updated=False,
                        had_cache_before=True,
                        score=sa_decision.get("score"),
                        force_update_reason=sa_decision.get("force_update_reason"),
                    )
                x = x + block_cache[sa_block_key]
            else:
                sa_result = self._sa_block(sa_input, tgt_mask, tgt_key_padding_mask)
                if should_update_sa:
                    block_cache[sa_block_key] = sa_result.detach()
                if trace_enabled:
                    FastDiffusionPolicy._record_block_decision(
                        cache=cache,
                        block_key=sa_block_key,
                        reused=False,
                        updated=True,
                        had_cache_before=(sa_block_key in block_cache),
                        score=sa_decision.get("score"),
                        force_update_reason=sa_decision.get("force_update_reason"),
                    )
                FastDiffusionPolicy._update_online_block_state(
                    cache,
                    sa_block_key,
                    did_update=bool(should_update_sa),
                    current_input=sa_input,
                )
                x = x + sa_result

            mha_input = self.norm2(x).detach()
            mha_decision = FastDiffusionPolicy._online_block_should_update(
                cache=cache,
                block_key=mha_block_key,
                has_cache_before=(mha_block_key in block_cache),
                current_input=mha_input,
            )
            should_update_mha = bool(mha_decision["should_update"])
            has_mha_cache = mha_block_key in block_cache and not should_update_mha
            if has_mha_cache:
                if trace_enabled:
                    FastDiffusionPolicy._record_block_decision(
                        cache=cache,
                        block_key=mha_block_key,
                        reused=True,
                        updated=False,
                        had_cache_before=True,
                        score=mha_decision.get("score"),
                        force_update_reason=mha_decision.get("force_update_reason"),
                    )
                x = x + block_cache[mha_block_key]
            else:
                mha_result = self._mha_block(mha_input, memory, memory_mask, memory_key_padding_mask)
                if should_update_mha:
                    block_cache[mha_block_key] = mha_result.detach()
                if trace_enabled:
                    FastDiffusionPolicy._record_block_decision(
                        cache=cache,
                        block_key=mha_block_key,
                        reused=False,
                        updated=True,
                        had_cache_before=(mha_block_key in block_cache),
                        score=mha_decision.get("score"),
                        force_update_reason=mha_decision.get("force_update_reason"),
                    )
                FastDiffusionPolicy._update_online_block_state(
                    cache,
                    mha_block_key,
                    did_update=bool(should_update_mha),
                    current_input=mha_input,
                )
                x = x + mha_result

            ff_input = self.norm3(x).detach()
            ff_decision = FastDiffusionPolicy._online_block_should_update(
                cache=cache,
                block_key=ff_block_key,
                has_cache_before=(ff_block_key in block_cache),
                current_input=ff_input,
            )
            should_update_ff = bool(ff_decision["should_update"])
            has_ff_cache = ff_block_key in block_cache and not should_update_ff
            if has_ff_cache:
                if trace_enabled:
                    FastDiffusionPolicy._record_block_decision(
                        cache=cache,
                        block_key=ff_block_key,
                        reused=True,
                        updated=False,
                        had_cache_before=True,
                        score=ff_decision.get("score"),
                        force_update_reason=ff_decision.get("force_update_reason"),
                    )
                x = x + block_cache[ff_block_key]
            else:
                ff_result = self._ff_block(ff_input)
                if should_update_ff:
                    block_cache[ff_block_key] = ff_result.detach()
                if trace_enabled:
                    FastDiffusionPolicy._record_block_decision(
                        cache=cache,
                        block_key=ff_block_key,
                        reused=False,
                        updated=True,
                        had_cache_before=(ff_block_key in block_cache),
                        score=ff_decision.get("score"),
                        force_update_reason=ff_decision.get("force_update_reason"),
                    )
                FastDiffusionPolicy._update_online_block_state(
                    cache,
                    ff_block_key,
                    did_update=bool(should_update_ff),
                    current_input=ff_input,
                )
                x = x + ff_result

            return x

        layer.forward = types.MethodType(forward_with_cache, layer)

    @staticmethod
    def apply_cache(policy, cache_mode=None, online_block_gate: Dict = None, **kwargs):
        if cache_mode in (None, "original"):
            logger.info("Using original policy path directly.")
            return policy
        if cache_mode not in ("adaptive", "hybrid"):
            raise RuntimeError(
                "BAC_final_test supports cache_mode='original', 'skip', 'adaptive', or 'hybrid'. "
                "Step skip is handled in the policy sampling loop; block-level adaptive reuse is handled here."
            )
        if online_block_gate is None:
            raise RuntimeError("online_block_gate config is required for adaptive/hybrid modes")

        assert hasattr(policy, "model")
        model = policy.model
        assert isinstance(model, TransformerForDiffusion)

        cache = {
            "mode": cache_mode,
            "num_steps": policy.num_inference_steps,
            "current_step": -1,
            "block_cache": {},
            "online_block_gate": online_block_gate,
            "block_trace": [],
            "block_keys": [],
            "enable_trace": bool(kwargs.get("enable_trace", False)),
        }
        policy._cache = cache

        cacheable_layers = FastDiffusionPolicy._find_cacheable_layers(model)
        policy._cacheable_layers = cacheable_layers
        logger.info(f"Found {len(cacheable_layers)} cacheable Transformer layers for adaptive reuse")

        for layer_name, _ in cacheable_layers:
            FastDiffusionPolicy._register_block_key(cache, f"{layer_name}_sa_block")
            FastDiffusionPolicy._register_block_key(cache, f"{layer_name}_mha_block")
            FastDiffusionPolicy._register_block_key(cache, f"{layer_name}_ff_block")

        for layer_name, layer in cacheable_layers:
            FastDiffusionPolicy._add_cache_to_block(layer, layer_name, cache)

        original_forward = model.forward

        def forward_with_cache(self, sample, timestep, cond=None, **kwargs):
            cache["current_step"] += 1
            if cache["enable_trace"]:
                FastDiffusionPolicy._ensure_block_trace_step(cache)
            return original_forward(sample, timestep, cond, **kwargs)

        model.forward = types.MethodType(forward_with_cache, model)

        def reset_cache(self):
            if hasattr(self, "_cache"):
                self._cache["current_step"] = -1
                self._cache["block_cache"] = {}
                self._cache["block_trace"] = []
                self._cache["online_block_state"] = {
                    "last_update_steps": {},
                    "ema_ages": {},
                    "current_thresholds": {},
                    "feature_refs": {},
                    "prev_inputs": {},
                }
            return self

        def advance_cache_step(self):
            if hasattr(self, "_cache"):
                self._cache["current_step"] += 1
                if self._cache["enable_trace"]:
                    FastDiffusionPolicy._ensure_block_trace_step(self._cache)
            return self

        policy.reset_cache = types.MethodType(reset_cache, policy)
        policy.advance_cache_step = types.MethodType(advance_cache_step, policy)

        original_predict_action = policy.predict_action

        def predict_action_with_auto_reset(self, *args, **kwargs):
            self.reset_cache()
            return original_predict_action(*args, **kwargs)

        policy.predict_action = types.MethodType(predict_action_with_auto_reset, policy)
        logger.info("Applied adaptive block-level reuse wrapper successfully")
        return policy
