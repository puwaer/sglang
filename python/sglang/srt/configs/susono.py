# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Susono model configuration for SGLang.

Susono extends the Qwen3-Next backbone (hybrid full + GatedDeltaNet linear
attention + MoE with shared expert) with two additional components:

  1. Engram: deterministic N-gram hash-based conditional memory inserted at
     selected transformer layers.
  2. mHC (MHC-Lite): manifold-constrained hyper-connections that maintain
     n parallel residual streams updated via convex combinations of
     permutation matrices.

The field set is kept identical to the validated vLLM port
(``vllm/transformers_utils/configs/susono.py``); the SGLang-specific
``layers_block_type`` / ``linear_layer_ids`` / ``full_attention_layer_ids`` /
``mamba2_cache_params`` properties (mirrored from ``Qwen3NextConfig``) drive the
hybrid GatedDeltaNet state-cache allocation.
"""

import enum

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateShape,
    mamba2_state_dtype,
)
from sglang.srt.configs.update_config import adjust_tp_num_heads_if_necessary
from sglang.srt.utils import is_cpu

logger = logging.get_logger(__name__)
_is_cpu = is_cpu()


class HybridLayerType(enum.Enum):
    full_attention = "attention"
    linear_attention = "linear_attention"


class SusonoConfig(PretrainedConfig):
    """Configuration class for the Susono model."""

    model_type = "susono"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # Base architecture
        vocab_size=151680,
        hidden_size=2048,
        intermediate_size=5120,
        num_hidden_layers=24,
        num_attention_heads=8,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=256,
        # Linear attention (GatedDeltaNet)
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        # MoE
        decoder_sparse_step=1,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts_per_tok=4,
        num_experts=96,
        norm_topk_prob=True,
        output_router_logits=False,
        router_aux_loss_coef=0.002,
        moe_shared_expert_gate_bias_init=2.0,
        mlp_only_layers=None,
        layer_types=None,
        full_attention_interval=4,
        # Susono attention extensions
        qk_layernorm=True,
        attention_output_gate=True,
        # Engram
        use_engram=True,
        engram_max_ngram_size=3,
        engram_n_embed_per_ngram=99991,
        engram_embed_dim=672,
        engram_n_head_per_ngram=8,
        engram_layer_ids=None,
        engram_seed=0,
        engram_base_vocab_size=None,
        # mHC-Lite
        use_mhc=True,
        mhc_num_streams=4,
        mhc_sinkhorn_iterations=20,
        **kwargs,
    ):
        # Base architecture
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim

        # RoPE: accept both rope_scaling / rope_parameters / rope_theta forms
        rope_scaling = kwargs.pop("rope_scaling", None)
        rope_parameters = rope_scaling or rope_parameters or {
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        }
        rope_theta = kwargs.pop("rope_theta", None)
        if rope_theta is not None and "rope_theta" not in rope_parameters:
            rope_parameters["rope_theta"] = rope_theta
        partial_rotary_factor = kwargs.pop("partial_rotary_factor", 0.25)
        if "partial_rotary_factor" not in rope_parameters:
            rope_parameters["partial_rotary_factor"] = partial_rotary_factor
        self.rope_parameters = rope_parameters
        self.partial_rotary_factor = rope_parameters.get(
            "partial_rotary_factor", partial_rotary_factor
        )
        self.rope_theta = rope_parameters.get("rope_theta", 10000.0)
        # SGLang's get_rope reads `rope_scaling`; expose the same dict.
        self.rope_scaling = rope_parameters

        # Layer types: derive from full_attention_interval if not provided
        self.full_attention_interval = full_attention_interval
        if layer_types is None:
            layer_types = [
                "linear_attention"
                if bool((i + 1) % full_attention_interval)
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        self.layer_types = layer_types

        # Linear attention
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads

        # MoE
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.moe_shared_expert_gate_bias_init = moe_shared_expert_gate_bias_init
        self.mlp_only_layers = mlp_only_layers if mlp_only_layers is not None else []

        # Susono attention extensions
        self.qk_layernorm = qk_layernorm
        self.attention_output_gate = attention_output_gate

        # Engram
        self.use_engram = use_engram
        self.engram_max_ngram_size = engram_max_ngram_size
        self.engram_n_embed_per_ngram = engram_n_embed_per_ngram
        self.engram_embed_dim = engram_embed_dim
        self.engram_n_head_per_ngram = engram_n_head_per_ngram
        if engram_layer_ids is None:
            full_attn = [
                i for i, t in enumerate(self.layer_types) if t == "full_attention"
            ]
            if len(full_attn) >= 2:
                self.engram_layer_ids = [full_attn[0], full_attn[-1]]
            else:
                self.engram_layer_ids = list(full_attn)
        else:
            self.engram_layer_ids = list(engram_layer_ids)
        self.engram_seed = engram_seed
        self.engram_base_vocab_size = (
            engram_base_vocab_size if engram_base_vocab_size is not None
            else vocab_size
        )

        # mHC-Lite
        self.use_mhc = use_mhc
        self.mhc_num_streams = mhc_num_streams
        self.mhc_sinkhorn_iterations = mhc_sinkhorn_iterations

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    # ──────────────────────────────────────────────────────────────────────
    # Hybrid-layer / GatedDeltaNet state-cache hooks (mirror Qwen3NextConfig).
    # `layer_types` uses "full_attention"/"linear_attention"; SGLang's hybrid
    # GDN path expects HybridLayerType values ("attention"/"linear_attention").
    # ──────────────────────────────────────────────────────────────────────
    @property
    def layers_block_type(self):
        return [
            HybridLayerType.full_attention.value
            if t == "full_attention"
            else HybridLayerType.linear_attention.value
            for t in self.layer_types
        ]

    @property
    def linear_layer_ids(self):
        return [
            i
            for i, type_value in enumerate(self.layers_block_type)
            if type_value == HybridLayerType.linear_attention.value
        ]

    @property
    def full_attention_layer_ids(self):
        return [
            i
            for i, type_value in enumerate(self.layers_block_type)
            if type_value == HybridLayerType.full_attention.value
        ]

    @property
    def mamba2_cache_params(self) -> Mamba2CacheParams:
        from sglang.srt.layers.dp_attention import get_attention_tp_size

        if _is_cpu:
            world_size = get_attention_tp_size()
            adjust_tp_num_heads_if_necessary(self, world_size, False)

        shape = Mamba2StateShape.create(
            tp_world_size=get_attention_tp_size(),
            intermediate_size=self.linear_value_head_dim * self.linear_num_value_heads,
            n_groups=self.linear_num_key_heads,
            num_heads=self.linear_num_value_heads,
            head_dim=self.linear_value_head_dim,
            state_size=self.linear_key_head_dim,
            conv_kernel=self.linear_conv_kernel_dim,
        )

        return Mamba2CacheParams(
            shape=shape, layers=self.linear_layer_ids, dtype=mamba2_state_dtype(self)
        )


__all__ = ["SusonoConfig"]
