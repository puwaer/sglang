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
"""Inference-only Susono model for SGLang.

Susono extends the Qwen3-Next backbone (hybrid full + GatedDeltaNet attention
+ MoE with shared expert) with two extra components:

  - Engram: deterministic N-gram hash memory injected at selected layers.
  - mHC (MHC-Lite): manifold-constrained hyper-connections that maintain
    multiple parallel residual streams across layers.

Structure mirrors ``models/qwen3_next.py``; the Engram and mHC modules are
ported from the validated vLLM implementation
(``vllm/model_executor/models/susono.py``).  This port targets TP=1 / bf16
correctness — the mHC multi-stream path keeps an explicit ``[n, T, D]`` tensor
and re-materialises each layer's full output rather than using SGLang's fused
deferred-residual ``LayerCommunicator``.
"""

import itertools
import logging
import math
from typing import Iterable, Optional, Set, Tuple

import torch
import triton
from torch import nn

from sglang.srt.configs.susono import SusonoConfig
from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm as RMSNormGated
from sglang.srt.layers.attention.mamba.mamba import mamba_v2_sharded_weight_loader
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.models.qwen2_moe import Qwen2MoeMLP
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    add_prefix,
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_npu,
    make_layers,
    set_weight_attrs,
)

logger = logging.getLogger(__name__)

from sglang.jit_kernel.triton.gdn_fused_proj import fused_qkvzba_split_reshape_cat
from sglang.srt.layers.attention.fla.fused_norm_gate import FusedRMSNormGated

_is_cuda = is_cuda()
_is_npu = is_npu()
_is_cpu = is_cpu()
_is_amx_available = cpu_has_amx_support()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: derive per-sequence boundaries from the flat `positions` tensor.
# ──────────────────────────────────────────────────────────────────────────────


def _derive_seq_boundaries(
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Derive per-token sequence boundaries from a flat positions tensor.

    Returns:
        seq_ids:     [T] 0-indexed sequence ID of each token
        seq_end_idx: [T] exclusive end index (in the flat batch) of the
                     sequence containing each token
    """
    T = positions.shape[0]
    device = positions.device
    dtype = positions.dtype
    if T == 0:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty

    prev_positions = torch.cat(
        [torch.full((1,), -1, dtype=dtype, device=device), positions[:-1]]
    )
    is_new_seq = positions != prev_positions + 1
    seq_ids = is_new_seq.cumsum(0) - 1  # [T], long

    arange_T = torch.arange(T, device=device, dtype=torch.long)
    next_new_seq_idx = torch.cat(
        [is_new_seq[1:], torch.ones(1, dtype=torch.bool, device=device)]
    )
    seq_end_inclusive = arange_T.masked_fill(~next_new_seq_idx, T)
    seq_end_inclusive = seq_end_inclusive.flip(0).cummin(0).values.flip(0)
    seq_end_inclusive = seq_end_inclusive.clamp(max=T - 1)
    seq_end_idx = seq_end_inclusive + 1  # exclusive
    return seq_ids.to(torch.long), seq_end_idx.to(torch.long)


# ──────────────────────────────────────────────────────────────────────────────
# Engram: N-gram hash conditional memory.
# ──────────────────────────────────────────────────────────────────────────────


class SusonoCompressedTokenizer(nn.Module):
    """Maps raw token IDs to a compressed vocabulary via a precomputed
    integer lookup table. The mapping buffer is loaded from the checkpoint."""

    def __init__(self, base_vocab_size: int, seed: int = 0) -> None:
        super().__init__()
        self.base_vocab_size = base_vocab_size
        self.register_buffer(
            "mapping",
            torch.arange(base_vocab_size, dtype=torch.long),
            persistent=True,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.mapping[token_ids]


class SusonoNgramHashMapping(nn.Module):
    """Deterministic N-gram XOR-mix hashing.

    Multipliers/offsets are seeded deterministically from
    (engram_seed, layer_id).  They are also saved in the checkpoint (HF
    registers them as persistent buffers); loading the trained values avoids
    cross-torch-version RNG divergence, so they are registered persistent and
    included in the load_weights param dict.
    """

    def __init__(self, config, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.max_ngram_size = config.engram_max_ngram_size
        self.n_head_per_ngram = config.engram_n_head_per_ngram

        self.primes = [
            config.engram_n_embed_per_ngram
            for _ in range(2, self.max_ngram_size + 1)
        ]
        self.vocab_sizes = [p * self.n_head_per_ngram for p in self.primes]
        offsets = [0]
        for vs in self.vocab_sizes[:-1]:
            offsets.append(offsets[-1] + vs)
        self.register_buffer(
            "offsets",
            torch.tensor(offsets, dtype=torch.long),
            persistent=False,
        )

        multipliers = self._build_multipliers(config, layer_id)
        self.register_buffer("multipliers", multipliers, persistent=False)

    @staticmethod
    def _build_multipliers(config, layer_id: int) -> torch.Tensor:
        shape = (
            config.engram_max_ngram_size - 1,
            config.engram_n_head_per_ngram,
            config.engram_max_ngram_size,
        )
        gen = torch.Generator()
        gen.manual_seed(config.engram_seed * 10007 + layer_id * 1009)
        mults = torch.randint(
            1, 2**31, shape, generator=gen, dtype=torch.long, device="cpu"
        )
        mults = mults * 2 + 1  # ensure odd
        return mults

    def forward(
        self,
        compressed_ids: torch.Tensor,  # [T] flat
        seq_end_idx: torch.Tensor,  # [T] exclusive end index per token
    ) -> torch.Tensor:
        """Returns hash indices [T, total_heads]."""
        T = compressed_ids.shape[0]
        device = compressed_ids.device
        arange_T = torch.arange(T, device=device, dtype=torch.long)

        all_indices = []
        for order_idx, k in enumerate(range(2, self.max_ngram_size + 1)):
            prime = self.primes[order_idx]
            offset = self.offsets[order_idx]

            ngrams_per_shift = []
            for j in range(k):
                shifted_idx = torch.minimum(arange_T + j, seq_end_idx - 1)
                ngrams_per_shift.append(compressed_ids[shifted_idx])
            ngrams = torch.stack(ngrams_per_shift, dim=-1)  # [T, k]

            mults = self.multipliers[order_idx, :, :k].to(device)  # [n_head, k]
            products = ngrams.unsqueeze(1) * mults.unsqueeze(0)  # [T, n_head, k]
            hash_val = products[..., 0]
            for pos in range(1, k):
                hash_val = hash_val ^ products[..., pos]

            head_offset = offset + torch.arange(
                self.n_head_per_ngram, device=device, dtype=torch.long
            ) * prime
            indices = (hash_val % prime) + head_offset  # [T, n_head]
            all_indices.append(indices)

        return torch.cat(all_indices, dim=-1)  # [T, total_heads]


class SusonoMultiHeadEmbedding(nn.Module):
    """Flat embedding table covering all N-gram orders and hash heads."""

    def __init__(self, total_rows: int, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.table = nn.Embedding(total_rows, embed_dim)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.table(indices)


class SusonoShortConv(nn.Module):
    """Per-sequence causal 1-D depthwise convolution on a flat [T, C] tensor."""

    def __init__(self, channels: int, kernel_size: int = 4) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=channels,  # depthwise
        )

    def forward(
        self,
        x: torch.Tensor,  # [T, C]
        query_start_loc: torch.Tensor,  # [num_seqs + 1]
    ) -> torch.Tensor:
        ks = self.kernel_size
        out = torch.empty_like(x)
        num_seqs = query_start_loc.shape[0] - 1
        starts = query_start_loc[:-1]
        ends = query_start_loc[1:]
        for s in range(num_seqs):
            start = int(starts[s].item())
            end = int(ends[s].item())
            if end <= start:
                continue
            seg = x[start:end].transpose(0, 1).unsqueeze(0)  # [1, C, S]
            conv_out = self.conv(seg)  # [1, C, S + ks - 1]
            if ks > 1:
                conv_out = conv_out[..., : -(ks - 1)]  # causal trim
            out[start:end] = conv_out.squeeze(0).transpose(0, 1)
        return out


class SusonoEngramModule(nn.Module):
    """Engram conditional N-gram memory injected at selected layers.

    Operates on [num_tokens, D] hidden states with sequence boundaries derived
    from the flat positions tensor.
    """

    def __init__(self, config, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        hidden_size = config.hidden_size

        self.tokenizer = SusonoCompressedTokenizer(
            config.engram_base_vocab_size, config.engram_seed
        )
        self.ngram_hash = SusonoNgramHashMapping(config, layer_id)

        total_rows = sum(self.ngram_hash.vocab_sizes)
        self.multi_head_emb = SusonoMultiHeadEmbedding(
            total_rows, config.engram_embed_dim
        )

        num_total_heads = (
            (config.engram_max_ngram_size - 1) * config.engram_n_head_per_ngram
        )
        self.short_conv = SusonoShortConv(
            channels=config.engram_embed_dim, kernel_size=4
        )

        self.head_proj = nn.Linear(
            num_total_heads * config.engram_embed_dim,
            config.engram_embed_dim,
            bias=False,
        )
        self.gate_proj = nn.Linear(
            hidden_size, config.engram_embed_dim, bias=False
        )
        self.out_proj = nn.Linear(
            config.engram_embed_dim, hidden_size, bias=False
        )

    def forward(
        self,
        input_ids: torch.Tensor,  # [T] flat token ids
        hidden_states: torch.Tensor,  # [T, D]
        positions: torch.Tensor,  # [T] flat positions
    ) -> torch.Tensor:
        """Returns the memory increment to add to hidden_states. [T, D]"""
        T, _ = hidden_states.shape
        if T == 0:
            return hidden_states.new_zeros(hidden_states.shape)

        _, seq_end_idx = _derive_seq_boundaries(positions)

        device = input_ids.device
        seq_end_unique, _ = torch.unique_consecutive(
            seq_end_idx, return_inverse=True
        )
        cu_seqlens = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device), seq_end_unique]
        )

        compressed = self.tokenizer(input_ids)  # [T]
        indices = self.ngram_hash(compressed, seq_end_idx)  # [T, total_heads]
        emb = self.multi_head_emb(indices)  # [T, total_heads, D_e]
        emb = self.head_proj(emb.reshape(T, -1))  # [T, D_e]
        emb = self.short_conv(emb, cu_seqlens)  # [T, D_e]

        gate = torch.sigmoid(self.gate_proj(hidden_states))  # [T, D_e]
        emb = gate * emb
        return self.out_proj(emb)  # [T, D]


# ──────────────────────────────────────────────────────────────────────────────
# mHC-Lite: Manifold-Constrained Hyper-Connections.
# X tensor convention: [n, T, D] where n = num_streams, T = num_tokens.
# ──────────────────────────────────────────────────────────────────────────────


_susono_perm_mats_cache: dict = {}


def _get_susono_perm_mats(n: int, device: torch.device) -> torch.Tensor:
    """Return all n! permutation matrices [n!, n, n], cached per (n, device)."""
    key = (n, str(device))
    if key not in _susono_perm_mats_cache:
        perms = list(itertools.permutations(range(n)))
        idx = torch.tensor(perms, dtype=torch.long)
        eye = torch.eye(n, dtype=torch.float32)
        _susono_perm_mats_cache[key] = eye[idx].to(device)  # [n!, n, n]
    return _susono_perm_mats_cache[key]


class _SusonoPlainRMSNorm(nn.Module):
    """Pure-torch (1+weight) RMSNorm, identical to HF SusonoRMSNorm.

    Used for the mHC stream norm whose dim is n*hidden_size (e.g. 8192)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        x = x * (1.0 + self.weight.to(torch.float32))
        return x.to(orig_dtype)


class SusonoMHC(nn.Module):
    """MHC-Lite manifold-constrained hyper-connections."""

    def __init__(
        self,
        hidden_size: int,
        num_streams: int,
        layer_index: int = 0,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_streams = num_streams
        n = num_streams
        num_perms = math.factorial(n)
        self.num_perms = num_perms

        init_idx = layer_index % n

        self.norm = _SusonoPlainRMSNorm(hidden_size * n, eps=rms_norm_eps)

        init_alpha_pre = torch.ones(n) * -1.0
        init_alpha_pre[init_idx] = 1.0
        init_alpha_res = torch.ones(num_perms) * -8.0
        init_alpha_res[0] = 0.0  # identity permutation
        self.static_alpha = nn.Parameter(torch.cat([init_alpha_pre, init_alpha_res]))

        self.dynamic_alpha_fn = nn.Parameter(
            torch.zeros(hidden_size * n, n + num_perms)
        )
        self.pre_branch_scale = nn.Parameter(torch.ones(1) * 1e-2)
        self.residual_scale = nn.Parameter(torch.ones(1) * 1e-2)

        init_beta = torch.ones(n) * -1.0
        init_beta[init_idx] = 1.0
        self.static_beta = nn.Parameter(init_beta)
        self.dynamic_beta_fn = nn.Parameter(torch.zeros(hidden_size * n, n))
        self.h_post_scale = nn.Parameter(torch.ones(1) * 1e-2)

    def forward(self, X: torch.Tensor):
        """X: [n, T, D] → (branch_input [T, D], add_residual closure)."""
        n = self.num_streams
        T, D = X.shape[1], X.shape[2]
        device = X.device
        dtype = X.dtype

        X_perm = X.permute(1, 0, 2).reshape(T, n * D)
        normed = self.norm(X_perm)

        wc = normed.to(self.dynamic_alpha_fn.dtype) @ self.dynamic_alpha_fn
        static = self.static_alpha.to(wc.dtype)
        alpha_pre = torch.sigmoid(
            self.pre_branch_scale * wc[:, :n] + static[:n]
        )  # [T, n]
        res_coeff = torch.softmax(
            self.residual_scale * wc[:, n:] + static[n:], dim=-1
        )  # [T, n!]

        P = _get_susono_perm_mats(n, device).to(res_coeff.dtype)  # [n!, n, n]
        H_res = torch.einsum("tp,pij->tij", res_coeff, P)

        alpha_pre_d = alpha_pre.to(dtype)
        branch_input = (alpha_pre_d.t().unsqueeze(-1) * X).sum(dim=0)

        X_tnd = X.permute(1, 0, 2)
        residual_streams = torch.einsum("tij,tjd->tid", H_res.to(dtype), X_tnd)

        dyn_beta = normed.to(self.dynamic_beta_fn.dtype) @ self.dynamic_beta_fn
        beta = (
            dyn_beta * self.h_post_scale
            + self.static_beta.to(dyn_beta.dtype).unsqueeze(0)
        )
        beta = 2.0 * torch.sigmoid(beta)  # [T, n]
        beta_d = beta.to(dtype)

        def add_residual(x_out: torch.Tensor) -> torch.Tensor:
            contrib = beta_d.t().unsqueeze(-1) * x_out.unsqueeze(0)
            return contrib + residual_streams.permute(1, 0, 2)

        return branch_input, add_residual


# ──────────────────────────────────────────────────────────────────────────────
# MoE block with shared expert (shared_expert_gate has a bias, unlike Qwen2Moe).
# ──────────────────────────────────────────────────────────────────────────────


class SusonoSparseMoeBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: SusonoConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.layer_id = layer_id
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=config.norm_topk_prob,
            layer_id=layer_id,
        )

        self.experts = get_moe_impl_class(quant_config)(
            layer_id=self.layer_id,
            top_k=config.num_experts_per_tok,
            num_experts=config.num_experts
            + get_global_server_args().ep_num_redundant_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            routing_method_type=RoutingMethodType.RenormalizeNaive,
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )

        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_expert", prefix),
            )
        else:
            self.shared_expert = None

        # Susono's shared-expert gate has a bias (checkpoint:
        # mlp.shared_expert_gate.bias) — Qwen2Moe hardcodes bias=False, so we
        # define our own gate here.
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        shared_output = None
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            shared_output = (
                torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared_output
            )

        router_logits, _ = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        final_hidden_states = self.experts(hidden_states, topk_output)

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


# ──────────────────────────────────────────────────────────────────────────────
# GatedDeltaNet linear attention (ported from Qwen3GatedDeltaNet).
# ──────────────────────────────────────────────────────────────────────────────


class SusonoGatedDeltaNet(nn.Module):
    def __init__(
        self,
        config: SusonoConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_tp_size = get_attention_tp_size()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.alt_stream = alt_stream

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_id = layer_id
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            quant_config=None,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("conv1d", prefix),
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        self.in_proj_qkvz = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[self.key_dim, self.key_dim, self.value_dim, self.value_dim],
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("in_proj_qkvz", prefix),
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )

        self.in_proj_ba = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[self.num_v_heads] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("in_proj_ba", prefix),
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )

        # The Susono checkpoint stores in_proj_qkvz / in_proj_ba as single fused
        # tensors (shard_id=None). Override the loader to do a contiguous TP
        # slice for that case, delegating split checkpoints to the standard
        # MergedColumnParallelLinear loader.
        self._override_weight_loader(
            self.in_proj_qkvz, self._make_packed_weight_loader(self.in_proj_qkvz)
        )
        self._override_weight_loader(
            self.in_proj_ba, self._make_packed_weight_loader(self.in_proj_ba)
        )

        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight,
            {
                "weight_loader": mamba_v2_sharded_weight_loader(
                    [query_key_settings, query_key_settings, value_settings],
                    self.attn_tp_size,
                    self.attn_tp_rank,
                )
            },
        )

        self.dt_bias = nn.Parameter(torch.zeros(self.num_v_heads // self.attn_tp_size))
        self.A_log = nn.Parameter(
            torch.zeros(self.num_v_heads // self.attn_tp_size, dtype=torch.float32)
        )
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        self.norm = (
            RMSNormGated(
                self.head_v_dim,
                eps=self.layer_norm_epsilon,
                group_size=None,
                norm_before_gate=True,
                device=torch.get_device_module().current_device(),
                dtype=config.torch_dtype,
            )
            if not get_global_server_args().disable_piecewise_cuda_graph
            else FusedRMSNormGated(
                self.head_v_dim,
                eps=self.layer_norm_epsilon,
                activation=self.activation,
                device=torch.get_device_module().current_device(),
                dtype=config.torch_dtype,
            )
        )

        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            input_is_parallel=True,
            reduce_results=True,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("out_proj", prefix),
        )

        self.attn = RadixLinearAttention(
            layer_id=layer_id,
            num_q_heads=self.num_k_heads // self.attn_tp_size,
            num_k_heads=self.num_k_heads // self.attn_tp_size,
            num_v_heads=self.num_v_heads // self.attn_tp_size,
            head_q_dim=self.head_k_dim,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_weights=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )

    @staticmethod
    def _override_weight_loader(module, new_loader):
        param = module.weight
        if hasattr(param, "_weight_loader"):
            param._weight_loader = new_loader
        else:
            param.weight_loader = new_loader

    @staticmethod
    def _make_packed_weight_loader(module):
        original_loader = module.weight.weight_loader

        def weight_loader(param, loaded_weight, loaded_shard_id=None):
            if loaded_shard_id is None:
                output_dim = getattr(param, "output_dim", None)
                if output_dim is not None and module.tp_size > 1:
                    shard_size = param.data.shape[output_dim]
                    start_idx = module.tp_rank * shard_size
                    loaded_weight = loaded_weight.narrow(
                        output_dim, start_idx, shard_size
                    )
                assert param.data.shape == loaded_weight.shape, (
                    f"Shape mismatch: param {param.data.shape} vs "
                    f"loaded {loaded_weight.shape}"
                )
                param.data.copy_(loaded_weight)
            else:
                original_loader(param, loaded_weight, loaded_shard_id)

        return weight_loader

    def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.attn_tp_size,
            (
                self.head_k_dim
                + self.head_k_dim
                + (self.head_v_dim + self.head_v_dim)
                * self.num_v_heads
                // self.num_k_heads
            ),
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (
            self.num_k_heads // self.attn_tp_size,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads,
        ]

        query, key, value, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=2)
        b, a = torch.split(mixed_ba, split_arg_list_ba, dim=2)

        value = value.reshape(value.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b = b.reshape(b.size(0), self.num_v_heads // self.attn_tp_size)
        a = a.reshape(a.size(0), self.num_v_heads // self.attn_tp_size)

        return query, key, value, z, b, a

    def _forward_input_proj(self, hidden_states: torch.Tensor):
        if (
            _is_cpu
            or _is_npu
            or not get_global_server_args().disable_piecewise_cuda_graph
        ):
            DUAL_STREAM_TOKEN_THRESHOLD = 0
        else:
            DUAL_STREAM_TOKEN_THRESHOLD = 1024

        seq_len, _ = hidden_states.shape
        if (
            self.alt_stream is not None
            and get_is_capture_mode()
            and seq_len < DUAL_STREAM_TOKEN_THRESHOLD
        ):
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
            with torch.cuda.stream(self.alt_stream):
                projected_states_ba, _ = self.in_proj_ba(hidden_states)
            current_stream.wait_stream(self.alt_stream)
        else:
            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
            projected_states_ba, _ = self.in_proj_ba(hidden_states)
        return projected_states_qkvz, projected_states_ba

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        projected_states_qkvz, projected_states_ba = self._forward_input_proj(
            hidden_states
        )

        if self.num_v_heads // self.num_k_heads in [1, 2, 4] and not _is_cpu:
            mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat(
                projected_states_qkvz,
                projected_states_ba,
                triton.cdiv(self.num_k_heads, self.attn_tp_size),
                triton.cdiv(self.num_v_heads, self.attn_tp_size),
                self.head_k_dim,
                self.head_v_dim,
            )
        elif _is_cpu and _is_amx_available:
            mixed_qkv, z, b, a = (
                torch.ops.sgl_kernel.fused_qkvzba_split_reshape_cat_cpu(
                    projected_states_qkvz,
                    projected_states_ba,
                    self.num_k_heads // self.attn_tp_size,
                    self.num_v_heads // self.attn_tp_size,
                    self.head_k_dim,
                    self.head_v_dim,
                )
            )
        else:
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                projected_states_qkvz, projected_states_ba
            )
            query, key, value = map(
                lambda x: x.reshape(x.shape[0], -1), (query, key, value)
            )
            mixed_qkv = torch.cat((query, key, value), dim=-1)

        core_attn_out = self.attn(forward_batch, mixed_qkv=mixed_qkv, a=a, b=b)

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])

        if core_attn_out.shape != z.shape:
            core_attn_out_pad = torch.zeros_like(z)
            core_attn_out_pad[: core_attn_out.shape[0], :] = core_attn_out
            core_attn_out = core_attn_out_pad

        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-2], -1)

        output, _ = self.out_proj(core_attn_out)
        return output


# ──────────────────────────────────────────────────────────────────────────────
# Full softmax attention (QK-norm + output gate + partial rotary).
# ──────────────────────────────────────────────────────────────────────────────


class SusonoAttention(nn.Module):
    def __init__(
        self,
        config: SusonoConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_tp_size = get_attention_tp_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % self.attn_tp_size == 0
        self.num_heads = self.total_num_heads // self.attn_tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= self.attn_tp_size:
            assert self.total_num_kv_heads % self.attn_tp_size == 0
        else:
            assert self.attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.attn_tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = getattr(config, "rope_theta", 10000)
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self.partial_rotary_factor = config.partial_rotary_factor
        self.layer_id = layer_id
        self.alt_stream = alt_stream

        self.attn_output_gate = getattr(config, "attention_output_gate", True)

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            rope_scaling=self.rope_scaling,
            base=self.rope_theta,
            partial_rotary_factor=self.partial_rotary_factor,
            is_neox_style=True,
            dtype=torch.get_default_dtype(),
        )

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=True,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("o_proj", prefix),
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor):
        q_by_head = q.reshape(-1, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        q = q_by_head.view(q.shape)
        k = k_by_head.view(k.shape)
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        output, _ = self.o_proj(attn_output)
        return output


# ──────────────────────────────────────────────────────────────────────────────
# Decoder layer + model + ForCausalLM.
# ──────────────────────────────────────────────────────────────────────────────


class SusonoDecoderLayer(nn.Module):
    def __init__(
        self,
        config: SusonoConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.layer_type = config.layer_types[layer_id]

        if self.layer_type == "linear_attention":
            self.linear_attn = SusonoGatedDeltaNet(
                config,
                layer_id,
                quant_config=quant_config,
                alt_stream=alt_stream,
                prefix=add_prefix("linear_attn", prefix),
            )
        elif self.layer_type == "full_attention":
            self.self_attn = SusonoAttention(
                config,
                layer_id,
                quant_config=quant_config,
                alt_stream=alt_stream,
                prefix=add_prefix("self_attn", prefix),
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        mlp_only_layers = getattr(config, "mlp_only_layers", []) or []
        if (layer_id not in mlp_only_layers) and (
            config.num_experts > 0
            and (layer_id + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = SusonoSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Standard pre-norm block returning the full [T, D] output.

        Unlike the Qwen3-Next layers, this does NOT use LayerCommunicator's
        deferred residual — mHC needs the materialised per-layer output.
        """
        if forward_batch.forward_mode.is_idle():
            return hidden_states

        # Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            attn_out = self.linear_attn(hidden_states, forward_batch)
        else:
            attn_out = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states = residual + attn_out

        # Feed-forward
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if isinstance(self.mlp, SusonoSparseMoeBlock):
            hidden_states = self.mlp(hidden_states, forward_batch)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class SusonoModel(nn.Module):
    def __init__(
        self,
        config: SusonoConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        alt_stream = torch.cuda.Stream() if _is_cuda else None

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        def get_layer(idx: int, prefix: str):
            return SusonoDecoderLayer(
                config,
                idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            )

        self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )

        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Engram modules at selected layers.
        self.engram_active = bool(getattr(config, "use_engram", False)) and bool(
            getattr(config, "engram_layer_ids", [])
        )
        if self.engram_active:
            self._engram_layer_map = {
                layer_idx: i for i, layer_idx in enumerate(config.engram_layer_ids)
            }
            self.engram_modules = nn.ModuleList(
                [
                    SusonoEngramModule(config, layer_id)
                    for layer_id in config.engram_layer_ids
                ]
            )

        # mHC-Lite per-layer modules and final aggregation projection.
        self.use_mhc = bool(getattr(config, "use_mhc", False))
        if self.use_mhc:
            self.mhc_modules = nn.ModuleList(
                [
                    SusonoMHC(
                        config.hidden_size,
                        config.mhc_num_streams,
                        layer_index=i,
                        rms_norm_eps=config.rms_norm_eps,
                    )
                    for i in range(config.num_hidden_layers)
                ]
            )
            self.stream_proj = nn.Linear(
                config.hidden_size * config.mhc_num_streams,
                config.hidden_size,
                bias=False,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        engram_active = self.engram_active and (input_ids is not None)

        if self.use_mhc:
            n = self.config.mhc_num_streams
            X = hidden_states.unsqueeze(0).expand(n, -1, -1).contiguous()

            for i in range(len(self.layers)):
                layer = self.layers[i]
                with get_global_expert_distribution_recorder().with_current_layer(i):
                    x_in, add_residual = self.mhc_modules[i](X)
                    if engram_active and i in self._engram_layer_map:
                        eidx = self._engram_layer_map[i]
                        x_in = x_in + self.engram_modules[eidx](
                            input_ids, x_in, positions
                        )
                    x_out = layer(positions, x_in, forward_batch)
                    X = add_residual(x_out)

            if not forward_batch.forward_mode.is_idle():
                n_s, T, D = X.shape
                X_flat = X.permute(1, 0, 2).reshape(T, n_s * D)
                hidden_states = self.stream_proj(X_flat)
                hidden_states = self.norm(hidden_states)
            else:
                # idle passthrough: collapse streams without final norm
                hidden_states = X.mean(dim=0)
            return hidden_states

        # Non-mHC standard path
        for i in range(len(self.layers)):
            layer = self.layers[i]
            with get_global_expert_distribution_recorder().with_current_layer(i):
                if engram_active and i in self._engram_layer_map:
                    eidx = self._engram_layer_map[i]
                    hidden_states = hidden_states + self.engram_modules[eidx](
                        input_ids, hidden_states, positions
                    )
                hidden_states = layer(positions, hidden_states, forward_batch)

        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.norm(hidden_states)
        return hidden_states


class SusonoForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: SusonoConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        if quant_config is not None and hasattr(quant_config, "packed_modules_mapping"):
            quant_config.packed_modules_mapping = self.packed_modules_mapping
        self.quant_config = quant_config
        self.model = SusonoModel(config, quant_config, prefix=add_prefix("model", prefix))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, inputs_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Engram / mHC submodules have their own gate_proj etc. that must NOT
        # be packed into qkv_proj / gate_up_proj.
        susono_extra_substrings = (
            "engram_modules",
            "mhc_modules",
            "stream_proj",
        )

        params_dict = dict(self.named_parameters())
        # Include persistent buffers (e.g. Engram tokenizer.mapping,
        # ngram_hash.multipliers/offsets) so they can be loaded.
        for buf_name, buf in self.named_buffers():
            if buf_name not in params_dict:
                params_dict[buf_name] = buf
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if name.startswith("mtp."):
                continue

            # Susono stores MoE experts as stacked, gate-up-fused 3D tensors:
            #   experts.gate_up_proj : [E, 2*I, H]  (rows [:I]=gate->w1, [I:]=up->w3)
            #   experts.down_proj    : [E, H, I]    (-> w2)
            if ".mlp.experts.gate_up_proj" in name or ".mlp.experts.down_proj" in name:
                is_gate_up = name.endswith("gate_up_proj")
                if is_gate_up:
                    param_name = name.replace(
                        "experts.gate_up_proj", "experts.w13_weight"
                    )
                else:
                    param_name = name.replace(
                        "experts.down_proj", "experts.w2_weight"
                    )
                if param_name not in params_dict:
                    continue
                param = params_dict[param_name]
                weight_loader = param.weight_loader
                num_local_experts = loaded_weight.shape[0]
                if is_gate_up:
                    half = loaded_weight.shape[1] // 2
                    for eid in range(num_local_experts):
                        weight_loader(
                            param,
                            loaded_weight[eid][:half],
                            param_name,
                            shard_id="w1",
                            expert_id=eid,
                        )
                        weight_loader(
                            param,
                            loaded_weight[eid][half:],
                            param_name,
                            shard_id="w3",
                            expert_id=eid,
                        )
                else:
                    for eid in range(num_local_experts):
                        weight_loader(
                            param,
                            loaded_weight[eid],
                            param_name,
                            shard_id="w2",
                            expert_id=eid,
                        )
                loaded_params.add(param_name)
                continue

            is_susono_extra = any(s in name for s in susono_extra_substrings)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                if is_susono_extra:
                    continue
                replaced_name = name.replace(weight_name, param_name)
                if replaced_name.endswith(".bias") and replaced_name not in params_dict:
                    continue
                if replaced_name not in params_dict:
                    continue
                name = replaced_name
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name.endswith("_scale") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.warning(
                        f"Parameter {name} not found in params_dict, skip loading"
                    )
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None,
        )


EntryClass = SusonoForCausalLM
