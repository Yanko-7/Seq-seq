# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Full definition of a decoder-only transformer-based language model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""

import math
from functools import partial
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Self

from src.config import Config
from litgpt.scripts.convert_hf_checkpoint import qkv_reassemble
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
)
from flash_attn import flash_attn_interface

_MaskType = Union[torch.Tensor, BlockMask]


class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(
            config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias
        )
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(
                    Block(config, block_idx) for block_idx in range(config.n_layer)
                ),
                ln_f=RMSNorm(config.n_embd, eps=config.norm_eps),
            )
        )
        self.mask_cache: Optional[torch.Tensor] = None
        self.max_seq_length = self.config.block_size

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory
        """
        if value > self.config.block_size:
            raise ValueError(
                f"Cannot attend to {value}, block size is only {self.config.block_size}."
                " This is likely because the input text exceeds the supported context length of this model."
            )
        self._max_seq_length = value
        if not hasattr(self, "cos"):
            # first call
            cos, sin = self.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        # override
        elif value != self.cos.size(0):
            self.cos, self.sin = self.rope_cache(device=self.cos.device)
        # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
        # if the kv cache is expected
        if self.mask_cache is not None and self.mask_cache.shape[-1] < value:
            print(
                f"Warning: KV cache has length {self.mask_cache.shape[-1]} < {value} = max_seq_length. Call 'set_kv_cache' before doing any forwards!"
            )

    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.cos, self.sin = self.rope_cache(device=self.cos.device)

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = 0,
        mask: Optional[_MaskType] = None,
        input_pos: Optional[torch.Tensor] = None,
        input_pos_maxp1: Optional[int] = None,
        lm_head_chunk_size: int = 0,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        If `input_pos` is provided, the KV cache uses K and V vectors for
        positions smaller than entries in `input_pos`. For efficiency, pass
        `input_pos_maxp1` as `max(input_pos) + 1` if already available from
        your forward algorithm. This slices the KV cache buffers and speeds
        up multi-head attention.

        Without `input_pos_maxp1`, the computation uses the full KV cache
        (`max_seq_length`) with masking applied. Note that inferring
        `input_pos_maxp1` from `input_pos` causes graph breaks and prevents
        compilation.

        Args:
            idx: Token indices of input sequences, shape `(B, T)`, where `B`
                is batch size.
            input_pos: Optional. Positions of input tokens. The default is
                `arange(T)`. Can have shape `(T,)` or `(B, T)` (batched index).
            input_pos_maxp1: Optional. See above.
            lm_head_chunk_size: Optional. If `lm_head_chunk_size > 0`, the final
                `lm_head` computation is done in chunks of this size.

        Returns:
            Logit outputs, shape `(B, T, config.padded_vocab_size)`. If
            `lm_head_chunk_size > 0`, this is a list of chunks of shape
            `(B, lm_head_chunk_size, config.padded_vocab_size)`, the final
            entry can be shorter.

        """
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(
                f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}."
            )
        if input_pos is not None:
            # 无论是 Packed 训练还是推理，只要自定义了位置，就按 input_pos 取 cos/sin
            if input_pos.dim() > 2:
                raise ValueError(f"input_pos must be 1D or 2D, got {input_pos.shape}")
            if input_pos.dim() == 1:
                input_pos = input_pos.unsqueeze(0)
            cos = batched_index_select(self.cos, 0, input_pos)
            sin = batched_index_select(self.sin, 0, input_pos)
        else:
            cos = self.cos[:T].unsqueeze(0)
            sin = self.sin[:T].unsqueeze(0)

        # 【修改】不再仅通过 input_pos 判断推理，而是明确判断是否需要裁剪 KV Cache
        if self.mask_cache is not None and input_pos_maxp1 is not None:
            _mask = batched_index_select(self.mask_cache, 2, input_pos)
            if _mask.dim() > 4:
                _mask = _mask.view(*(_mask.shape[0:1] + _mask.shape[2:]))
            if input_pos_maxp1 > self.max_seq_length:
                raise ValueError(f"Positions must be in [0,{self.max_seq_length})")
            # 覆盖 mask 变量，供推理使用
            mask = _mask[..., :input_pos_maxp1]

        x = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        if self.config.scale_embeddings:
            x = x * torch.tensor(self.config.n_embd**0.5, dtype=x.dtype)

        for _, block in enumerate(self.transformer.h):
            x = block(
                x,
                cos,
                sin,
                cu_seqlens,
                max_seq_len,
                mask,
                input_pos,
                input_pos_maxp1,
            )
        x = self.transformer.ln_f(x)
        clamp_head = (
            partial(do_softcapping, thresh=self.config.final_logit_softcapping)
            if self.config.final_logit_softcapping is not None
            else nn.Identity()
        )
        if lm_head_chunk_size > 0:
            # chunk the lm head logits to reduce the peak memory used by autograd
            return [
                clamp_head(self.lm_head(x_i))
                for x_i in x.split(lm_head_chunk_size, dim=1)
            ]
        else:
            return clamp_head(self.lm_head(x))  # (B, T, padded_vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def rope_cache(
        self, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_adjustments is None:
            extra_config = None

        else:
            adjusted_params_required = [
                "factor",
                "low_freq_factor",
                "high_freq_factor",
                "original_max_seq_len",
            ]
            params_present = [
                param in self.config.rope_adjustments
                for param in adjusted_params_required
            ]
            num_params_present = sum(params_present)

            if num_params_present == 0:
                extra_config = None  # uses standard RoPE
            elif num_params_present == 4:
                # These parameters should always be used together so that we don't interfere with standard rope
                extra_config = {
                    name: self.config.rope_adjustments[name]
                    for name in adjusted_params_required
                }
            elif "factor" in self.config.rope_adjustments:
                # linear RoPE
                adjusted_params_required = ["factor"]
                extra_config = {
                    name: self.config.rope_adjustments[name]
                    for name in adjusted_params_required
                }
            else:
                # Some but not all parameters are specified; raise an error
                missing_params = [
                    param
                    for param, present in zip(adjusted_params_required, params_present)
                    if not present
                ]
                raise ValueError(
                    f"The following adjusted RoPE parameters are missing in rope_adjustments: {', '.join(missing_params)}. "
                    "All adjusted RoPE parameters must be specified together."
                )

        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config.rope_n_elem,
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
            extra_config=extra_config,
            rope_local_base_freq=self.config.rope_local_base_freq,
        )

    def set_kv_cache(
        self,
        batch_size: int,
        max_seq_length: Optional[int] = None,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if rope_cache_length is None:
            if len(self.cos.shape) == 2:
                rope_cache_length = self.cos.size(-1)
            else:
                rope_cache_length = self.cos[..., 0].size(-1)

        if max_seq_length is None:
            max_seq_length = self.max_seq_length

        # initialize the kv cache for all blocks
        for block in self.transformer.h:
            block.attn.kv_cache = block.attn.build_kv_cache(
                batch_size,
                max_seq_length,
                rope_cache_length,
                device,
                dtype,
            )

        if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            # passing `attn_mask` to SDPA disables the flash implementation. since we only need the mask
            # for the kv-cache support (only during inference), we only create it in that situation
            self.mask_cache = build_mask_cache(max_seq_length, device)

    def clear_kv_cache(self) -> None:
        self.mask_cache = None
        for block in self.transformer.h:
            block.attn.kv_cache = None


class Block(nn.Module):
    def __init__(
        self,
        config: Config,
        block_idx: int,
    ) -> None:
        super().__init__()
        if not config.parallel_residual and config.shared_attention_norm:
            raise NotImplementedError(
                "No checkpoint amongst the ones we support uses this configuration"
                " (non-parallel residual and shared attention norm)."
            )

        self.norm_1 = (
            nn.Identity()
            if not config.norm_1
            else config.norm_class(config.n_embd, eps=config.norm_eps)
        )
        self.attn = CausalSelfAttention(config, block_idx)
        self.post_attention_norm = (
            config.norm_class(config.n_embd, eps=config.norm_eps)
            if config.post_attention_norm
            else nn.Identity()
        )
        self.norm_2 = (
            nn.Identity()
            if not config.norm_2
            else (
                None
                if config.shared_attention_norm
                else config.norm_class(config.n_embd, eps=config.norm_eps)
            )
        )
        self.mlp = config.mlp_class(config)
        if (
            config.first_k_dense_replace is not None
            and block_idx < config.first_k_dense_replace
        ):
            self.mlp = LLaMAMLP(config)
        self.post_mlp_norm = (
            config.norm_class(config.n_embd, eps=config.norm_eps)
            if config.post_mlp_norm
            else nn.Identity()
        )

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = 0,
        mask: Optional[_MaskType] = None,
        input_pos: Optional[torch.Tensor] = None,
        input_pos_maxp1: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Non-parallel residual       Parallel residual
           ┌─ x                     ┌─ x ──────────────────┐             Note: if `shared_attention_norm` is True,
           │  ↓                     │  ↓                   ↓                   the output from `norm_1` is reused
           │  norm_1                │  norm_1  ───────►    norm_2
           │  ↓                     │  ↓                   ↓
           │  attn                  │  attn                MLP
           │  ↓                     │  ↓                   ↓
           |  post_attn_norm        |  post_attn_norm      post_mlp_norm
           |  ↓                     |  ↓                   ↓
        ┌─ └► +                     └► + ◄─────────────────┘
        |     ↓
        │     norm_2
        │     ↓
        │     MLP
        │     ↓
        |     post_mlp_norm
        |     ↓
        └───► +
        """

        x_normed = self.norm_1(x)
        attention_output = self.attn(
            x_normed,
            cos,
            sin,
            cu_seqlens,
            max_seq_len,
            mask,
            input_pos,
            input_pos_maxp1,
        )
        attention_output = self.post_attention_norm(attention_output)

        if self.config.parallel_residual:
            if not self.config.shared_attention_norm:
                x_normed = self.norm_2(x)
            x = attention_output + x
        else:
            x = attention_output + x
            x_normed = self.norm_2(x)

        return self.post_mlp_norm(self.mlp(x_normed)) + x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__()
        # key, query and value projections for all heads, but in a batch
        self.attention_dropout = config.attention_dropout
        self.qkv = nn.Linear(
            config.n_embd,
            (2 * config.n_head + 2 * config.n_query_groups)
            * config.head_size,  # support for grouped/multi queries
            bias=config.bias or config.attn_bias,
        )
        # output projection
        self.proj = nn.Linear(
            config.head_size * config.n_head, config.n_embd, bias=config.bias
        )
        # disabled by default
        self.kv_cache: Optional[KVCache] = None
        self.apply_sliding_window_attention = False
        if (
            config.sliding_window_size is not None
            and config.sliding_window_indices is not None
        ):
            self.apply_sliding_window_attention = config.sliding_window_indices[
                block_idx
            ]

        if config.norm_qk:
            self.norm_q = RMSNorm(config.head_size, eps=config.norm_eps)
            self.norm_k = RMSNorm(config.head_size, eps=config.norm_eps)
        else:
            self.norm_q = self.norm_k = None

        self.config = config
        self.block_idx = block_idx

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = 0,
        mask: Optional[_MaskType] = None,
        input_pos: Optional[torch.Tensor] = None,
        input_pos_maxp1: Optional[int] = None,
    ) -> torch.Tensor:
        # Notation:
        # - B          | batch size
        # - T          | time-step (sequence length)
        # - C          | model's embeddings size (n_embd)
        # - C*         | attentions's embeddings size
        # - hs         | head size
        # - nh_(q,k,v) | number of heads for query, key and value
        # - n_query_groups = nh_k = nh_v | number of query groups sharing key and value heads
        # alternative notation: num_kv_groups = n_query_groups
        # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
        # │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
        # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
        #   │    │    │    │         │        │                 │
        # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
        # │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
        # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
        #   │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
        # ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
        # │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
        # └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
        # ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
        #         MHA                    GQA                   MQA
        #   n_query_groups=4       n_query_groups=2      n_query_groups=1
        #
        # credit https://arxiv.org/pdf/2305.13245.pdf
        head_size = self.config.head_size
        n_head = self.config.n_head
        n_query_groups = self.config.n_query_groups
        rope_n_elem = self.config.rope_n_elem
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # Perform a single multiplication operation using a combined QKV matrix to calculate `query`, `key`, and `value`
        # instead of individually multiplying the input `x` with the respective weight matrices.
        qkv = self.qkv(x)  # (B, T, 4xC*)

        # Define query, key and value sizes.
        # If grouped/multi query is enabled, these sizes are not equal (see the diagram above).
        query_size = n_head * head_size
        key_size = value_size = n_query_groups * head_size
        # Split qkv into query, key and value matrices.
        gate, q, k, v = qkv.split(
            (query_size, query_size, key_size, value_size), dim=-1
        )  # 4x(B, T, C*)

        # To place the num_heads (nh) dimension right after the batch (B) dimension, the first step is to decouple the
        # embedding size (C) into num_heads (nh) and head_size (hs).

        # The original GQA paper is followed here and the term query groups is used.
        # alternative notation: Query groups are also referred to as KV groups.
        q = q.view(B, T, n_head, head_size)  # (B, T, nh_q, hs)
        k = k.view(B, T, n_query_groups, head_size)  # (B, T, n_query_groups, hs)
        v = v.view(B, T, n_query_groups, head_size)  # (B, T, n_query_groups, hs)

        if self.config.norm_qk:
            q = self.norm_q(q)
            k = self.norm_k(k)

        # Unlike standard positional embeddings rotary embeddings must be applied at every layer.
        q_roped = apply_rope(q[..., :rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., :rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., rope_n_elem:]), dim=-1)  # (B, T, nh_q, hs)
        k = torch.cat((k_roped, k[..., rope_n_elem:]), dim=-1)  # (B, T, nh_k, hs)

        # Apply kv-cache during inference.
        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)

            if self.apply_sliding_window_attention:
                actual_kv_len = k.size(2)
                if mask is not None and mask.size(-1) != actual_kv_len:
                    mask = mask[..., :actual_kv_len]

            if input_pos_maxp1 is not None:
                # Subselect along sequence dimension
                k = k[..., :input_pos_maxp1, :]
                v = v[..., :input_pos_maxp1, :]
            # k, v: (B, nh_k, input_pos_maxp1, hs)
            # If input_pos_maxp1 is None -> max_seq_length

        if mask is None and self.kv_cache is None:  # Packed train
            ############################# Flash Attention  #############################
            y = flash_attn_interface.flash_attn_varlen_func(
                q[0],
                k[0],
                v[0],
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seq_len,
                max_seqlen_k=max_seq_len,
                causal=True,
                dropout_p=self.attention_dropout if self.training else 0.0,
            )

        else:  # Batch inference
            # # The tensors `query`, `key`, and `value` are now accurately structured: within each batch element (B), there are
            # # multiple heads (nh), and within each head, there is a sequence of elements (T), each represented by a vector
            # # of size `hs`.
            q = q.transpose(1, 2)  # (B, nh_q, T, hs)
            k = k.transpose(1, 2)  # (B, nh_k, T, hs)
            v = v.transpose(1, 2)  # (B, nh_v, T, hs)
            ############################# Flex Attention  #############################
            if n_query_groups != n_head and (input_pos is None or n_query_groups != 1):
                q_per_kv = n_head // n_query_groups
                k = k.repeat_interleave(q_per_kv, dim=1)  # (B, nh_q, T, hs)
                v = v.repeat_interleave(q_per_kv, dim=1)  # (B, nh_q, T, hs)
            # ↓ (B, nh, T, hs) @ (B, nh, T, hs).mT --> (B, nh, T, T) @ (B, nh, T, hs) --> (B, nh, T, hs)
            y = self.scaled_dot_product_attention(q, k, v, mask)

        y = y.reshape(B, T, head_size * n_head).contiguous() * torch.sigmoid(gate)
        # Output projection.
        return self.proj(y)  # (B, T, C)

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Union[torch.Tensor, "BlockMask", None] = None,
    ) -> torch.Tensor:

        if isinstance(mask, BlockMask):
            softcap_thresh = self.config.attention_logit_softcapping

            # 如果开启了 softcapping，必须通过 score_mod 注入到 flex_attention 底层 kernel 中
            if softcap_thresh is not None:

                def softcap_mod(score, b, h, q_idx, kv_idx):
                    # flex_attention 已经默认除了 sqrt(head_size)，这里直接做 tanh 截断
                    return softcap_thresh * torch.tanh(score / softcap_thresh)

                y = flex_attention(
                    q, k, v, block_mask=mask, enable_gqa=True, score_mod=softcap_mod
                )
            else:
                y = flex_attention(q, k, v, enable_gqa=True, block_mask=mask)

            return y.transpose(1, 2)

        scale = 1.0 / math.sqrt(
            self.config.attention_scores_scalar or self.config.head_size
        )

        # with softcapping we cannot use SDPA
        if self.config.attention_logit_softcapping is not None:
            scores = q @ k.mT * scale
            scores = do_softcapping(scores, self.config.attention_logit_softcapping)
            if mask is None:
                mask = torch.ones(
                    q.size(2), q.size(2), dtype=q.dtype, device=q.device
                ).triu(diagonal=1)
                mask.masked_fill_(mask.bool(), torch.finfo(q.dtype).min)
            scores = scores + mask
            scores = F.softmax(scores, dim=-1, dtype=torch.float).to(dtype=q.dtype)
            y = scores @ v
        else:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=0.0,
                scale=scale,
                is_causal=mask is None,
            )
        return y.transpose(1, 2)

    def build_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "KVCache":
        if (
            self.apply_sliding_window_attention
            and self.config.sliding_window_size is not None
        ):
            effective_cache_size = min(max_seq_length, self.config.sliding_window_size)
        else:
            effective_cache_size = max_seq_length

        v_shape = (
            batch_size,
            self.config.n_query_groups,
            effective_cache_size,
            self.config.head_size,
        )

        if rope_cache_length is None:
            if self.config.rotary_percentage != 1.0:
                raise TypeError(
                    "Please pass the `rope_cache_length=gpt.cos.size(-1)` value"
                )
            k_shape = v_shape
        else:
            k_shape = (
                batch_size,
                self.config.n_query_groups,
                effective_cache_size,
                rope_cache_length + self.config.head_size - self.config.rope_n_elem,
            )

        return KVCache(
            k_shape,
            v_shape,
            device=device,
            dtype=dtype,
            is_sliding_window=self.apply_sliding_window_attention,
            sliding_window_size=self.config.sliding_window_size
            if self.apply_sliding_window_attention
            else None,
        )

    def _load_from_state_dict(
        self, state_dict: dict, prefix: str, *args: Any, **kwargs: Any
    ) -> None:
        """For compatibility with legacy checkpoints."""

        for attr in ("weight", "bias"):
            legacy_key = f"{prefix}attn.{attr}"
            current_key = f"{prefix}qkv.{attr}"
            if legacy_key in state_dict:
                state_dict[current_key] = qkv_reassemble(
                    state_dict.pop(legacy_key), self.config
                )

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def reset_parameters(self) -> None:
        qkv_std = math.sqrt(2.0 / 5.0 / self.config.n_embd)
        nn.init.normal_(self.qkv.weight, mean=0.0, std=qkv_std)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        proj_std = 1.0 / math.sqrt(self.config.n_embd) / self.config.n_layer
        nn.init.normal_(self.proj.weight, mean=0.0, std=proj_std)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        if self.norm_q is not None:
            nn.init.ones_(self.norm_q.weight)
        if self.norm_k is not None:
            nn.init.ones_(self.norm_k.weight)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config, intermediate_size: Optional[int] = None) -> None:
        super().__init__()
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.fc_1 = nn.Linear(config.n_embd, self.intermediate_size, bias=config.bias)
        self.fc_2 = nn.Linear(config.n_embd, self.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(self.intermediate_size, config.n_embd, bias=config.bias)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = F.silu(x_fc_1) * x_fc_2
        return self.proj(x)


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    device: Optional[torch.device] = None,
    base: int = 10000,
    condense_ratio: int = 1,
    extra_config: Optional[dict] = None,
    rope_local_base_freq: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Enhanced Transformer with Rotary Position Embedding.

    Args:
        seq_len (int): Sequence length.
        n_elem (int): Number of elements (head dimension).
        device (torch.device, optional): Device for tensor allocations.
        base (int, optional): Base for computing inverse frequencies.
        condense_ratio (int, optional): Ratio to condense the position indices.
        extra_config (dict, optional): Configuration parameters for frequency adjustments (used by Llama 3.1 and 3.2)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Cosine and sine caches for RoPE.
            Shapes are `(seq_len, n_elem)`.
    """

    # Compute the inverse frequencies theta
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    if extra_config is not None:
        factor = extra_config["factor"]
        if "original_max_seq_len" in extra_config:
            orig_context_len = extra_config["original_max_seq_len"]
            low_freq_factor = extra_config["low_freq_factor"]
            high_freq_factor = extra_config["high_freq_factor"]

            wavelen = 2 * torch.pi / theta
            ratio = orig_context_len / wavelen
            smooth_factor = (ratio - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            smooth_factor = torch.clamp(smooth_factor, min=0.0, max=1.0)

            # Compute adjusted_theta without masked indexing
            adjusted_theta = (1 - smooth_factor) * (
                theta / factor
            ) + smooth_factor * theta
            theta = adjusted_theta
        else:
            theta = theta / factor

    # Create position indices `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device).float() / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)
    # If `n_elem` is odd, the final dimension of `idx_theta` has size
    # `n_elem + 1`, so need to cut something off.
    # Due to a current bug in Hugging Face, in the case `n_elem == 1`, we leave
    # `idx_theta`, `cos`, `sin` as is. Things work out in `apply_rope` due to
    # broadcasting. If we shorten `idx_theta`, unit tests comparing to
    # Hugging Face fail.
    # https://github.com/huggingface/transformers/issues/35233
    if idx_theta.shape[-1] > n_elem > 1:
        idx_theta = idx_theta[..., :n_elem]

    # if rope_local_base_freq is given, have a separate rope value for local embedding
    # For now, we use default RoPE for local embedding
    if rope_local_base_freq is not None:
        local_theta = 1.0 / (
            rope_local_base_freq
            ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem)
        )
        local_idx_theta = torch.outer(seq_idx, local_theta)
        local_idx_theta = local_idx_theta.repeat(1, 2)
        if local_idx_theta.shape[-1] > n_elem > 1:
            local_idx_theta = local_idx_theta[..., :n_elem]

        idx_theta = torch.stack((idx_theta, local_idx_theta), dim=-1)

    return torch.cos(idx_theta), torch.sin(idx_theta)


def batched_index_select(t, dim, idx):
    """index_select for batched index and unbatched t"""
    if idx.dim() == 1:
        return torch.index_select(t, dim, idx)

    *batch_shape, idx_size = idx.shape
    res = torch.index_select(t, dim, idx.reshape(-1))  # flat index
    # split out single batch idx
    res = res.view(*t.shape[:dim], -1, idx_size, *t.shape[dim + 1 :])
    if dim > 0:
        # move batch dim to front, this is np.rollaxis(res, dim, 0) for tensors
        dims = [dim] + list(range(res.dim()))
        del dims[dim + 1]
        res = res.permute(dims)
    # unflatten batch dims
    res = res.view(*batch_shape, *res.shape[1:])
    return res


def batched_index_copy_(t, dim, idx, val):
    """Index copy for batched t, idx, val"""

    if t.device.type == "mps":
        # Normalize negative dimensions
        if dim < 0:
            dim = t.dim() + dim
        if idx.dim() == 1:
            idx_shape = [1] * val.dim()
            idx_shape[dim] = -1
            idx_expanded = idx.view(*idx_shape)
            idx_expanded = idx_expanded.expand_as(val)
            t.scatter_(dim, idx_expanded, val)
            return t

        elif idx.dim() == 2:
            assert dim != 0, "Cannot index the batch dimension"
            batch_size = idx.size(0)
            idx_size = idx.size(1)
            assert batch_size == t.size(0) == val.size(0)

            idx_shape = [batch_size] + [1] * (val.dim() - 1)
            idx_shape[dim] = idx_size
            idx_expanded = idx.view(*idx_shape)
            idx_expanded = idx_expanded.expand_as(val)

            t.scatter_(dim, idx_expanded, val)
            return t
        else:
            raise NotImplementedError(f"idx.dim() == {idx.dim()} not supported")

    else:
        if idx.dim() == 1:
            return t.index_copy_(dim, idx, val)

        assert idx.dim() == 2, f"multiple batch dims not yet {idx.shape=}"
        assert dim != 0, f"cannot index batch dim {dim=}"
        batch_size, idx_size = idx.shape
        assert batch_size == t.size(0)
        assert batch_size == val.size(0)

        # if we can view the batch and indexed dimensions together, we could
        # do index trickery. This is, sadly, not the case for kvcache so we
        # fall back to for loop
        for i in range(batch_size):
            unbatched_dim = dim if dim < 0 else dim - 1
            t[i].index_copy_(unbatched_dim, idx[i], val[i])
        return t


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 2,  # 默认为 2 (即在 T 和 head_size 之间插入 N)
) -> torch.Tensor:
    """
    Applies RoPE transform to `x`.

    Args:
        x: Input tensor.
        cos: Cached cosines, shape `(B, T, head_size)` or `(1, T, head_size)`
        sin: Cached sines, shape `(B, T, head_size)` or `(1, T, head_size)`
        unsqueeze_dim: The dimension index to insert a size-1 dimension for broadcasting
                       the attention heads.

    Returns:
        Encoded tensor with the same shape as `x`.
    """
    if cos.dim() != 3:
        raise ValueError(f"cos must be three-dimensional, but shape is {cos.shape}")
    if cos.shape != sin.shape:
        raise ValueError(
            f"cos, sin must have same shape, but cos.shape={cos.shape}, sin.shape={sin.shape}"
        )

    head_size_half = x.size(-1) // 2
    x1 = x[..., :head_size_half]
    x2 = x[..., head_size_half:]
    rotated = torch.cat((-x2, x1), dim=-1)

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


def do_softcapping(x: torch.Tensor, thresh: float) -> torch.Tensor:
    return torch.tanh(x / thresh) * thresh


class KVCache(nn.Module):
    """
    Buffers `k`, `v` have shape
    `(batch_size, n_query_groups, max_seq_length, head_size)`.
    """

    def __init__(
        self,
        k_shape: Tuple[int, int, int, int],
        v_shape: Tuple[int, int, int, int],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        is_sliding_window: bool = False,
        sliding_window_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "k", torch.zeros(k_shape, device=device, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v", torch.zeros(v_shape, device=device, dtype=dtype), persistent=False
        )
        self.is_sliding_window = is_sliding_window
        self.sliding_window_size = sliding_window_size
        self.max_cache_len = k_shape[2]

    def forward(
        self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Writes new values `k` and `v` into the cache at the positions specified
        by `input_pos` along the sequence dimension (`max_seq_length`). The batch
        size of `k` and `v` (`bs`) must be smaller or equal to `KVCache` batch
        size. Returns the full buffers, adjusted to the batch size `bs`.

        Args:
            input_pos: Position index, `(bs, T)` or `(T,)`
            k: New values, `(bs, n_query_groups, T, head_size)`
            v: New values, `(bs, n_query_groups, T, head_size)`

        Returns:
            k_full, v_full, `(bs, n_query_groups, max_seq_length, head_size)`

        """
        # move the buffer to the activation dtype for when AMP is used
        if self.k.dtype != k.dtype:
            self.k = self.k.to(k.dtype)
        if self.v.dtype != v.dtype:
            self.v = self.v.to(v.dtype)
        # update the cache
        bs = k.size(0)
        if self.is_sliding_window:
            # Circular buffer for sliding window
            cache_positions = input_pos % self.max_cache_len
            k = batched_index_copy_(self.k[:bs, ...], -2, cache_positions, k)
            v = batched_index_copy_(self.v[:bs, ...], -2, cache_positions, v)

            max_pos = input_pos.max().item()
            if max_pos < self.max_cache_len:
                k = k[:, :, : max_pos + 1, :]
                v = v[:, :, : max_pos + 1, :]
        else:
            # Standard KV cache (global attention)
            k = batched_index_copy_(self.k[:bs, ...], -2, input_pos, k)
            v = batched_index_copy_(self.v[:bs, ...], -2, input_pos, v)

        return k, v

    def reset_parameters(self) -> None:
        torch.nn.init.zeros_(self.k)
        torch.nn.init.zeros_(self.v)


def build_mask_cache(
    max_seq_length: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    ones = torch.ones((max_seq_length, max_seq_length), device=device, dtype=torch.bool)
    return torch.tril(ones).unsqueeze(0).unsqueeze(0)


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(
        self, size: int, dim: int = -1, eps: float = 1e-6, add_unit_offset: bool = False
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim
        self.add_unit_offset = add_unit_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        # NOTE: the original RMSNorm paper implementation is not equivalent
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        weight = (1 + self.weight) if self.add_unit_offset else self.weight
        return (x_normed * weight.float()).to(dtype=dtype)

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)
