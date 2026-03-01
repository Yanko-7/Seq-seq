# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Literal, Optional, Type, Union

import yaml
from typing_extensions import Self


def find_multiple(n: int, k: int) -> int:
    """Utility function for finding the nearest value to n which is a multiple of k.

    NOTE: We define this function in this module rather than `litgpt.utils` so that users can import
    this file to do configuration manipulations in Python environments which do not include all the dependencies
    demanded by `litgpt.utils`.
    """
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class Config:
    name: str = ""
    hf_config: dict = field(default_factory=dict)
    # General size parameters
    block_size: int = 4096
    n_layer: int = 16
    n_embd: int = 4096
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: Optional[int] = None
    # Transformer block (structure, normalizations)
    norm_class_name: Literal["LayerNorm", "RMSNorm"] = "LayerNorm"
    norm_eps: float = 1e-5
    norm_qk: bool = False
    norm_qk_type: Literal["default", "olmo2"] = "default"
    post_attention_norm: bool = False
    post_mlp_norm: bool = False
    parallel_residual: bool = True
    shared_attention_norm: bool = False
    # Transformer block (self-attention)
    n_head: int = 32
    head_size: Optional[int] = None
    attention_dropout: float = 0.1
    # to use multi-head attention (MHA), set this to `n_head` (default)
    # to use multi-query attention (MQA), set this to 1
    # to use grouped-query attention (GQA), set this to a value in between
    # Example with `n_head=4`
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
    n_query_groups: Optional[int] = None
    attn_bias: bool = False
    attention_scores_scalar: Optional[int] = None
    # If `sliding_window_size` is given, sliding window attention with this
    # size is used in layers where `sliding_window_indices` has a 1. The
    # default is all 1, so that sliding window attention is used in all
    # layers. If `len(sliding_window_indices) > n_layer`, we only use the
    # initial part.
    sliding_window_size: Optional[int] = None
    sliding_window_indices: Optional[List[int]] = None
    # if `attention_logit_softcapping` is used, cannot use optimized
    # `torch.nn.functional.scaled_dot_product_attention` (which implements
    # Flash attention), may result in higher memory and runtime footprint.
    attention_logit_softcapping: Optional[float] = None
    # Rotary position embedding (RoPE)
    rope_base: int = 10000
    rotary_percentage: float = 0.25
    rope_condense_ratio: int = 1
    rope_adjustments: Optional[dict] = None
    # Transformer block (MLP)
    intermediate_size: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    bias: bool = True
    mlp_class_name: Literal["GptNeoxMLP", "LLaMAMLP", "GemmaMLP", "LLaMAMoE"] = (
        "LLaMAMLP"
    )
    gelu_approximate: str = "none"
    n_expert: int = 0
    n_shared_expert: Optional[int] = None
    n_expert_groups: Optional[int] = None
    n_topk_groups: Optional[int] = None
    n_topk_scores_per_group: Optional[int] = None
    n_expert_per_token: int = 0
    first_k_dense_replace: Optional[int] = None
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = False
    # GPT before/after blocks
    scale_embeddings: bool = False
    lm_head_bias: bool = False
    final_logit_softcapping: Optional[float] = None
    norm_1: bool = True
    norm_2: bool = True
    latent_attention: Optional[dict] = None
    # The base period of the RoPE embeddings for local attention.
    # If not provided, `rope_base` will be used for both local and global attention.
    rope_local_base_freq: Optional[float] = None
    # If provided, must have `>= n_layer` entries, either 0 or 1. For 0,
    # `rope_base` is used, for 1 `rope_local_base_freq` is used. If
    # `len(rope_indices) > n_layer`, we only use the initial part.
    rope_indices: Optional[List[int]] = None

    def __post_init__(self):
        if not self.name:
            self.name = self.hf_config.get("name", self.name)

        if self.head_size is None:
            assert self.n_embd % self.n_head == 0
            self.head_size = self.n_embd // self.n_head

        # vocab size should be a power of 2 to be optimal on hardware. compute the closest value
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(
                self.vocab_size, self.padding_multiple
            )
        else:
            # vocab size shouldn't be larger than padded vocab size
            self.vocab_size = min(self.vocab_size, self.padded_vocab_size)

        # compute the number of query groups
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head

        # compute the intermediate size for MLP if not set
        if self.intermediate_size is None:
            if self.mlp_class_name == "LLaMAMLP":
                raise ValueError(
                    f"The config {self.name!r}, needs to set the `intermediate_size`"
                )
            self.intermediate_size = 4 * self.n_embd

        self.rope_n_elem = int(self.rotary_percentage * self.head_size)

        if self.sliding_window_size is not None:
            self.sliding_window_indices = check_indicator_and_length(
                self.sliding_window_indices,
                name="sliding_window_indices",
                required_length=self.n_layer,
            )

        if self.rope_local_base_freq is not None:
            self.rope_indices = check_indicator_and_length(
                self.rope_indices,
                name="rope_indices",
                required_length=self.n_layer,
            )

        if self.latent_attention is not None:
            self.q_lora_rank = self.latent_attention.get("q_lora_rank")
            self.kv_lora_rank = self.latent_attention.get("kv_lora_rank")
            self.qk_rope_head_dim = self.latent_attention.get("qk_rope_head_dim")
            self.qk_nope_head_dim = self.latent_attention.get("qk_nope_head_dim")
            self.v_head_dim = self.latent_attention.get("v_head_dim")
            assert (
                self.q_lora_rank
                and self.kv_lora_rank
                and self.qk_rope_head_dim
                and self.qk_nope_head_dim
                and self.v_head_dim
            ) is not None
            assert self.n_head == self.n_query_groups, (
                "Latent attention does not support MQA/GQA"
            )
            self.qk_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
            self.rope_n_elem = self.qk_rope_head_dim
        if self.first_k_dense_replace is not None:
            assert self.mlp_class_name == "LLaMAMoE"
        if self.n_expert_groups is not None:
            assert (
                self.n_expert % self.n_expert_groups == 0 and self.n_expert_groups > 1
            )
            assert self.n_topk_groups is not None
            experts_per_group = self.n_expert // self.n_expert_groups
            assert (
                self.n_topk_scores_per_group is not None
                and self.n_topk_scores_per_group <= experts_per_group
            )

    @classmethod
    def from_file(cls, path: Union[str, Path], **kwargs: Any) -> Self:
        with open(path, encoding="utf-8") as fp:
            file_kwargs = yaml.safe_load(fp)
            if file_kwargs is None:
                raise ValueError(f"{path} is empty which is likely unexpected.")
        file_kwargs.update(kwargs)
        return cls(**file_kwargs)

    @property
    def mlp_class(self) -> Type:
        # `self.mlp_class_name` cannot be the type to keep the config serializable
        import litgpt.model

        return getattr(litgpt.model, self.mlp_class_name)

    @property
    def norm_class(self) -> Type:
        # `self.norm_class_name` cannot be the type to keep the config serializable

        from functools import partial

        import torch  # Torch import is lazy to make config loading faster

        if self.norm_class_name == "RMSNorm":
            from litgpt.model import RMSNorm

            return partial(RMSNorm, add_unit_offset="Gemma" in self.name)

        if self.norm_class_name == "LayerNorm" and "OLMo" in self.name:
            # this makes it equivalent to `torch.nn.functional.layer_norm`
            # that is used by OLMo
            # Table 5 caption in the OLMo paper shows this - https://aclanthology.org/2024.acl-long.841
            return partial(torch.nn.LayerNorm, elementwise_affine=False)

        return getattr(torch.nn, self.norm_class_name)


def check_indicator_and_length(
    params: Optional[List[int]],
    name: str,
    required_length: int,
    use_initial_part: bool = True,
    def_val: int = 1,
) -> List[int]:
    if params is None:
        return [def_val] * required_length
    if len(params) != required_length:
        if use_initial_part and len(params) > required_length:
            params = params[:required_length]
        else:
            raise ValueError(f"{name} = {params}, must have length {required_length}")
    if not set(params).issubset({0, 1}):
        raise ValueError(f"{name} = {params}, must only contain 0 and 1")
    return params
