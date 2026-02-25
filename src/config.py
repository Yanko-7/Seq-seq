from dataclasses import dataclass


@dataclass
class ForgeTraceConfig:
    block_size: int = 2048  # 序列长度
    vocab_size: int = 32000
    padding_multiple: int = 64
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1024
    n_query_groups: int = 4  # GQA组数 (K/V头数为4)
    rotary_percentage: float = 1.0
    parallel_residual: bool = False
    bias: bool = False
    norm_class_name: str = "RMSNorm"
    mlp_class_name: str = "LLaMAMLP"  #  SwiGLU

    # name="Qwen3-0.6B{}",
    # hf_config=dict(org="Qwen", name="Qwen3-0.6B{}"),
    # block_size=40960,
    # vocab_size=151643,
    # padded_vocab_size=151936,
    # n_layer=28,
    # n_head=16,
    # n_embd=1024,
    # n_query_groups=8,
    # rotary_percentage=1.0,
    # parallel_residual=False,
    # bias=False,
    # norm_class_name="RMSNorm",
    # mlp_class_name="LLaMAMLP",
    # intermediate_size=3072,
    # norm_eps=1e-6,
    # rope_base=1000000,
    # head_size=128,
    # norm_qk=True,
