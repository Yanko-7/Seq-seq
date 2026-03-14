"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.

The whole thing is made as efficient as possible.
"""

from collections import Counter
import os

import torch
import torch.nn.functional as F
import re
import logging
import torch.distributed as dist

from scripts.visual import PointGridVisualizer
from tqdm import tqdm

from src.brep_builder import BRepBuilder, save_model_and_image
from src.engine import ColoredFormatter, Engine
from src.tokenizer import TokenValidationStatus


def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(
        ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler])


setup_default_logging()
logger = logging.getLogger(__name__)


def gen_batch(engine, num_samples=64):

    tokens = [engine.tokenizer.get_bos_token_id()]
    validation_stats = Counter()
    total_samples = 0
    for i in tqdm(range(5)):
        result_tokens = engine.generate_batch(
            tokens, num_samples=num_samples, max_tokens=5000, seed=114514 + i
        )[0]

        # 3. 对当前 batch 的每一行进行 check
        for row_tokens in result_tokens:
            status, msg, error_idx = engine.tokenizer.validate_tokens(row_tokens)

            # 使用枚举的 name (如 'SUCCESS', 'UNCLOSED_LOOP') 作为 key 进行统计
            validation_stats[status.name] += 1
            total_samples += 1

    log_lines = [f"\n📊 校验统计报告 (总计 {total_samples} 个样本):", "-" * 40]

    for status_name, count in validation_stats.most_common():
        percentage = (count / total_samples) * 100
        log_lines.append(f"{status_name:<25}: {count:>5} 占比 {percentage:>5.2f}%")

    # 使用换行符拼接并一次性打到 log 里
    logger.info("\n".join(log_lines))


def gen_vis(engine, seed=42):
    tokens = [engine.tokenizer.get_bos_token_id()]
    ids = [engine.tokenizer.get_bos_token_id()]

    for token_column, token_masks in engine.generate(
        tokens, max_tokens=5000, num_samples=1, seed=seed
    ):
        for token, mask in zip(token_column, token_masks):
            ids.append(token)

    ids = torch.tensor(ids, device="cuda")
    status, _, _ = engine.tokenizer.validate_tokens(ids)
    try:
        brepdata = engine.tokenizer.decode(ids)
        viz = PointGridVisualizer()
    except Exception as e:
        print(f"Error occurred: {e}")
        if status == TokenValidationStatus.SUCCESS:
            print("Tokens are valid but decoding failed.")
    else:
        save_path = f"sample_{seed}.png"
        viz.visualize(brepdata, save_path=save_path)
        if status != TokenValidationStatus.SUCCESS:
            print("Tokens are invalid but decoding succeeded.")


if __name__ == "__main__":
    from src.model import ForgeTrace
    from src.tokenizer import Tokenizer
    from src.gptv3 import GPTConfig

    torch.serialization.add_safe_globals([GPTConfig])

    model = ForgeTrace.load_from_checkpoint(
        checkpoint_path="checkpoint/nanochat/latest-step060000.ckpt"
    )
    tokenizer = Tokenizer(checkpoint_dir="weights")
    model = model.to(device="cuda", dtype=torch.bfloat16)
    tokenizer = tokenizer.to(device="cuda")
    engine = Engine(model, tokenizer)
    gen_vis(engine, seed=114514 + 5)
