import argparse
from dataclasses import dataclass, field
import math
import os
from collections import Counter
from typing import Dict, List
import numpy as np
import torch
import torch.distributed as dist
import wandb
import gc
from src.checkcpoint_mang import save_checkpoint
from src.common import (
    compute_init,
    compute_cleanup,
    print0,
    get_base_dir,
    DummyWandb,
    autodetect_device_type,
)
from src.engine import Engine
from src.gptv3 import GPTConfig
from src.model import ForgeTrace
from src.tokenizer import Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Reinforcement learning on GSM8K")
    parser.add_argument("--run", type=str, default="wandb", help="wandb run name")
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=16,
        help="Micro-batch size per forward pass",
    )
    parser.add_argument(
        "--rollout-size",
        type=int,
        default=256,
        help="Number of samples generated per rollout",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--embedding-lr", type=float, default=0.2)
    parser.add_argument("--unembedding-lr", type=float, default=0.004)
    parser.add_argument("--matrix-lr", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--init-lr-frac", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=60)
    parser.add_argument("--save-every", type=int, default=60)
    parser.add_argument("--total_steps", type=int, default=3000)
    parser.add_argument("--device_type", type=str, default="")
    parser.add_argument("--model_tag", type=str, default="")
    parser.add_argument("--gen-max-batch-size", type=int, default=128)
    return parser.parse_args()


def reward_fn(seq, tokenizer):
    status, _, _ = tokenizer.validate_tokens(seq)
    # return 1.0 if status == TokenValidationStatus.SUCCESS else -1.0
    raise NotImplementedError("Reward function not implemented")


@dataclass
class Episode:
    seq_len: int
    input_ids: np.ndarray
    target_ids: np.ndarray
    reward: float
    reward_info: Dict[str, float] = field(default_factory=dict)


def compute_advantages(episodes: List[Episode]) -> List[Episode]:
    rewards = np.array([ep.reward for ep in episodes], dtype=np.float32)
    normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    for ep, adv in zip(episodes, normalized_rewards):
        ep.reward_info["advantage"] = adv.item()

    return episodes


@torch.no_grad()
def rollout(
    model, tokenizer, rollout_size, reward_fn, args, device, rank
) -> List[Episode]:
    engine = Engine(model, tokenizer)
    bos, pad = tokenizer.get_bos_token_id(), tokenizer.get_pad_token_id()

    results = []
    num_batches = math.ceil(rollout_size / args.gen_max_batch_size)

    for i in range(num_batches):
        seed = hash((i, rank)) & 0x7FFFFFFF
        seqs, masks = engine.generate_batch(
            [bos], num_samples=args.gen_max_batch_size, seed=seed
        )

        max_len = max(map(len, seqs))

        for seq, mask in zip(seqs, masks):
            seq_padded = seq + [pad] * (max_len - len(seq))
            mask_padded = mask + [0] * (max_len - len(mask))

            input_ids = np.array(seq_padded[:-1], dtype=np.int64)
            target_ids = np.array(seq_padded[1:], dtype=np.int64)
            mask_arr = np.array(mask_padded[1:], dtype=bool)

            target_ids[~mask_arr] = -1

            results.append(
                Episode(
                    seq_len=len(seq),
                    input_ids=input_ids,
                    target_ids=target_ids,
                    reward=float(reward_fn(seq, tokenizer)),
                )
            )

    del engine
    gc.collect()
    torch.cuda.empty_cache()

    return results


@torch.no_grad()
def evaluate(engine, tokenizer, ddp, batch_size=160, max_tokens=4096):
    bos = [tokenizer.get_bos_token_id()]
    seqs, _ = engine.generate_batch(
        bos, num_samples=batch_size, max_tokens=max_tokens, temperature=1.0
    )

    local_stats = Counter(tokenizer.validate_tokens(seq)[0].name for seq in seqs)
    local_total = len(seqs)

    if ddp:
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, (local_stats, local_total))
        return sum((g[0] for g in gathered), Counter()), sum(g[1] for g in gathered)
    return local_stats, local_total


def main():
    args = parse_args()

    # Init compute
    device_type = autodetect_device_type() if not args.device_type else args.device_type
    ddp, rank, local_rank, _, device = compute_init(device_type)
    torch.cuda.set_device(local_rank)
    is_master = rank == 0

    # Init Wandb
    wandb_run = (
        DummyWandb()
        if args.run == "dummy" or not is_master
        else wandb.init(project="nanochat-rl", name=args.run, config=vars(args))
    )

    # Init Model & Engine
    torch.serialization.add_safe_globals([GPTConfig])
    model = ForgeTrace.load_from_checkpoint(
        "checkpoint/nanochat/latest-step060000.ckpt", map_location="cpu"
    )
    model = model.to(device=device, dtype=torch.bfloat16)
    tokenizer = Tokenizer(checkpoint_dir="weights")
    engine = Engine(model, tokenizer)

    # Optimizer Setup
    optimizer = model.model.setup_optimizer(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        weight_decay=args.weight_decay,
    )
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"] = group["lr"] * args.init_lr_frac

    # Training Loop
    for step in range(args.total_steps):
        model.eval()

        # Evaluation
        if step % args.eval_every == 0:
            stats, total = evaluate(engine, tokenizer, ddp)
            if is_master:
                print0(
                    f"\n📊 校验统计报告 (Step: {step}, 总样本数: {total}):\n{'-' * 40}"
                )
                val_logs = {"step": step}
                for name, count in stats.most_common():
                    pct = (count / total) * 100 if total > 0 else 0
                    print0(f"{name:<25}: {count:>5} 占比 {pct:>5.2f}%")
                    val_logs[f"val/{name}_pct"] = pct
                wandb_run.log(val_logs)

        # Rollout & Training
        model.train()
        episodes = rollout(
            model, tokenizer, args.rollout_size, reward_fn, args, device, rank
        )
        episodes = compute_advantages(episodes)

        num_passes = len(episodes) // args.train_batch_size  # 反向传播次数

        for i in range(num_passes):
            b0, b1 = i * args.train_batch_size, (i + 1) * args.train_batch_size
            inputs = [ep.input_ids for ep in episodes[b0:b1]]
            targets = [ep.target_ids for ep in episodes[b0:b1]]
            advantages = [ep.reward_info["advantage"] for ep in episodes[b0:b1]]

            logp = -model.model(
                idx=inputs, targets=targets, loss_reduction="none"
            ).view_as(inputs)

            ## 实际上DAPO公式是pi(a|s) / pi_ref(a|s) * A(s,a)
            # 这里reference model就是自己，会退化成 1 * A(s,a)
            # 为了传递梯度，采用logp * A(s,a) ，梯度上是等价的
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()
            num_valid = (targets >= 0).sum().clamp(min=1)

            loss = -(pg_obj / (num_valid * num_passes))
            loss.backward()

        # Metrics aggregation
        mean_reward = torch.tensor([ep.reward for ep in episodes], device=device).mean()
        mean_seq_len = (
            torch.tensor([ep.seq_len for ep in episodes], device=device).float().mean()
        )
        if ddp:
            dist.all_reduce(mean_reward, op=dist.ReduceOp.AVG)
            dist.all_reduce(mean_seq_len, op=dist.ReduceOp.AVG)

        print0(
            f"Step {step}/{args.total_steps} | Loss: {loss.item():.4f} | Avg Reward: {mean_reward.item():.4f} | Avg Seq Len: {mean_seq_len.item():.1f}"
        )
        wandb_run.log(
            {
                "step": step,
                "reward": mean_reward.item(),
                "sequence_length": mean_seq_len.item(),
            }
        )

        # Optimizer step & LR decay
        lrm = 1.0 - (step / args.total_steps)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm

        optimizer.step()
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        wandb_run.log({"step": step, "lrm": lrm})

        # Checkpointing
        if is_master and (
            (step > 0 and step % args.save_every == 0) or step == args.total_steps - 1
        ):
            ckpt_dir = os.path.join(
                get_base_dir(),
                "chatrl_checkpoints",
                args.model_tag or f"d{model.config.n_layer}",
            )
            save_checkpoint(
                ckpt_dir,
                step,
                model.state_dict(),
                None,
                {"model_config": model.config.__dict__},
            )
            print0(f"✅ Saved model checkpoint to {ckpt_dir}")

    wandb_run.finish()
    compute_cleanup()


if __name__ == "__main__":
    main()
