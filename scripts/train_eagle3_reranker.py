#!/usr/bin/env python3
"""Train the experimental in-loop EAGLE3 reranker.

RERANKER PIPELINE: this script trains a low-rank scorer that reranks EAGLE
branch-token candidates at every draft timestep. It does not modify the
original EAGLE3 training flow or draft checkpoint.
"""

import argparse
import json
import math
import os
import time
from argparse import Namespace
from typing import Dict

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from specforge.core.eagle3_reranker_pipeline import (
    LowRankTokenReranker,
    OnlineEagle3RerankerModel,
)
from specforge.distributed import destroy_distributed, init_distributed
from specforge.utils import print_args_with_dots, print_on_rank0, print_with_rank

from train_eagle3 import (
    build_dataloaders,
    build_draft_model,
    build_parser as build_eagle3_parser,
    build_target_model,
    get_dp_data_shard_from_tp,
    print_cuda_memory_debug,
    sanity_check,
)


def build_parser() -> argparse.ArgumentParser:
    parser = build_eagle3_parser()
    parser.description = "Train an online in-loop EAGLE3 reranker"
    group = parser.add_argument_group("eagle3 reranker")
    group.add_argument(
        "--eagle-draft-model-path",
        type=str,
        default=None,
        help="Frozen trained EAGLE3 draft checkpoint. Used as --ckpt-dir if set.",
    )
    group.add_argument(
        "--vocab-mapping-path",
        type=str,
        default=None,
        help="Optional exact vocab mapping for the frozen EAGLE draft.",
    )
    group.add_argument(
        "--load-generated-vocab-mapping",
        action="store_true",
        help=(
            "Load mapping generated from training data. Off by default because "
            "the frozen EAGLE checkpoint should normally keep its own mapping."
        ),
    )
    group.add_argument("--reranker-top-k", type=int, default=20)
    group.add_argument("--reranker-rank-dim", type=int, default=256)
    group.add_argument(
        "--no-reranker-normalize",
        action="store_true",
        help="Disable L2 normalization in the low-rank dot-product scorer.",
    )
    group.add_argument(
        "--combine-draft-logits-for-loss",
        action="store_true",
        help="Train the reranker score added to draft log-prob instead of alone.",
    )
    return parser


def validate_args(args: Namespace) -> None:
    if args.eagle_draft_model_path is not None:
        args.ckpt_dir = args.eagle_draft_model_path
    if args.ckpt_dir is None:
        raise ValueError("Pass --eagle-draft-model-path or --ckpt-dir.")
    if args.train_hidden_states_path is not None:
        raise ValueError("Reranker training is online-only; omit hidden states path.")
    if args.is_vlm:
        raise ValueError("Reranker training currently supports text-only models.")
    if args.reranker_top_k < 1:
        raise ValueError("--reranker-top-k must be >= 1")


def maybe_wrap_ddp(module: torch.nn.Module) -> torch.nn.Module:
    if dist.get_world_size() == 1:
        return module
    return DDP(module, device_ids=[torch.cuda.current_device()])


def unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module


def reduce_metrics(loss: torch.Tensor, metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    names = ["loss", *metrics.keys()]
    packed = torch.stack(
        [loss.detach().float(), *[metrics[name].detach().float() for name in metrics]]
    )
    dist.all_reduce(packed, op=dist.ReduceOp.AVG)
    return {name: packed[idx].item() for idx, name in enumerate(names)}


def build_online_batch(args: Namespace, data: dict, target_model):
    eagle3_data = target_model.generate_eagle3_data(
        input_ids=data["input_ids"].cuda(),
        attention_mask=data["attention_mask"].cuda(),
        loss_mask=data["loss_mask"].cuda(),
        shard_returns=args.shard_target_output,
    )
    return {
        "input_ids": get_dp_data_shard_from_tp(
            eagle3_data.input_ids, args.shard_target_output
        ),
        "attention_mask": get_dp_data_shard_from_tp(
            eagle3_data.attention_mask, args.shard_target_output
        ),
        "loss_mask": get_dp_data_shard_from_tp(
            eagle3_data.loss_mask, args.shard_target_output
        ),
        "target": get_dp_data_shard_from_tp(
            eagle3_data.target, args.shard_target_output
        ),
        "hidden_states": get_dp_data_shard_from_tp(
            eagle3_data.hidden_states, args.shard_target_output
        ),
        "position_ids": data["position_ids"].cuda() if "position_ids" in data else None,
    }


def save_checkpoint(
    *,
    args: Namespace,
    reranker_model: OnlineEagle3RerankerModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
) -> None:
    if dist.get_rank() != 0:
        return
    checkpoint_dir = os.path.join(args.output_dir, f"reranker_epoch_{epoch}_step_{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    reranker = unwrap_module(reranker_model.reranker)
    torch.save(reranker.state_dict(), os.path.join(checkpoint_dir, "reranker.pt"))
    with open(os.path.join(checkpoint_dir, "reranker_config.json"), "w") as f:
        json.dump(reranker.config_dict(), f, indent=2)
    torch.save(
        {
            "epoch": epoch,
            "global_step": step,
            "args": vars(args),
            "optimizer": optimizer.state_dict(),
        },
        os.path.join(checkpoint_dir, "training_state.pt"),
    )
    print_on_rank0(f"Saved reranker checkpoint to {checkpoint_dir}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    set_seed(args.seed)

    init_distributed(
        timeout=args.dist_timeout,
        tp_size=args.tp_size,
        sp_ring_size=args.sp_ring_size,
        sp_ulysses_size=args.sp_ulysses_size,
    )
    sanity_check(args)
    print_args_with_dots(args)
    print_with_rank("Initialized distributed environment")

    print_cuda_memory_debug("before build_draft_model")
    draft_model_config, draft_model, _, _ = build_draft_model(args)
    print_cuda_memory_debug("after build_draft_model")

    print_cuda_memory_debug("before build_target_model")
    target_model, processor = build_target_model(args, draft_model_config, is_online=True)
    print_cuda_memory_debug("after build_target_model")

    print_cuda_memory_debug("before build_dataloaders")
    train_dataloader, generated_vocab_mapping_path, _ = build_dataloaders(
        args, draft_model_config, processor
    )
    print_cuda_memory_debug("after build_dataloaders")

    if args.vocab_mapping_path is not None:
        draft_model.load_vocab_mapping(args.vocab_mapping_path)
        print_with_rank(f"Loaded explicit vocab mapping: {args.vocab_mapping_path}")
    elif args.load_generated_vocab_mapping:
        draft_model.load_vocab_mapping(generated_vocab_mapping_path)
        print_with_rank(f"Loaded generated vocab mapping: {generated_vocab_mapping_path}")
    else:
        print_with_rank("Keeping vocab mapping from frozen EAGLE checkpoint")

    reranker = LowRankTokenReranker(
        hidden_size=draft_model_config.hidden_size,
        token_hidden_size=draft_model.lm_head.weight.shape[-1],
        rank_dim=args.reranker_rank_dim,
        normalize=not args.no_reranker_normalize,
    ).cuda()
    reranker_model = OnlineEagle3RerankerModel(
        draft_model=draft_model,
        reranker=reranker,
        length=args.ttt_length,
        top_k=args.reranker_top_k,
        attention_backend=args.attention_backend,
        combine_draft_logits_for_loss=args.combine_draft_logits_for_loss,
    ).cuda()
    reranker_model.reranker = maybe_wrap_ddp(reranker_model.reranker)
    reranker_model.train()
    reranker_model.draft_model.eval()

    optimizer = torch.optim.AdamW(
        reranker_model.reranker.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    if args.total_steps is None:
        args.total_steps = args.num_epochs * math.ceil(
            len(train_dataloader) / args.draft_accumulation_steps
        )

    global_step = 0
    last_time = time.time()
    print_on_rank0(
        f"Starting reranker training: top_k={args.reranker_top_k}, "
        f"rank_dim={args.reranker_rank_dim}"
    )

    for epoch in range(args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch + 1)
        progress = (
            tqdm(train_dataloader, desc=f"Reranker Epoch {epoch}", leave=True)
            if dist.get_rank() == 0
            else train_dataloader
        )
        for data in progress:
            global_step += 1
            batch = build_online_batch(args, data, target_model)
            output = reranker_model(**batch)
            loss = output.loss / args.draft_accumulation_steps
            loss.backward()

            if global_step % args.draft_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    reranker_model.reranker.parameters(), args.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % (args.log_interval * args.draft_accumulation_steps) == 0:
                reduced = reduce_metrics(output.loss, output.metrics)
                if dist.get_rank() == 0:
                    elapsed = time.time() - last_time
                    last_time = time.time()
                    progress.set_postfix(
                        {
                            "loss": f"{reduced['loss']:.4f}",
                            "r@1": f"{reduced['restricted_top1']:.3f}",
                            "H": f"{reduced['teacher_entropy']:.3f}",
                            "time": f"{elapsed:.2f}s",
                        }
                    )

            if global_step % (args.save_interval * args.draft_accumulation_steps) == 0:
                save_checkpoint(
                    args=args,
                    reranker_model=reranker_model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=global_step,
                )

            if args.max_num_steps is not None and global_step >= args.max_num_steps:
                break

        if args.max_num_steps is not None and global_step >= args.max_num_steps:
            break

    save_checkpoint(
        args=args,
        reranker_model=reranker_model,
        optimizer=optimizer,
        epoch=epoch,
        step=global_step,
    )
    destroy_distributed()


if __name__ == "__main__":
    main()
