#!/usr/bin/env python3
"""Train an online EAGLE3 candidate verifier/reranker.

This script is experimental and opt-in. It does not train or alter the EAGLE
draft checkpoint. It freezes a trained EAGLE3 draft model, obtains target
outputs online, and trains a small verifier that scores EAGLE top-k candidates
before final target verification.
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

from specforge import Eagle3CandidateVerifier, OnlineEagle3VerifierModel
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
    parser.description = "Train an online EAGLE3 verifier/reranker"
    group = parser.add_argument_group("eagle3 verifier")
    group.add_argument(
        "--eagle-draft-model-path",
        type=str,
        default=None,
        help=(
            "Path to the trained EAGLE3 draft checkpoint to freeze. "
            "If set, it is used as --ckpt-dir for the draft loader."
        ),
    )
    group.add_argument(
        "--vocab-mapping-path",
        type=str,
        default=None,
        help=(
            "Optional exact vocab mapping to load into the frozen EAGLE draft. "
            "Use this only when the checkpoint does not already carry d2t/t2d."
        ),
    )
    group.add_argument(
        "--load-generated-vocab-mapping",
        action="store_true",
        help=(
            "Load the mapping generated from --train-data-path. Off by default "
            "because a frozen EAGLE checkpoint must keep its original mapping."
        ),
    )
    group.add_argument("--verifier-top-k", type=int, default=8)
    group.add_argument("--verifier-hidden-size", type=int, default=512)
    group.add_argument("--verifier-candidate-embed-dim", type=int, default=128)
    group.add_argument("--verifier-dropout", type=float, default=0.0)
    group.add_argument(
        "--verifier-label-type",
        type=str,
        default="target_top1",
        choices=["target_top1", "target_prob"],
        help=(
            "target_top1 trains for greedy target agreement; target_prob trains "
            "a soft label from the target probability assigned to each candidate."
        ),
    )
    group.add_argument(
        "--verifier-positive-weight",
        type=float,
        default=1.0,
        help="BCE positive-class weight for sparse target_top1 labels.",
    )
    return parser


def validate_verifier_args(args: Namespace) -> None:
    if args.train_hidden_states_path is not None:
        raise ValueError(
            "train_eagle3_verifier.py is online-only. Use --train-data-path and "
            "do not pass --train-hidden-states-path."
        )
    if args.is_vlm:
        raise ValueError("Verifier training currently supports text-only EAGLE3 data.")
    if args.compact_teacher:
        raise ValueError("--compact-teacher is offline-only and not used here.")
    if args.eagle_draft_model_path is not None:
        args.ckpt_dir = args.eagle_draft_model_path
    if args.ckpt_dir is None:
        raise ValueError(
            "Pass --eagle-draft-model-path or --ckpt-dir so the verifier trains "
            "against a frozen trained EAGLE draft."
        )
    if args.verifier_top_k < 1:
        raise ValueError("--verifier-top-k must be >= 1")


def maybe_wrap_ddp(module: torch.nn.Module) -> torch.nn.Module:
    if dist.get_world_size() == 1:
        return module
    return DDP(module, device_ids=[torch.cuda.current_device()])


def unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module


def reduce_metrics(
    loss: torch.Tensor, metrics: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    names = ["loss", *metrics.keys()]
    values = [
        loss.detach().float(),
        *[metrics[name].detach().float() for name in metrics],
    ]
    packed = torch.stack(values)
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
    verifier_model: OnlineEagle3VerifierModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
) -> None:
    if dist.get_rank() != 0:
        return
    checkpoint_dir = os.path.join(args.output_dir, f"verifier_epoch_{epoch}_step_{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    verifier = unwrap_module(verifier_model.verifier)
    torch.save(verifier.state_dict(), os.path.join(checkpoint_dir, "verifier.pt"))
    with open(os.path.join(checkpoint_dir, "verifier_config.json"), "w") as f:
        json.dump(verifier.config_dict(), f, indent=2)
    torch.save(
        {
            "epoch": epoch,
            "global_step": step,
            "args": vars(args),
            "optimizer": optimizer.state_dict(),
        },
        os.path.join(checkpoint_dir, "training_state.pt"),
    )
    print_on_rank0(f"Saved verifier checkpoint to {checkpoint_dir}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_verifier_args(args)
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

    verifier = Eagle3CandidateVerifier(
        draft_hidden_size=draft_model_config.hidden_size,
        draft_vocab_size=draft_model_config.draft_vocab_size,
        max_depth=args.ttt_length,
        candidate_embed_dim=args.verifier_candidate_embed_dim,
        hidden_size=args.verifier_hidden_size,
        dropout=args.verifier_dropout,
    ).cuda()

    verifier_model = OnlineEagle3VerifierModel(
        draft_model=draft_model,
        verifier=verifier,
        length=args.ttt_length,
        top_k=args.verifier_top_k,
        attention_backend=args.attention_backend,
        label_type=args.verifier_label_type,
        positive_weight=args.verifier_positive_weight,
    ).cuda()
    verifier_model.verifier = maybe_wrap_ddp(verifier_model.verifier)
    verifier_model.train()
    verifier_model.draft_model.eval()

    optimizer = torch.optim.AdamW(
        verifier_model.verifier.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    if args.total_steps is None:
        steps_per_epoch = math.ceil(
            len(train_dataloader) / args.draft_accumulation_steps
        )
        args.total_steps = args.num_epochs * steps_per_epoch
    global_step = 0
    last_time = time.time()

    print_on_rank0(
        f"Starting verifier training for {args.num_epochs} epochs, "
        f"top_k={args.verifier_top_k}, label_type={args.verifier_label_type}"
    )

    for epoch in range(args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch + 1)
        progress = (
            tqdm(train_dataloader, desc=f"Verifier Epoch {epoch}", leave=True)
            if dist.get_rank() == 0
            else train_dataloader
        )

        for data in progress:
            global_step += 1
            batch = build_online_batch(args, data, target_model)
            output = verifier_model(**batch)
            loss = output.loss / args.draft_accumulation_steps
            loss.backward()

            if global_step % args.draft_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    verifier_model.verifier.parameters(), args.max_grad_norm
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
                            "v@1": f"{reduced['verifier_recall_at_1']:.3f}",
                            "d@k": f"{reduced['draft_recall_at_k']:.3f}",
                            "time": f"{elapsed:.2f}s",
                        }
                    )

            if global_step % (args.save_interval * args.draft_accumulation_steps) == 0:
                save_checkpoint(
                    args=args,
                    verifier_model=verifier_model,
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
        verifier_model=verifier_model,
        optimizer=optimizer,
        epoch=epoch,
        step=global_step,
    )
    destroy_distributed()


if __name__ == "__main__":
    main()
