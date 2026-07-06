#!/usr/bin/env python3
"""Train the TEST TIME EAGLE EXP hidden adapter.

TEST TIME EAGLE EXP: this script freezes an already-trained EAGLE3 draft model
and trains only a gated residual hidden adapter before the draft LM head. The
original EAGLE3 training script and checkpoint format are not modified.
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

from specforge.core.eagle3_hidden_adapter_pipeline import (
    GatedResidualHiddenAdapter,
    OnlineEagle3HiddenAdapterModel,
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


# TEST TIME EAGLE EXP: extend the normal EAGLE3 parser instead of changing it.
def build_parser() -> argparse.ArgumentParser:
    parser = build_eagle3_parser()
    parser.description = "Train a frozen-EAGLE3 gated hidden adapter"
    group = parser.add_argument_group("test time eagle exp hidden adapter")
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
    group.add_argument(
        "--hidden-adapter-bottleneck-size",
        type=int,
        default=0,
        help="0 uses a single Linear(hidden, hidden); >0 uses hidden->bottleneck->hidden.",
    )
    group.add_argument(
        "--hidden-adapter-gate-type",
        choices=["scalar", "channel"],
        default="scalar",
    )
    group.add_argument("--hidden-adapter-gate-init", type=float, default=0.0)
    group.add_argument("--hidden-adapter-dropout", type=float, default=0.0)
    group.add_argument("--hidden-kl-weight", type=float, default=0.8)
    group.add_argument("--hidden-mse-weight", type=float, default=0.2)
    group.add_argument(
        "--hidden-loss-type",
        choices=["mse", "norm_mse", "cosine"],
        default="mse",
        help="Auxiliary hidden alignment loss.",
    )
    group.add_argument(
        "--feed-corrected-hidden",
        action="store_true",
        help=(
            "Use corrected hidden states as the next TTT state inside the window. "
            "Default is safer LM-head-side correction only."
        ),
    )
    return parser


# TEST TIME EAGLE EXP: validate the side experiment without changing base checks.
def validate_args(args: Namespace) -> None:
    if args.eagle_draft_model_path is not None:
        args.ckpt_dir = args.eagle_draft_model_path
    if args.ckpt_dir is None:
        raise ValueError("Pass --eagle-draft-model-path or --ckpt-dir.")
    if args.train_hidden_states_path is not None:
        raise ValueError("Hidden-adapter training is online-only; omit hidden states path.")
    if args.is_vlm:
        raise ValueError("Hidden-adapter training currently supports text-only models.")
    if args.attention_backend == "usp":
        raise ValueError("Hidden-adapter training does not support USP yet.")
    if args.hidden_kl_weight < 0 or args.hidden_mse_weight < 0:
        raise ValueError("Loss weights must be non-negative.")


# TEST TIME EAGLE EXP: DDP wraps only the adapter, not the frozen EAGLE draft.
def maybe_wrap_ddp(module: torch.nn.Module) -> torch.nn.Module:
    if dist.get_world_size() == 1:
        return module
    return DDP(module, device_ids=[torch.cuda.current_device()])


# TEST TIME EAGLE EXP: unwrap adapter before saving its state dict.
def unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module


# TEST TIME EAGLE EXP: average scalar metrics across ranks for progress logs.
def reduce_metrics(
    loss: torch.Tensor, metrics: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    names = ["loss", *metrics.keys()]
    packed = torch.stack(
        [loss.detach().float(), *[metrics[name].detach().float() for name in metrics]]
    )
    dist.all_reduce(packed, op=dist.ReduceOp.AVG)
    return {name: packed[idx].item() for idx, name in enumerate(names)}


# TEST TIME EAGLE EXP: online batch builder keeps target hidden/logit collection
# identical to normal EAGLE3 online training.
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


# TEST TIME EAGLE EXP: save adapter-only checkpoints so the original EAGLE draft
# checkpoint remains untouched.
def save_checkpoint(
    *,
    args: Namespace,
    hidden_adapter_model: OnlineEagle3HiddenAdapterModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
) -> None:
    if dist.get_rank() != 0:
        return
    checkpoint_dir = os.path.join(
        args.output_dir, f"hidden_adapter_epoch_{epoch}_step_{step}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    hidden_adapter = unwrap_module(hidden_adapter_model.hidden_adapter)
    torch.save(
        hidden_adapter.state_dict(),
        os.path.join(checkpoint_dir, "hidden_adapter.pt"),
    )
    with open(os.path.join(checkpoint_dir, "hidden_adapter_config.json"), "w") as f:
        json.dump(hidden_adapter.config_dict(), f, indent=2)
    torch.save(
        {
            "epoch": epoch,
            "global_step": step,
            "args": vars(args),
            "optimizer": optimizer.state_dict(),
        },
        os.path.join(checkpoint_dir, "training_state.pt"),
    )
    print_on_rank0(f"Saved hidden adapter checkpoint to {checkpoint_dir}")


# TEST TIME EAGLE EXP: train only the gated adapter on frozen EAGLE states.
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

    bottleneck_size = (
        None
        if args.hidden_adapter_bottleneck_size <= 0
        else args.hidden_adapter_bottleneck_size
    )
    hidden_adapter = GatedResidualHiddenAdapter(
        hidden_size=draft_model_config.hidden_size,
        bottleneck_size=bottleneck_size,
        gate_type=args.hidden_adapter_gate_type,
        gate_init=args.hidden_adapter_gate_init,
        dropout=args.hidden_adapter_dropout,
    ).cuda()
    hidden_adapter_model = OnlineEagle3HiddenAdapterModel(
        draft_model=draft_model,
        hidden_adapter=hidden_adapter,
        length=args.ttt_length,
        attention_backend=args.attention_backend,
        kl_weight=args.hidden_kl_weight,
        hidden_weight=args.hidden_mse_weight,
        hidden_loss_type=args.hidden_loss_type,
        feed_corrected_hidden=args.feed_corrected_hidden,
    ).cuda()
    hidden_adapter_model.hidden_adapter = maybe_wrap_ddp(
        hidden_adapter_model.hidden_adapter
    )
    hidden_adapter_model.train()
    hidden_adapter_model.draft_model.eval()

    optimizer = torch.optim.AdamW(
        hidden_adapter_model.hidden_adapter.parameters(),
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
        "Starting TEST TIME EAGLE EXP hidden-adapter training: "
        f"ttt_length={args.ttt_length}, kl_weight={args.hidden_kl_weight}, "
        f"hidden_weight={args.hidden_mse_weight}, "
        f"feed_corrected_hidden={args.feed_corrected_hidden}"
    )

    for epoch in range(args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch + 1)
        progress = (
            tqdm(train_dataloader, desc=f"HiddenAdapter Epoch {epoch}", leave=True)
            if dist.get_rank() == 0
            else train_dataloader
        )
        for data in progress:
            global_step += 1
            batch = build_online_batch(args, data, target_model)
            output = hidden_adapter_model(**batch)
            loss = output.loss / args.draft_accumulation_steps
            loss.backward()

            if global_step % args.draft_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    hidden_adapter_model.hidden_adapter.parameters(),
                    args.max_grad_norm,
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
                            "kl": f"{reduced['kl_loss']:.4f}",
                            "hid": f"{reduced['hidden_loss']:.4f}",
                            "acc": f"{reduced['acc']:.3f}",
                            "ar": f"{reduced['acceptance_rate']:.3f}",
                            "gate": f"{reduced['gate_abs_mean']:.4f}",
                            "time": f"{elapsed:.2f}s",
                        }
                    )

            if global_step % (args.save_interval * args.draft_accumulation_steps) == 0:
                save_checkpoint(
                    args=args,
                    hidden_adapter_model=hidden_adapter_model,
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
        hidden_adapter_model=hidden_adapter_model,
        optimizer=optimizer,
        epoch=epoch,
        step=global_step,
    )
    destroy_distributed()


if __name__ == "__main__":
    main()
