from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache


Tensor = torch.Tensor


def _padding(tensor: Tensor, left: bool = True) -> Tensor:
    zeropadding = torch.zeros_like(tensor[:, -1:])
    if left:
        return torch.cat((zeropadding, tensor[:, :-1]), dim=1)
    return torch.cat((tensor[:, 1:], zeropadding), dim=1)


@dataclass
class RerankedTreeOutput:
    token_ids: Tensor
    parent_indices: Tensor
    scores: Tensor
    depths: Tensor
    hidden_states: Tensor
    active_indices: Tensor
    tree_mask: Tensor
    tree_position_ids: Tensor
    retrieve_indices: Tensor


@dataclass
class RerankerTrainOutput:
    loss: Tensor
    losses: List[Tensor]
    metrics: Dict[str, Tensor]


# RERANKER PIPELINE: low-rank scorer used during every EAGLE tree expansion step.
class LowRankTokenReranker(nn.Module):
    """Score branch-token pairs over a restricted candidate vocabulary.

    The reranker projects an EAGLE branch hidden state and candidate lm-head rows
    into a small shared space, then scores with a dot product. It is intentionally
    independent from the draft hidden state used for the next EAGLE step: the
    reranker chooses which branch-token pairs survive, while EAGLE child hidden
    states continue generation.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        token_hidden_size: Optional[int] = None,
        rank_dim: int = 256,
        normalize: bool = True,
        logit_scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        token_hidden_size = token_hidden_size or hidden_size
        self.hidden_size = hidden_size
        self.token_hidden_size = token_hidden_size
        self.rank_dim = rank_dim
        self.normalize = normalize

        self.hidden_proj = nn.Linear(hidden_size, rank_dim, bias=False)
        self.token_proj = nn.Linear(token_hidden_size, rank_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init)).log())

    def project_hidden(self, hidden_states: Tensor) -> Tensor:
        projected = self.hidden_proj(hidden_states)
        if self.normalize:
            projected = F.normalize(projected.float(), dim=-1).to(projected.dtype)
        return projected

    def project_token_rows(self, token_rows: Tensor) -> Tensor:
        projected = self.token_proj(token_rows)
        if self.normalize:
            projected = F.normalize(projected.float(), dim=-1).to(projected.dtype)
        return projected

    def score_candidate_rows(
        self,
        branch_hidden: Tensor,
        candidate_rows: Tensor,
    ) -> Tensor:
        """Return scores for branch hidden states and candidate head rows.

        Args:
            branch_hidden: ``[B, H]``.
            candidate_rows: either ``[U, H]`` for a shared unique candidate set
                or ``[B, K, H]`` for per-branch candidate rows.

        Returns:
            ``[B, U]`` or ``[B, K]`` scores.
        """
        scorer_dtype = self.hidden_proj.weight.dtype
        branch_hidden = branch_hidden.to(scorer_dtype)
        candidate_rows = candidate_rows.to(scorer_dtype)

        hidden = self.project_hidden(branch_hidden)
        token = self.project_token_rows(candidate_rows)
        scale = self.logit_scale.exp().to(hidden.dtype)

        if token.dim() == 2:
            return (hidden @ token.T) * scale
        if token.dim() == 3:
            return (hidden[:, None, :] * token).sum(dim=-1) * scale
        raise ValueError(
            "candidate_rows must have shape [U, H] or [B, K, H], "
            f"got {tuple(candidate_rows.shape)}"
        )

    def config_dict(self) -> Dict[str, object]:
        return {
            "hidden_size": self.hidden_size,
            "token_hidden_size": self.token_hidden_size,
            "rank_dim": self.rank_dim,
            "normalize": self.normalize,
            "logit_scale_init": float(self.logit_scale.detach().exp().cpu().item()),
        }


def _make_retrieve_indices(parent_indices: Tensor, depths: Tensor) -> Tensor:
    parent_list = parent_indices.detach().cpu().tolist()
    children = {idx: [] for idx in range(len(parent_list))}
    for idx, parent in enumerate(parent_list):
        if parent >= 0:
            children[parent].append(idx)

    leaves = [idx for idx in range(1, len(parent_list)) if not children[idx]]
    max_depth = int(depths.max().item()) if len(parent_list) > 0 else 0
    paths = torch.full(
        (len(leaves), max_depth + 1),
        -1,
        dtype=torch.long,
        device=parent_indices.device,
    )
    for row, leaf in enumerate(leaves):
        path = []
        cur = leaf
        while cur >= 0:
            path.append(cur)
            cur = parent_list[cur]
        path.reverse()
        paths[row, : len(path)] = torch.tensor(
            path, dtype=torch.long, device=parent_indices.device
        )
    return paths


def _make_tree_mask(parent_indices: Tensor) -> Tensor:
    n_nodes = parent_indices.numel()
    parent_list = parent_indices.detach().cpu().tolist()
    mask = torch.eye(n_nodes, dtype=torch.bool, device=parent_indices.device)
    mask[:, 0] = True
    for idx in range(1, n_nodes):
        parent = parent_list[idx]
        while parent >= 0:
            mask[idx, parent] = True
            parent = parent_list[parent]
    return mask


def _candidate_target_ids(candidate_draft_ids: Tensor, d2t: Optional[Tensor]) -> Tensor:
    if d2t is None:
        return candidate_draft_ids
    return candidate_draft_ids + d2t.to(candidate_draft_ids.device)[candidate_draft_ids]


def restricted_candidate_kl_loss(
    *,
    student_scores: Tensor,
    teacher_logits: Tensor,
    candidate_target_ids: Tensor,
    position_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Train restricted-vocab reranker scores from target logits."""
    teacher_candidate_logits = teacher_logits.gather(-1, candidate_target_ids)
    teacher_probs = F.softmax(teacher_candidate_logits.float(), dim=-1)
    student_log_probs = F.log_softmax(student_scores.float(), dim=-1)
    per_position_loss = -(teacher_probs * student_log_probs).sum(dim=-1)

    if position_mask is None:
        mask = torch.ones_like(per_position_loss)
    else:
        mask = position_mask
        if mask.dim() == per_position_loss.dim() + 1:
            mask = mask.squeeze(-1)
        mask = mask.to(per_position_loss.device, dtype=per_position_loss.dtype)

    loss = (per_position_loss * mask).sum() / mask.sum().clamp_min(1e-6)

    with torch.no_grad():
        teacher_top = teacher_candidate_logits.argmax(dim=-1)
        student_top = student_scores.argmax(dim=-1)
        top1 = ((teacher_top == student_top).float() * mask).sum() / mask.sum().clamp_min(
            1e-6
        )
        entropy = -(teacher_probs * teacher_probs.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = (entropy * mask).sum() / mask.sum().clamp_min(1e-6)
    return loss, {
        "restricted_top1": top1.detach(),
        "teacher_entropy": entropy.detach(),
    }


# RERANKER PIPELINE: inference tree builder that reranks at every draft timestep.
class RerankedEagleTreeBuilder:
    """Build a parent-linked EAGLE draft tree with per-step reranking.

    This class does not call the target model and does not alter the original
    EAGLE pipeline. Callers provide two callbacks:
      * ``draft_logits_fn(hidden)`` returns draft logits for active branches.
      * ``advance_hidden_fn(parent_hidden, token_ids, depth)`` returns EAGLE
        child hidden states for the selected branch-token pairs.
    """

    def __init__(
        self,
        *,
        reranker: LowRankTokenReranker,
        token_row_fn: Callable[[Tensor], Tensor],
        draft_logits_fn: Callable[[Tensor], Tensor],
        advance_hidden_fn: Callable[[Tensor, Tensor, int], Tensor],
        top_k_per_branch: int = 20,
        beam_size: int = 8,
        max_depth: int = 6,
        draft_score_weight: float = 1.0,
        reranker_score_weight: float = 1.0,
    ) -> None:
        self.reranker = reranker
        self.token_row_fn = token_row_fn
        self.draft_logits_fn = draft_logits_fn
        self.advance_hidden_fn = advance_hidden_fn
        self.top_k_per_branch = top_k_per_branch
        self.beam_size = beam_size
        self.max_depth = max_depth
        self.draft_score_weight = draft_score_weight
        self.reranker_score_weight = reranker_score_weight

    @torch.no_grad()
    def build_tree(
        self,
        *,
        root_hidden: Tensor,
        root_token_id: Optional[int] = None,
        root_score: Optional[Tensor] = None,
    ) -> RerankedTreeOutput:
        if root_hidden.dim() != 2:
            raise ValueError(f"root_hidden must be [B, H], got {root_hidden.shape}")

        device = root_hidden.device
        batch_size = root_hidden.shape[0]
        if root_score is None:
            root_score = torch.zeros(batch_size, device=device, dtype=root_hidden.dtype)
        root_tokens = torch.full(
            (batch_size,),
            -1 if root_token_id is None else int(root_token_id),
            dtype=torch.long,
            device=device,
        )

        token_nodes: List[Tensor] = [root_tokens[i] for i in range(batch_size)]
        parent_nodes: List[Tensor] = [
            torch.tensor(-1, dtype=torch.long, device=device) for _ in range(batch_size)
        ]
        score_nodes: List[Tensor] = [root_score[i] for i in range(batch_size)]
        depth_nodes: List[Tensor] = [
            torch.tensor(0, dtype=torch.long, device=device) for _ in range(batch_size)
        ]
        hidden_nodes: List[Tensor] = [root_hidden[i] for i in range(batch_size)]
        active_indices = torch.arange(batch_size, dtype=torch.long, device=device)

        for depth in range(1, self.max_depth + 1):
            active_list = active_indices.detach().cpu().tolist()
            parent_hidden = torch.stack([hidden_nodes[i] for i in active_list])
            parent_scores = torch.stack([score_nodes[i] for i in active_list])
            draft_logits = self.draft_logits_fn(parent_hidden)
            draft_log_probs = F.log_softmax(draft_logits.float(), dim=-1).to(
                parent_hidden.dtype
            )
            top_k = min(self.top_k_per_branch, draft_log_probs.shape[-1])
            topk_log_probs, topk_token_ids = torch.topk(
                draft_log_probs, k=top_k, dim=-1
            )

            unique_token_ids, inverse = torch.unique(
                topk_token_ids.reshape(-1), sorted=True, return_inverse=True
            )
            candidate_rows = self.token_row_fn(unique_token_ids)
            unique_scores = self.reranker.score_candidate_rows(
                parent_hidden, candidate_rows
            )
            reranker_scores = unique_scores.gather(
                1, inverse.view_as(topk_token_ids)
            ).to(topk_log_probs.dtype)

            combined_scores = (
                parent_scores[:, None]
                + self.draft_score_weight * topk_log_probs
                + self.reranker_score_weight * reranker_scores
            )
            flat_scores = combined_scores.reshape(-1)
            keep = min(self.beam_size, flat_scores.numel())
            selected_scores, selected_flat = torch.topk(flat_scores, k=keep)
            selected_parent_offsets = selected_flat // top_k
            selected_child_offsets = selected_flat % top_k
            selected_parent_indices = active_indices[selected_parent_offsets]
            selected_parent_hidden = parent_hidden[selected_parent_offsets]
            selected_token_ids = topk_token_ids[
                selected_parent_offsets, selected_child_offsets
            ]
            selected_child_hidden = self.advance_hidden_fn(
                selected_parent_hidden, selected_token_ids, depth
            )

            new_active = []
            for i in range(keep):
                node_idx = len(token_nodes)
                token_nodes.append(selected_token_ids[i])
                parent_nodes.append(selected_parent_indices[i])
                score_nodes.append(selected_scores[i])
                depth_nodes.append(torch.tensor(depth, dtype=torch.long, device=device))
                hidden_nodes.append(selected_child_hidden[i])
                new_active.append(node_idx)
            active_indices = torch.tensor(new_active, dtype=torch.long, device=device)

        token_ids = torch.stack(token_nodes).long()
        parent_indices = torch.stack(parent_nodes).long()
        scores = torch.stack(score_nodes)
        depths = torch.stack(depth_nodes).long()
        hidden_states = torch.stack(hidden_nodes)
        tree_mask = _make_tree_mask(parent_indices)
        retrieve_indices = _make_retrieve_indices(parent_indices, depths)
        return RerankedTreeOutput(
            token_ids=token_ids,
            parent_indices=parent_indices,
            scores=scores,
            depths=depths,
            hidden_states=hidden_states,
            active_indices=active_indices,
            tree_mask=tree_mask,
            tree_position_ids=depths,
            retrieve_indices=retrieve_indices,
        )


# RERANKER PIPELINE: online trainer that learns restricted logits at every step.
class OnlineEagle3RerankerModel(nn.Module):
    """Train a low-rank reranker from frozen EAGLE states and target logits."""

    def __init__(
        self,
        *,
        draft_model: nn.Module,
        reranker: LowRankTokenReranker,
        length: int = 7,
        top_k: int = 20,
        attention_backend: str = "sdpa",
        combine_draft_logits_for_loss: bool = False,
        d2t: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.draft_model = draft_model
        self.reranker = reranker
        self.length = length
        self.top_k = top_k
        self.attention_backend = attention_backend
        self.combine_draft_logits_for_loss = combine_draft_logits_for_loss
        self.register_buffer("d2t_override", d2t if d2t is not None else None)

        for param in self.draft_model.parameters():
            param.requires_grad = False
        self.draft_model.eval()

    def _position_ids(
        self, position_ids: Optional[Tensor], seq_length: int, device: torch.device
    ) -> Tensor:
        if position_ids is not None:
            return position_ids.long().view(-1, seq_length)
        return torch.arange(seq_length, dtype=torch.long, device=device).view(
            1, seq_length
        )

    def _attention_mask(
        self,
        *,
        attention_mask: Optional[Tensor],
        hidden_states: Tensor,
        batch_size: int,
        seq_length: int,
    ) -> Tensor:
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=hidden_states.device
            )
        if self.attention_backend == "sdpa":
            return self.draft_model.prepare_decoder_attention_mask(
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                batch_size=batch_size,
                seq_length=seq_length,
                past_key_values_length=0,
            )
        return attention_mask

    def _candidate_rows(self, candidate_draft_ids: Tensor) -> Tensor:
        return self.draft_model.lm_head.weight[candidate_draft_ids]

    def _candidate_target_ids(self, candidate_draft_ids: Tensor) -> Tensor:
        d2t = self.d2t_override
        if d2t is None:
            d2t = getattr(self.draft_model, "d2t", None)
        return _candidate_target_ids(candidate_draft_ids, d2t)

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        loss_mask: Tensor,
        target: Tensor,
        hidden_states: Tensor,
        position_ids: Optional[Tensor] = None,
    ) -> RerankerTrainOutput:
        batch_size, seq_length, _ = hidden_states.shape
        with torch.no_grad():
            draft_hidden_states = self.draft_model.project_hidden_states(hidden_states)

        if self.attention_backend in ["sdpa", "fa", "usp"]:
            cache_hidden = [[], []]
            past_key_values = None
        elif self.attention_backend == "flex_attention":
            cache_hidden = None
            past_key_values = DynamicCache()
        else:
            raise ValueError(f"Unknown attention backend: {self.attention_backend}")

        position_ids = self._position_ids(
            position_ids, seq_length, draft_hidden_states.device
        )
        attention_mask = self._attention_mask(
            attention_mask=attention_mask,
            hidden_states=draft_hidden_states,
            batch_size=batch_size,
            seq_length=seq_length,
        )
        target_padded = F.pad(target, pad=(0, 0, 0, self.length), value=0.0)
        global_input_ids = input_ids
        global_loss_mask = loss_mask

        losses: List[Tensor] = []
        metric_values: Dict[str, List[Tensor]] = {
            "restricted_top1": [],
            "teacher_entropy": [],
        }

        for idx in range(self.length):
            target_slice = target_padded[:, idx : idx + seq_length, :].contiguous()
            with torch.no_grad():
                inputs_embeds = self.draft_model.embed_input_ids(global_input_ids)
                inputs_embeds = inputs_embeds.to(draft_hidden_states.dtype)
                step_hidden_states = self.draft_model.backbone(
                    input_embeds=inputs_embeds,
                    hidden_states=draft_hidden_states,
                    cache_hidden=cache_hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                draft_hidden_states = step_hidden_states
                draft_logits = self.draft_model.compute_logits(step_hidden_states)
                draft_log_probs = F.log_softmax(draft_logits.float(), dim=-1)
                top_k = min(self.top_k, draft_log_probs.shape[-1])
                topk_draft_log_probs, topk_draft_ids = torch.topk(
                    draft_log_probs, k=top_k, dim=-1
                )
                candidate_target_ids = self._candidate_target_ids(topk_draft_ids)

            flat_hidden = step_hidden_states.reshape(-1, step_hidden_states.shape[-1])
            candidate_rows = self._candidate_rows(topk_draft_ids).reshape(
                -1, top_k, self.draft_model.lm_head.weight.shape[-1]
            )
            reranker_scores = self.reranker.score_candidate_rows(
                flat_hidden.detach(), candidate_rows.detach()
            ).view(batch_size, seq_length, top_k)
            if self.combine_draft_logits_for_loss:
                reranker_scores = reranker_scores + topk_draft_log_probs.to(
                    reranker_scores.dtype
                )

            mask = global_loss_mask
            if mask.dim() == 2:
                mask = mask[..., None]
            loss, metrics = restricted_candidate_kl_loss(
                student_scores=reranker_scores,
                teacher_logits=target_slice,
                candidate_target_ids=candidate_target_ids,
                position_mask=mask,
            )
            losses.append(loss)
            for key, value in metrics.items():
                metric_values[key].append(value)

            if idx != self.length - 1:
                global_input_ids = _padding(global_input_ids, left=False)
                global_loss_mask = _padding(global_loss_mask, left=False)

        metrics = {
            key: torch.stack(values).mean() for key, values in metric_values.items()
        }
        return RerankerTrainOutput(
            loss=sum(losses) / len(losses),
            losses=losses,
            metrics=metrics,
        )
