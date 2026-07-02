from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from specforge.modeling.draft import Eagle3DraftModel
from specforge.utils import padding


VerifierLabelType = Literal["target_top1", "target_prob"]


@dataclass
class VerifierForwardOutput:
    loss: torch.Tensor
    losses: List[torch.Tensor]
    metrics: Dict[str, torch.Tensor]


class Eagle3CandidateVerifier(nn.Module):
    """Score EAGLE candidate tokens before final target verification.

    This model is intentionally small: it predicts whether a draft-vocab
    candidate is worth keeping in the target verification tree from draft
    hidden states, candidate ids, draft log-probabilities, and the TTT depth.
    It never replaces target verification; it only learns a reranking signal.
    """

    def __init__(
        self,
        *,
        draft_hidden_size: int,
        draft_vocab_size: int,
        max_depth: int = 8,
        candidate_embed_dim: int = 128,
        hidden_size: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.draft_hidden_size = draft_hidden_size
        self.draft_vocab_size = draft_vocab_size
        self.max_depth = max_depth
        self.candidate_embed_dim = candidate_embed_dim
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.candidate_embed = nn.Embedding(draft_vocab_size, candidate_embed_dim)
        self.depth_embed = nn.Embedding(max_depth, candidate_embed_dim)
        feature_size = draft_hidden_size + candidate_embed_dim * 2 + 2
        self.scorer = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        *,
        draft_hidden_states: torch.Tensor,
        candidate_token_ids: torch.Tensor,
        draft_log_probs: torch.Tensor,
        draft_margins: torch.Tensor,
        depth: int,
    ) -> torch.Tensor:
        scorer_dtype = next(self.scorer.parameters()).dtype
        batch_size, seq_len, top_k = candidate_token_ids.shape
        hidden = draft_hidden_states.to(scorer_dtype)[:, :, None, :].expand(
            batch_size, seq_len, top_k, draft_hidden_states.shape[-1]
        )
        token_embeds = self.candidate_embed(candidate_token_ids).to(scorer_dtype)
        depth_id = min(depth, self.max_depth - 1)
        depth_ids = torch.full_like(candidate_token_ids, depth_id)
        depth_embeds = self.depth_embed(depth_ids).to(scorer_dtype)
        scalar_features = torch.stack(
            (draft_log_probs.to(scorer_dtype), draft_margins.to(scorer_dtype)), dim=-1
        )
        features = torch.cat(
            (hidden, token_embeds, depth_embeds, scalar_features), dim=-1
        )
        return self.scorer(features).squeeze(-1)

    def config_dict(self) -> Dict[str, object]:
        return {
            "draft_hidden_size": self.draft_hidden_size,
            "draft_vocab_size": self.draft_vocab_size,
            "max_depth": self.max_depth,
            "candidate_embed_dim": self.candidate_embed_dim,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
        }


class OnlineEagle3VerifierModel(nn.Module):
    """Train a verifier/reranker from online EAGLE + target outputs.

    The draft EAGLE model is frozen. During each forward pass we reproduce the
    EAGLE3 TTT unroll, take the top-k draft candidates at each position/depth,
    and train the verifier against labels derived from the online target logits.
    """

    def __init__(
        self,
        *,
        draft_model: Eagle3DraftModel,
        verifier: Eagle3CandidateVerifier,
        length: int = 7,
        top_k: int = 8,
        attention_backend: str = "sdpa",
        label_type: VerifierLabelType = "target_top1",
        positive_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.draft_model = draft_model
        self.verifier = verifier
        self.length = length
        self.top_k = top_k
        self.attention_backend = attention_backend
        self.label_type = label_type
        self.positive_weight = positive_weight

        for param in self.draft_model.parameters():
            param.requires_grad = False
        self.draft_model.eval()

    def _prepare_position_ids(
        self,
        position_ids: Optional[torch.Tensor],
        *,
        seq_length: int,
        past_key_values_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.attention_backend == "usp":
            return position_ids
        if position_ids is not None:
            return position_ids.long().view(-1, seq_length)
        return (
            torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            .unsqueeze(0)
            .view(-1, seq_length)
        )

    def _prepare_attention_mask(
        self,
        *,
        attention_mask: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        batch_size: int,
        seq_length: int,
        past_key_values_length: int,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length + past_key_values_length),
                dtype=torch.bool,
                device=hidden_states.device,
            )
        if self.attention_backend == "sdpa":
            return self.draft_model.prepare_decoder_attention_mask(
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                batch_size=batch_size,
                seq_length=seq_length,
                past_key_values_length=past_key_values_length,
            )
        return attention_mask

    def _target_prob_labels(
        self,
        *,
        target_logits: torch.Tensor,
        candidate_target_ids: torch.Tensor,
    ) -> torch.Tensor:
        target_log_probs = F.log_softmax(target_logits.float(), dim=-1)
        return target_log_probs.gather(-1, candidate_target_ids).exp()

    def _build_labels(
        self,
        *,
        target_logits: torch.Tensor,
        target_token_ids: torch.Tensor,
        candidate_target_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.label_type == "target_top1":
            return (candidate_target_ids == target_token_ids.unsqueeze(-1)).float()
        if self.label_type == "target_prob":
            return self._target_prob_labels(
                target_logits=target_logits,
                candidate_target_ids=candidate_target_ids,
            )
        raise ValueError(f"Unknown verifier label_type: {self.label_type}")

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        target: torch.Tensor,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> VerifierForwardOutput:
        if target is None:
            raise ValueError("OnlineEagle3VerifierModel requires target logits.")

        batch_size, seq_length, _ = hidden_states.shape
        past_key_values_length = 0

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

        if past_key_values is not None and len(past_key_values) > 0:
            past_key_values_length = past_key_values[0][0].shape[2]

        position_ids = self._prepare_position_ids(
            position_ids=position_ids,
            seq_length=seq_length,
            past_key_values_length=past_key_values_length,
            device=draft_hidden_states.device,
        )
        attention_mask = self._prepare_attention_mask(
            attention_mask=attention_mask,
            hidden_states=draft_hidden_states,
            batch_size=batch_size,
            seq_length=seq_length,
            past_key_values_length=past_key_values_length,
        )

        target_padded = F.pad(target, pad=(0, 0, 0, self.length), value=0.0)
        target_token_ids_padded = F.pad(
            target.argmax(-1),
            pad=(0, self.length),
            value=0,
        )

        global_input_ids = input_ids
        global_loss_mask = loss_mask
        losses: List[torch.Tensor] = []
        metric_values: Dict[str, List[torch.Tensor]] = {
            "draft_recall_at_1": [],
            "draft_recall_at_k": [],
            "verifier_recall_at_1": [],
            "positive_rate": [],
        }

        for idx in range(self.length):
            target_slice = target_padded[:, idx : idx + seq_length, :].contiguous()
            target_token_ids = target_token_ids_padded[
                :, idx : idx + seq_length
            ].contiguous()

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
                topk_log_probs, topk_draft_ids = torch.topk(
                    draft_log_probs,
                    k=top_k,
                    dim=-1,
                )
                if top_k > 1:
                    top1_margin = topk_log_probs[..., :1] - topk_log_probs[..., 1:2]
                else:
                    top1_margin = torch.zeros_like(topk_log_probs[..., :1])
                draft_margins = top1_margin.expand_as(topk_log_probs)
                candidate_target_ids = (
                    topk_draft_ids
                    + self.draft_model.d2t[topk_draft_ids].to(topk_draft_ids.device)
                )
                labels = self._build_labels(
                    target_logits=target_slice,
                    target_token_ids=target_token_ids,
                    candidate_target_ids=candidate_target_ids,
                )
                target_top1_match = (
                    candidate_target_ids == target_token_ids.unsqueeze(-1)
                )

            scores = self.verifier(
                draft_hidden_states=step_hidden_states.detach(),
                candidate_token_ids=topk_draft_ids.detach(),
                draft_log_probs=topk_log_probs.detach(),
                draft_margins=draft_margins.detach(),
                depth=idx,
            )

            mask = global_loss_mask.to(scores.device, dtype=scores.dtype)
            if mask.dim() == 2:
                mask = mask[..., None]
            candidate_mask = mask.expand_as(scores)

            pos_weight = torch.tensor(
                self.positive_weight,
                device=scores.device,
                dtype=scores.dtype,
            )
            loss_per_candidate = F.binary_cross_entropy_with_logits(
                scores,
                labels.to(scores.dtype),
                pos_weight=pos_weight,
                reduction="none",
            )
            loss = (
                (loss_per_candidate * candidate_mask).sum()
                / candidate_mask.sum().clamp_min(1e-6)
            )
            losses.append(loss)

            with torch.no_grad():
                position_mask = mask.squeeze(-1).bool()
                denom = position_mask.sum().clamp_min(1)
                target_match = target_top1_match
                draft_recall_at_1 = (
                    target_match[..., 0].logical_and(position_mask).sum() / denom
                )
                draft_recall_at_k = (
                    target_match.any(dim=-1).logical_and(position_mask).sum() / denom
                )
                selected = scores.argmax(dim=-1)
                verifier_selected_match = target_match.gather(
                    -1, selected.unsqueeze(-1)
                ).squeeze(-1)
                verifier_recall_at_1 = (
                    verifier_selected_match.logical_and(position_mask).sum() / denom
                )
                positive_rate = (
                    (labels * candidate_mask).sum()
                    / candidate_mask.sum().clamp_min(1e-6)
                )
                metric_values["draft_recall_at_1"].append(draft_recall_at_1.detach())
                metric_values["draft_recall_at_k"].append(draft_recall_at_k.detach())
                metric_values["verifier_recall_at_1"].append(
                    verifier_recall_at_1.detach()
                )
                metric_values["positive_rate"].append(positive_rate.detach())

            if idx != self.length - 1:
                global_input_ids = padding(global_input_ids, left=False)
                global_loss_mask = padding(global_loss_mask, left=False)

        metrics = {
            key: torch.stack(values).mean() for key, values in metric_values.items()
        }
        return VerifierForwardOutput(
            loss=sum(losses) / len(losses),
            losses=losses,
            metrics=metrics,
        )
