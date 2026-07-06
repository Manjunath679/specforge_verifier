from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache


Tensor = torch.Tensor


@dataclass
class HiddenAdapterTrainOutput:
    loss: Tensor
    losses: List[Tensor]
    kl_losses: List[Tensor]
    hidden_losses: List[Tensor]
    metrics: Dict[str, Tensor]


# TEST TIME EAGLE EXP: local padding helper keeps this experiment independent from
# the original EAGLE3 training module.
def _padding(tensor: Tensor, left: bool = True) -> Tensor:
    zeropadding = torch.zeros_like(tensor[:, -1:])
    if left:
        return torch.cat((zeropadding, tensor[:, :-1]), dim=1)
    return torch.cat((tensor[:, 1:], zeropadding), dim=1)


# TEST TIME EAGLE EXP: target logits are converted to the draft vocab exactly for
# this side pipeline, leaving the original EAGLE3 loss path untouched.
def _compute_target_p_padded(
    *,
    target: Tensor,
    t2d: Tensor,
    loss_mask: Tensor,
    length: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    with torch.no_grad():
        target_head = target.float()
        target_token_ids = target_head.argmax(-1)
        target_mask = t2d.to(target_token_ids.device)[target_token_ids]
        target_mask = target_mask[..., None].int()
        position_mask = target_mask * loss_mask

        draft_target_head = target_head[..., t2d.to(target_head.device)]
        target_p = F.softmax(draft_target_head, dim=-1).detach()
        target_logsumexp = torch.logsumexp(target_head, dim=-1, keepdim=True)
        target_p_on_draft = torch.exp(draft_target_head - target_logsumexp).detach()
        target_token_ids = target_token_ids.detach()

        target_p_padded = F.pad(
            target_p,
            pad=(0, 0, 0, length),
            mode="constant",
            value=1 / target_p.shape[-1],
        )
        target_p_on_draft_padded = F.pad(
            target_p_on_draft,
            pad=(0, 0, 0, length),
            mode="constant",
            value=0.0,
        )
        target_token_ids_padded = F.pad(
            target_token_ids,
            pad=(0, length),
            mode="constant",
            value=0,
        )
    return (
        target_p_padded,
        target_p_on_draft_padded,
        target_token_ids_padded,
        position_mask,
    )


# TEST TIME EAGLE EXP: masked valid-token mean for KL, hidden loss, and metrics.
def _masked_mean(values: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    if mask.dim() == values.dim() + 1:
        mask = mask.squeeze(-1)
    mask = mask.to(device=values.device, dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(eps)


# TEST TIME EAGLE EXP: KL objective for the adapter's corrected logits.
def _masked_kl_loss(*, logits: Tensor, target_p: Tensor, position_mask: Tensor) -> Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    per_token = -(target_p.float() * log_probs).sum(dim=-1)
    return _masked_mean(per_token, position_mask)


# TEST TIME EAGLE EXP: hidden alignment is auxiliary supervision, not the main
# acceptance objective.
def _hidden_alignment_loss(
    *,
    predicted_hidden: Tensor,
    target_hidden: Tensor,
    mask: Tensor,
    loss_type: str,
) -> Tensor:
    predicted = predicted_hidden.float()
    target = target_hidden.float()
    if loss_type == "mse":
        per_token = (predicted - target).pow(2).mean(dim=-1)
    elif loss_type == "norm_mse":
        predicted = F.layer_norm(predicted, (predicted.shape[-1],))
        target = F.layer_norm(target, (target.shape[-1],))
        per_token = (predicted - target).pow(2).mean(dim=-1)
    elif loss_type == "cosine":
        per_token = 1 - F.cosine_similarity(predicted, target, dim=-1)
    else:
        raise ValueError(
            "--hidden-loss-type must be one of {'mse', 'norm_mse', 'cosine'}, "
            f"got {loss_type!r}"
        )
    return _masked_mean(per_token, mask)


# TEST TIME EAGLE EXP: expected speculative acceptance under the draft-vocab
# target probabilities.
def _acceptance_rate(
    *, logits: Tensor, target_p_on_draft: Tensor, position_mask: Tensor
) -> Tensor:
    draft_p = F.softmax(logits.float(), dim=-1).to(target_p_on_draft.dtype)
    per_token = torch.minimum(draft_p, target_p_on_draft).sum(dim=-1)
    return _masked_mean(per_token, position_mask)


# TEST TIME EAGLE EXP: top-1 target-token metric for quick training feedback.
def _accuracy(
    *,
    logits: Tensor,
    target_token_ids: Tensor,
    loss_mask: Tensor,
    d2t: Optional[Tensor],
) -> Tensor:
    pred_draft_token_ids = logits.argmax(dim=-1)
    if d2t is None:
        pred_target_token_ids = pred_draft_token_ids
    else:
        d2t = d2t.to(pred_draft_token_ids.device)
        pred_target_token_ids = pred_draft_token_ids + d2t[pred_draft_token_ids]
    correct = (pred_target_token_ids == target_token_ids).float()
    return _masked_mean(correct, loss_mask)


# TEST TIME EAGLE EXP: gated residual adapter starts as exact normal EAGLE and
# learns small hidden corrections before the draft LM head.
class GatedResidualHiddenAdapter(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        bottleneck_size: Optional[int] = None,
        gate_type: str = "scalar",
        gate_init: float = 0.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if gate_type not in {"scalar", "channel"}:
            raise ValueError("--hidden-adapter-gate-type must be 'scalar' or 'channel'")
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        self.gate_type = gate_type
        self.gate_init = gate_init
        self.dropout = dropout

        self.input_norm = nn.LayerNorm(hidden_size)
        if bottleneck_size is None or bottleneck_size <= 0:
            self.adapter = nn.Linear(hidden_size, hidden_size)
        else:
            self.adapter = nn.Sequential(
                nn.Linear(hidden_size, bottleneck_size),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck_size, hidden_size),
            )

        gate_shape = (hidden_size,) if gate_type == "channel" else (1,)
        self.gate = nn.Parameter(torch.full(gate_shape, float(gate_init)))
        self.reset_parameters()

    # TEST TIME EAGLE EXP: small adapter init avoids a sudden logit geometry jump.
    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # TEST TIME EAGLE EXP: residual correction, not replacement.
    def forward(self, hidden_states: Tensor) -> Tensor:
        adapter_dtype = self.input_norm.weight.dtype
        normalized = self.input_norm(hidden_states.to(adapter_dtype))
        delta = self.adapter(normalized)
        gate = torch.tanh(self.gate).view(*((1,) * (hidden_states.dim() - 1)), -1)
        corrected = hidden_states.to(adapter_dtype) + gate * delta
        return corrected.to(hidden_states.dtype)

    # TEST TIME EAGLE EXP: checkpoint metadata for loading the adapter later.
    def config_dict(self) -> Dict[str, object]:
        return {
            "hidden_size": self.hidden_size,
            "bottleneck_size": self.bottleneck_size,
            "gate_type": self.gate_type,
            "gate_init": self.gate_init,
            "dropout": self.dropout,
        }

    # TEST TIME EAGLE EXP: scalar metric for checking whether the adapter moved.
    def gate_abs_mean(self) -> Tensor:
        return torch.tanh(self.gate.detach()).abs().mean()


# TEST TIME EAGLE EXP: inference wrapper applies the hidden adapter only at the
# draft LM-head boundary while delegating backbone/state evolution to EAGLE.
class HiddenAdapterDraftWrapper(nn.Module):
    def __init__(
        self,
        *,
        draft_model: nn.Module,
        hidden_adapter: GatedResidualHiddenAdapter,
    ) -> None:
        super().__init__()
        self.draft_model = draft_model
        self.hidden_adapter = hidden_adapter

    @property
    def config(self):
        return self.draft_model.config

    @property
    def t2d(self):
        return self.draft_model.t2d

    @property
    def d2t(self):
        return self.draft_model.d2t

    @property
    def lm_head(self):
        return self.draft_model.lm_head

    # TEST TIME EAGLE EXP: delegate target-hidden projection unchanged.
    def project_hidden_states(self, hidden_states: Tensor) -> Tensor:
        return self.draft_model.project_hidden_states(hidden_states)

    # TEST TIME EAGLE EXP: delegate token embeddings unchanged.
    def embed_input_ids(self, input_ids: Tensor) -> Tensor:
        return self.draft_model.embed_input_ids(input_ids)

    # TEST TIME EAGLE EXP: delegate attention-mask construction unchanged.
    def prepare_decoder_attention_mask(self, *args, **kwargs) -> Tensor:
        return self.draft_model.prepare_decoder_attention_mask(*args, **kwargs)

    # TEST TIME EAGLE EXP: delegate recurrent draft hidden dynamics unchanged.
    def backbone(self, *args, **kwargs) -> Tensor:
        return self.draft_model.backbone(*args, **kwargs)

    # TEST TIME EAGLE EXP: this is the only inference behavior change.
    def compute_logits(self, hidden_states: Tensor) -> Tensor:
        corrected_hidden = self.hidden_adapter(hidden_states)
        return self.draft_model.compute_logits(corrected_hidden)


# TEST TIME EAGLE EXP: online trainer for FC-only hidden correction on top of a
# frozen EAGLE3 checkpoint.
class OnlineEagle3HiddenAdapterModel(nn.Module):
    """Train a gated residual hidden adapter without modifying EAGLE3 weights.

    Default behavior is LM-head-side correction only: the corrected hidden state
    is used for logits/loss, while the next TTT step receives the original draft
    hidden state. Set ``feed_corrected_hidden=True`` for the riskier truncated
    backprop experiment where corrected states are fed forward inside the window.
    """

    def __init__(
        self,
        *,
        draft_model: nn.Module,
        hidden_adapter: GatedResidualHiddenAdapter,
        length: int = 1,
        attention_backend: str = "sdpa",
        kl_weight: float = 0.8,
        hidden_weight: float = 0.2,
        hidden_loss_type: str = "mse",
        feed_corrected_hidden: bool = False,
    ) -> None:
        super().__init__()
        if length < 1:
            raise ValueError("length must be >= 1")
        if attention_backend == "usp":
            raise ValueError("Hidden adapter experiment does not support USP yet")
        if kl_weight < 0 or hidden_weight < 0:
            raise ValueError("kl_weight and hidden_weight must be non-negative")
        self.draft_model = draft_model
        self.hidden_adapter = hidden_adapter
        self.length = length
        self.attention_backend = attention_backend
        self.kl_weight = kl_weight
        self.hidden_weight = hidden_weight
        self.hidden_loss_type = hidden_loss_type
        self.feed_corrected_hidden = feed_corrected_hidden

        for param in self.draft_model.parameters():
            param.requires_grad = False
        self.draft_model.eval()

    # TEST TIME EAGLE EXP: position ids for text-only hidden-adapter training.
    def _position_ids(
        self, position_ids: Optional[Tensor], seq_length: int, device: torch.device
    ) -> Tensor:
        if position_ids is not None:
            return position_ids.long().view(-1, seq_length)
        return torch.arange(seq_length, dtype=torch.long, device=device).view(
            1, seq_length
        )

    # TEST TIME EAGLE EXP: reuse the draft model's causal mask where needed.
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

    # TEST TIME EAGLE EXP: DDP wraps the adapter, so logging helpers must unwrap it.
    def _gate_abs_mean(self) -> Tensor:
        hidden_adapter = self.hidden_adapter
        if hasattr(hidden_adapter, "module"):
            hidden_adapter = hidden_adapter.module
        return hidden_adapter.gate_abs_mean()

    # TEST TIME EAGLE EXP: draft-vocab KL and hidden-supervision slices share the
    # same TTT window offsets as original EAGLE.
    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        loss_mask: Tensor,
        target: Tensor,
        hidden_states: Tensor,
        position_ids: Optional[Tensor] = None,
    ) -> HiddenAdapterTrainOutput:
        (
            target_p_padded,
            target_p_on_draft_padded,
            target_token_ids_padded,
            position_mask,
        ) = _compute_target_p_padded(
            target=target,
            t2d=self.draft_model.t2d,
            loss_mask=loss_mask,
            length=self.length,
        )

        batch_size, seq_length, _ = hidden_states.shape
        with torch.no_grad():
            target_projected_hidden = self.draft_model.project_hidden_states(
                hidden_states
            )
        target_hidden_padded = F.pad(
            target_projected_hidden,
            pad=(0, 0, 0, self.length),
            mode="constant",
            value=0.0,
        )

        draft_hidden_states = target_projected_hidden
        position_ids = self._position_ids(
            position_ids, seq_length, draft_hidden_states.device
        )
        attention_mask = self._attention_mask(
            attention_mask=attention_mask,
            hidden_states=draft_hidden_states,
            batch_size=batch_size,
            seq_length=seq_length,
        )

        if self.attention_backend in ["sdpa", "fa"]:
            cache_hidden = [[], []]
            past_key_values = None
        elif self.attention_backend == "flex_attention":
            cache_hidden = None
            past_key_values = DynamicCache()
        else:
            raise ValueError(f"Unknown attention backend: {self.attention_backend}")

        global_input_ids = input_ids
        global_loss_mask = loss_mask
        global_position_mask = position_mask

        losses: List[Tensor] = []
        kl_losses: List[Tensor] = []
        hidden_losses: List[Tensor] = []
        accs: List[Tensor] = []
        acceptance_rates: List[Tensor] = []

        for idx in range(self.length):
            target_p = target_p_padded[:, idx : idx + seq_length, :].contiguous()
            target_p_on_draft = target_p_on_draft_padded[
                :, idx : idx + seq_length, :
            ].contiguous()
            target_token_ids = target_token_ids_padded[
                :, idx : idx + seq_length
            ].contiguous()
            step_target_hidden = target_hidden_padded[
                :, idx + 1 : idx + 1 + seq_length, :
            ].contiguous()

            inputs_embeds = self.draft_model.embed_input_ids(global_input_ids)
            inputs_embeds = inputs_embeds.to(draft_hidden_states.dtype)

            if self.feed_corrected_hidden:
                step_hidden_states = self.draft_model.backbone(
                    input_embeds=inputs_embeds,
                    hidden_states=draft_hidden_states,
                    cache_hidden=cache_hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                adapter_input = step_hidden_states
            else:
                with torch.no_grad():
                    step_hidden_states = self.draft_model.backbone(
                        input_embeds=inputs_embeds,
                        hidden_states=draft_hidden_states,
                        cache_hidden=cache_hidden,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                adapter_input = step_hidden_states.detach()

            corrected_hidden = self.hidden_adapter(adapter_input)
            logits = self.draft_model.compute_logits(corrected_hidden)

            kl_loss = _masked_kl_loss(
                logits=logits,
                target_p=target_p,
                position_mask=global_position_mask,
            )
            hidden_loss = _hidden_alignment_loss(
                predicted_hidden=corrected_hidden,
                target_hidden=step_target_hidden,
                mask=global_loss_mask,
                loss_type=self.hidden_loss_type,
            )
            loss = self.kl_weight * kl_loss + self.hidden_weight * hidden_loss

            losses.append(loss)
            kl_losses.append(kl_loss)
            hidden_losses.append(hidden_loss)
            accs.append(
                _accuracy(
                    logits=logits.detach(),
                    target_token_ids=target_token_ids,
                    loss_mask=global_loss_mask,
                    d2t=getattr(self.draft_model, "d2t", None),
                )
            )
            acceptance_rates.append(
                _acceptance_rate(
                    logits=logits.detach(),
                    target_p_on_draft=target_p_on_draft,
                    position_mask=global_position_mask,
                )
            )

            if self.feed_corrected_hidden:
                draft_hidden_states = corrected_hidden
            else:
                draft_hidden_states = step_hidden_states.detach()

            if idx != self.length - 1:
                global_input_ids = _padding(global_input_ids, left=False)
                global_loss_mask = _padding(global_loss_mask, left=False)
                global_position_mask = _padding(global_position_mask, left=False)

        metrics = {
            "kl_loss": torch.stack([loss.detach() for loss in kl_losses]).mean(),
            "hidden_loss": torch.stack(
                [loss.detach() for loss in hidden_losses]
            ).mean(),
            "acc": torch.stack(accs).mean(),
            "acceptance_rate": torch.stack(acceptance_rates).mean(),
            "gate_abs_mean": self._gate_abs_mean().to(losses[0].device),
        }
        return HiddenAdapterTrainOutput(
            loss=sum(losses) / len(losses),
            losses=losses,
            kl_losses=kl_losses,
            hidden_losses=hidden_losses,
            metrics=metrics,
        )
