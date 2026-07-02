# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
import hashlib
import re
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import Counter
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import os
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from transformers.activations import ACT2FN
from transformers import AutoModelForCausalLM, AutoTokenizer
from configs import EConfig
from safetensors import safe_open
from datasets import load_dataset
import multiprocessing


CHAT_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "llama3": {
        "system_prompt": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
        "assistant_header": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "end_of_turn_token": "<|eot_id|>",
    },
    "qwen": {
        "system_prompt": "You are a helpful assistant.",
        "assistant_header": "<|im_start|>assistant\n",
        "end_of_turn_token": "<|im_end|>\n",
    },
    "qwen3-instruct": {
        "system_prompt": "You are a helpful assistant.",
        "assistant_header": "<|im_start|>assistant\n",
        "end_of_turn_token": "<|im_end|>\n",
        "ignore_tokens": ["<think>\n\n</think>\n\n"],
    },
}


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _train_cfg_get(config: Any, key: str, default: Any = None) -> Any:
    aliases = {
        "gradient_checkpointing": ("gradient_checkpointing", "gradient_checkpoint"),
        "max_len": ("max_len", "max_length"),
        "chat_template": ("chat_template",),
        "is_preformatted": ("is_preformatted",),
        "train_only_last_turn": ("train_only_last_turn",),
        "target_dtype": ("target_dtype",),
        "trust_remote_code": ("trust_remote_code",),
        "ttt_length": ("ttt_length", "length"),
    }
    for name in aliases.get(key, (key,)):
        value = _cfg_get(config, name, None)
        if value is not None:
            return value
    return default


def _resolve_torch_dtype(dtype_name: Optional[Union[str, torch.dtype]]) -> torch.dtype:
    if isinstance(dtype_name, torch.dtype):
        return dtype_name
    if dtype_name is None:
        return torch.float16
    dtype_name = str(dtype_name).replace("torch.", "").lower()
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(dtype_name, torch.float16)


def _chat_template_config(name: str) -> Dict[str, Any]:
    if name not in CHAT_TEMPLATES:
        raise ValueError(
            f"Unsupported chat_template={name!r}. Add it to CHAT_TEMPLATES in cnets.py."
        )
    return CHAT_TEMPLATES[name]


def _normalize_conversation(source: List[Dict[str, Any]], system_prompt: Optional[str]) -> List[Dict[str, str]]:
    if not source:
        return []
    normalized = []
    for sentence in source:
        role = sentence.get("role", sentence.get("from", ""))
        content = sentence.get("content", sentence.get("value", ""))
        if role in ("human", "user"):
            role = "user"
        elif role in ("gpt", "assistant"):
            role = "assistant"
        normalized.append({"role": role, "content": content})

    messages = []
    if normalized and normalized[0]["role"] == "system":
        messages.append(normalized[0])
        normalized = normalized[1:]
    elif system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    while normalized and normalized[0]["role"] != "user":
        normalized = normalized[1:]

    expected = ["user", "assistant"]
    for idx, message in enumerate(normalized):
        if message["role"] != expected[idx % 2]:
            break
        messages.append(message)
    return messages


def _assistant_loss_mask(
    tokenizer,
    conversation: str,
    template: Dict[str, Any],
    max_len: int,
    train_only_last_turn: bool = False,
) -> torch.Tensor:
    input_ids = tokenizer(
        conversation,
        max_length=max_len,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]
    loss_mask = torch.zeros(len(input_ids), dtype=torch.long)

    assistant_header = template["assistant_header"]
    end_of_turn_token = template["end_of_turn_token"]
    if assistant_header is None:
        raise ValueError("assistant_header is required for EAGLE training masks")

    if end_of_turn_token:
        pattern = re.escape(assistant_header) + r"([\s\S]*?(?:" + re.escape(end_of_turn_token) + r"|$))"
    else:
        pattern = re.escape(assistant_header) + r"([\s\S]*?$)"
    matches = list(re.finditer(pattern, conversation, re.DOTALL))
    if train_only_last_turn and matches:
        matches = [matches[-1]]

    for match in matches:
        content_start_char = match.start(1)
        content_end_char = match.end(1)
        prefix_ids = tokenizer.encode(
            conversation[:content_start_char],
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
        )
        full_ids = tokenizer.encode(
            conversation[:content_end_char],
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
        )
        start_token_idx = min(len(prefix_ids), len(input_ids))
        end_token_idx = min(len(full_ids), len(input_ids))
        if start_token_idx < end_token_idx:
            loss_mask[start_token_idx:end_token_idx] = 1

    for token_str in template.get("ignore_tokens", []) or []:
        start = 0
        while True:
            idx = conversation.find(token_str, start)
            if idx == -1:
                break
            prefix_ids = tokenizer.encode(
                conversation[:idx],
                add_special_tokens=False,
                truncation=True,
                max_length=max_len,
            )
            full_ids = tokenizer.encode(
                conversation[: idx + len(token_str)],
                add_special_tokens=False,
                truncation=True,
                max_length=max_len,
            )
            start_token_idx = min(len(prefix_ids), len(input_ids))
            end_token_idx = min(len(full_ids), len(input_ids))
            if start_token_idx < end_token_idx:
                loss_mask[start_token_idx:end_token_idx] = 0
            start = idx + len(token_str)
    return input_ids, loss_mask


def preprocess_eagle3_examples(
    examples: Dict[str, List[Any]],
    tokenizer,
    max_len: int,
    chat_template: str = "llama3",
    is_preformatted: bool = False,
    train_only_last_turn: bool = False,
) -> Dict[str, List[torch.Tensor]]:
    template = _chat_template_config(chat_template)
    new_examples = {"attention_mask": [], "input_ids": [], "loss_mask": []}
    sources = examples["text"] if is_preformatted else examples["conversations"]

    for source in sources:
        if is_preformatted:
            conversation = source
        else:
            messages = _normalize_conversation(source, template["system_prompt"])
            if not messages:
                continue
            conversation = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.unk_token_id

        input_ids, loss_mask = _assistant_loss_mask(
            tokenizer,
            conversation,
            template,
            max_len,
            train_only_last_turn=train_only_last_turn,
        )
        if len(input_ids) > max_len:
            continue
        if loss_mask.sum() == 0:
            continue
        attention_mask = torch.ones_like(loss_mask)

        new_examples["input_ids"].append(input_ids[None, :])
        new_examples["loss_mask"].append(loss_mask[None, :])
        new_examples["attention_mask"].append(attention_mask[None, :])

    return new_examples

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)



class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if not hasattr(config, "head_dim") and (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size * 2, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=getattr(self.config, "rope_theta", 10000),
            )
        else:
            rope_scaling = self.config.rope_scaling

            def rope_get(key, default=None):
                if isinstance(rope_scaling, dict):
                    return rope_scaling.get(key, default)
                return getattr(rope_scaling, key, default)

            scaling_type = rope_get("rope_type", rope_get("type"))
            scaling_factor = rope_get("factor")
            if scaling_type in (None, "default"):
                self.rotary_emb = LlamaRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=getattr(self.config, "rope_theta", 10000),
                )
            elif scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=getattr(self.config, "rope_theta", 10000),
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=getattr(self.config, "rope_theta", 10000),
                )
            elif scaling_type == "llama3":
                self.rotary_emb = LlamaRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=getattr(self.config, "rope_theta", 10000),
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            cache_hidden: Optional[List[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        lck = len(cache_hidden[0])

        # cache_k = [self.k_proj(hidden) for hidden in cache_hidden]
        # cache_v = [self.v_proj(hidden) for hidden in cache_hidden]

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)


        cos, sin = self.rotary_emb(query_states, seq_len=q_len + lck)
        cos, sin = cos.to(query_states.device), sin.to(query_states.device)
        # query_states = apply_rotary_pos_emb(query_states, cos, sin, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids + lck)


        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Avoid modify hidden cache inplace which will cause in-place modification error when enable gradient checkpoint. 
        # Return the updated hidden cache instead.
        if cache_hidden is None:
            local_cache_k = []
            local_cache_v = []
        else:
            local_cache_k = list(cache_hidden[0])
            local_cache_v = list(cache_hidden[1])

        local_cache_k.append(key_states)
        local_cache_v.append(value_states)
            
        cache_k = local_cache_k
        cache_v = local_cache_v

        k0 = cache_k[0]
        v0 = cache_v[0]

        attn_weights = torch.matmul(query_states, k0.transpose(2, 3)) / math.sqrt(self.head_dim)
        lck = len(cache_k)


        attn_weights = attn_weights + attention_mask

        for i in range(1, lck):
            ki = cache_k[i]

            qi = query_states
            kiq = ki

            attn_weightsi = (qi * kiq).sum(-1) / math.sqrt(self.head_dim)
            attn_weights = torch.cat((attn_weights, attn_weightsi[..., None]), dim=-1)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights0 = attn_weights[..., :q_len]

        attn_output = torch.matmul(attn_weights0, v0)

        for i in range(1, lck):
            vi = cache_v[i]
            attn_weightsi = attn_weights[..., q_len + i - 1]
            attn_outputi = attn_weightsi[..., None] * vi
            attn_output = attn_output + attn_outputi

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        # Return the updated hidden cache.
        new_past_key_value = [local_cache_k,local_cache_v]
        return attn_output, new_past_key_value


class LlamaMLP(nn.Module):
    def __init__(self, config, last=True):
        super().__init__()
        self.last = last
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # if last:
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # else:
        #     self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size * 2, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaDecoderLayeremb(nn.Module):
    def __init__(self, config, last=True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config, last=last)
        self.last = last
        # self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # if self.index!=0:

        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            input_emb: torch.Tensor,
            hidden_states: torch.Tensor,
            cache_hidden: [List[torch.Tensor]] = [],
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)

        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)

        return_hidden = hidden_states

        # cache_hidden.append(hidden_states)

        # Self Attention
        hidden_states, latest_hidden_cache = self.self_attn(
            cache_hidden=cache_hidden,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states


        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, return_hidden)


        return outputs, latest_hidden_cache


@torch.no_grad()
def padding(tensor, left=True):
    zeropadding = torch.zeros_like(tensor[:, -1:])
    if left:
        tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
    else:
        tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
    return tensor


def process_data(data_chunk):

    token_dict = Counter()
    input_ids = data_chunk["input_ids"]
    loss_mask = data_chunk["loss_mask"]
    for i in range(len(input_ids)):
        ids= input_ids[i][0]
        mask = loss_mask[i][0]
        for j in range(len(ids)):
            if mask[j] == 1:
                token_dict[ids[j]] += 1

    return token_dict


def merge_dicts(dicts):
    """合并多个 Counter 字典"""
    result = Counter()
    for d in dicts:
        result.update(d)
    return result


class Model(nn.Module):
    def __init__(self, config, ds_config, training_config, load_head=False, load_emb=True, path=None):
        super().__init__() 
        # self.layers = nn.ModuleList(
        #     [LlamaDecoderLayer(config, index=index) for index in range(config.num_hidden_layers)])
        self.train_config = training_config
        # Settng dschf to allow efficient ZeRO-3 usage between hf and ds.
        if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
            dschf = HfDeepSpeedConfig(ds_config)
        else:
            dschf = None
        self.midlayer = LlamaDecoderLayeremb(config)
        self.gradient_checkpointing = _train_cfg_get(self.train_config, "gradient_checkpointing", True)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.draft_vocab_size = config.draft_vocab_size
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.length = int(_train_cfg_get(self.train_config, "ttt_length", 7))
        target_dtype = _resolve_torch_dtype(
            _train_cfg_get(self.train_config, "target_dtype", getattr(config, "torch_dtype", None))
        )
        self.target_dtype = target_dtype
        self.target_model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=target_dtype,
            trust_remote_code=bool(_train_cfg_get(self.train_config, "trust_remote_code", False)),
        )
        zero_stage = None
        if isinstance(ds_config, dict):
            zero_stage = ds_config.get("zero_optimization", {}).get("stage")
        if torch.cuda.is_available() and zero_stage != 3:
            self.target_model.to(torch.cuda.current_device())
        self.target_model.eval()
        self.aux_hidden_states_layers = self._resolve_aux_hidden_states_layers(config)
        self.fc=nn.Linear(self.hidden_size*3, self.hidden_size, bias=False)
        for param in self.target_model.parameters():
            param.requires_grad = False

        if not load_emb:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        else:

            from safetensors import safe_open
            import json
            import os
            try:
                with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                with safe_open(os.path.join(path, emb_path),
                               framework="pt",
                               device="cpu") as f:
                    tensor_slice = f.get_slice("model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()
            except:
                with open(os.path.join(path, "pytorch_model.bin.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                weights = torch.load(os.path.join(path, emb_path))
                tensor = weights["model.embed_tokens.weight"].float()
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx, _weight=tensor)

        self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)

        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    def _resolve_aux_hidden_states_layers(self, config):
        eagle_config = getattr(config, "eagle_config", None)
        if isinstance(eagle_config, dict) and "eagle_aux_hidden_state_layer_ids" in eagle_config:
            layers = eagle_config["eagle_aux_hidden_state_layer_ids"]
            if len(layers) != 3:
                raise ValueError("eagle_aux_hidden_state_layer_ids must contain exactly 3 layers")
            return list(layers)

        target_config = getattr(self.target_model, "config", None)
        num_layers = getattr(target_config, "num_hidden_layers", None)
        if num_layers is None and hasattr(target_config, "text_config"):
            num_layers = getattr(target_config.text_config, "num_hidden_layers", None)
        if num_layers is None:
            raise ValueError("Cannot infer target num_hidden_layers; set eagle_config.eagle_aux_hidden_state_layer_ids")
        return [1, num_layers // 2 - 1, num_layers - 4]

    def _get_transformer_layers(self):
        model = self.target_model
        candidates = [
            ("model", "layers"),
            ("language_model", "layers"),
            ("transformer", "h"),
            ("gpt_neox", "layers"),
        ]
        for parent_name, layers_name in candidates:
            parent = getattr(model, parent_name, None)
            if parent is not None and hasattr(parent, layers_name):
                return getattr(parent, layers_name)
        if hasattr(model, "layers"):
            return model.layers
        raise ValueError("Could not locate transformer layers on the target model")

    def scandata(self, datapath, tokenizerpath):
        N = self.draft_vocab_size
        cache_key = "|".join(
            [
                str(datapath),
                str(tokenizerpath),
                str(N),
                str(self.vocab_size),
                str(_train_cfg_get(self.train_config, "max_len", 2048)),
                str(_train_cfg_get(self.train_config, "chat_template", "llama3")),
                str(_train_cfg_get(self.train_config, "is_preformatted", False)),
                str(_train_cfg_get(self.train_config, "train_only_last_turn", False)),
            ]
        )
        cache_path = f"cache_{hashlib.md5(cache_key.encode()).hexdigest()}.pt"
        if not os.path.exists(cache_path):
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizerpath,
                trust_remote_code=bool(_train_cfg_get(self.train_config, "trust_remote_code", False)),
            )
            dataset = load_dataset('json', data_files=datapath)
            dataset = dataset['train']
            # dataset = dataset.select(range(96))
            original_columns1 = dataset.column_names
            num_proc = 48


            def preprocess_function(examples):
                processed = preprocess_eagle3_examples(
                    examples,
                    tokenizer,
                    max_len=int(_train_cfg_get(self.train_config, "max_len", 2048)),
                    chat_template=_train_cfg_get(self.train_config, "chat_template", "llama3"),
                    is_preformatted=bool(_train_cfg_get(self.train_config, "is_preformatted", False)),
                    train_only_last_turn=bool(_train_cfg_get(self.train_config, "train_only_last_turn", False)),
                )
                return {
                    "input_ids": processed["input_ids"],
                    "loss_mask": processed["loss_mask"],
                }

            dataset = dataset.map(
                preprocess_function,
                batched=True,
                num_proc=num_proc,
                remove_columns=original_columns1,
                load_from_cache_file=False
            )
            #dataset.set_format(type="torch")



            num_processes = num_proc
            chunk_size = len(dataset) // num_processes + (len(dataset) % num_processes > 0)
            chunks = [dataset[i:i + chunk_size] for i in range(0, len(dataset), chunk_size)]

            # 创建进程池
            with multiprocessing.Pool(num_processes) as pool:
                # 并行处理数据块
                results = pool.map(process_data, chunks)

            # 合并结果
            token_dict = merge_dicts(results)


            total_frequency = sum(token_dict.values())
            if total_frequency == 0:
                raise ValueError(
                    "No trainable tokens found while building the draft vocab. "
                    "Check --chat-template, --is-preformatted, and the dataset schema."
                )
            if len(token_dict) < N:
                existing_tokens = set(token_dict.keys())
                missing_tokens = set(range(N)) - existing_tokens
                for token in missing_tokens:
                    token_dict[token] = 0
                    if len(token_dict) >= N:
                        break
            top_N = token_dict.most_common(N)
            top_N_frequency_sum = sum(freq for key, freq in top_N)
            top_N_ratio = top_N_frequency_sum / total_frequency
            print(f"top {N} token frequency ratio: {top_N_ratio:.2%}")
            used_tokens = [key for key, freq in top_N]
            used_tokens.sort()
            d2t = [used_tokens[i] - i for i in range(len(used_tokens))]
            t2d = [i in used_tokens for i in range(self.vocab_size)]
            d2t = torch.tensor(d2t)
            t2d = torch.tensor(t2d)
            cache = {
                "d2t": d2t,
                "t2d": t2d
            }
            torch.save(cache, cache_path)
        else:
            cache = torch.load(cache_path)
            d2t = cache["d2t"]
            t2d = cache["t2d"]
        self.register_buffer("d2t", d2t)
        self.register_buffer("t2d", t2d)
        self.l1smooth = nn.SmoothL1Loss(reduction="none")

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @torch.no_grad()
    def dataprepare(self, input_ids, attention_mask, loss_mask):
        device = input_ids.device
        self.target_model.eval()
        captured_states = {}
        handles = []
        layers = self._get_transformer_layers()

        def get_hook(layer_idx):
            def hook(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured_states[layer_idx] = hidden

            return hook

        for layer_idx in self.aux_hidden_states_layers:
            if layer_idx < 0 or layer_idx >= len(layers):
                raise ValueError(
                    f"Aux hidden-state layer {layer_idx} is out of bounds for target with {len(layers)} layers"
                )
            handles.append(layers[layer_idx].register_forward_hook(get_hook(layer_idx)))

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.target_dtype)
            if torch.cuda.is_available() and self.target_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        try:
            with autocast_ctx:
                outs = self.target_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    use_cache=False,
                )
        finally:
            for handle in handles:
                handle.remove()

        if len(captured_states) != 3:
            raise RuntimeError(
                f"Expected to capture 3 aux hidden states, captured {len(captured_states)}"
            )
        hidden_states0 = captured_states[self.aux_hidden_states_layers[0]]
        hidden_states1 = captured_states[self.aux_hidden_states_layers[1]]
        hidden_states2 = captured_states[self.aux_hidden_states_layers[2]]
        hidden_states=torch.cat((hidden_states0,hidden_states1,hidden_states2),dim=-1)
        # hidden_states=torch.cat((hidden_states0,hidden_states1),dim=-1)
        target = outs.logits
        target = padding(target, left=False)
        input_ids = padding(input_ids, left=False)

        if target is not None:
            target = target.to(device)
            loss_mask = loss_mask[..., None]
            loss_mask = loss_mask.to(device)

        return hidden_states, target, loss_mask, input_ids

    def forward(
            self,
            # hidden_states,
            input_ids,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            loss_mask: Optional[torch.Tensor] = None,

    ):
        hidden_states, target, loss_mask, input_ids = self.dataprepare(input_ids, attention_mask, loss_mask)

        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        # with torch.no_grad():
        #     inputs_embeds = self.embed_tokens(input_ids)
        #     inputs_embeds = inputs_embeds.detach()

        if self.training and self.gradient_checkpointing and not hidden_states.requires_grad:
            hidden_states.requires_grad = True

        hidden_states=self.fc(hidden_states)

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = hidden_states.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        plosses = []
        vlosses = []
        acces = []
        cache_hidden = [[], []]

        for idx in range(self.length):
            last = idx == self.length - 1
            inputs_embeds = self.embed_tokens(input_ids)
            if self.training and self.gradient_checkpointing and not inputs_embeds.requires_grad:
                inputs_embeds.requires_grad = True
            inputs_embeds = inputs_embeds.to(hidden_states.dtype)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, None, output_attentions)

                    return custom_forward

                layer_outputs, cache_hidden = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.midlayer),
                    inputs_embeds,
                    hidden_states,
                    cache_hidden,
                    attention_mask,
                    position_ids,
                )
            else:

                layer_outputs, cache_hidden = self.midlayer(
                    input_emb=inputs_embeds,
                    hidden_states=hidden_states,
                    cache_hidden=cache_hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=output_attentions,
                    use_cache=True,
                )

            hidden_states_out = layer_outputs[0]
            # cache_hidden.append(layer_outputs[1])
            # kv_cahce = layer_outputs[-1]

            with torch.no_grad():
                # hidden_states_target = padding(hidden_states, left=False)
                target_head = target
                target_max_token = target_head.argmax(-1)
                # Move d2t to the same device as target_max_token
                self.t2d = self.t2d.to(target_max_token.device)
                target_mask = self.t2d[target_max_token]
                target_mask = target_mask[..., None].int()
                position_mask = target_mask * loss_mask
                target_head = target_head[..., self.t2d]
                target_head = target_head.float()
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()



            hidden_states = hidden_states_out

            hidden_states_out = self.norm(hidden_states_out)

            logits = self.lm_head(hidden_states_out)
            logits = logits.float()
            out_logp = nn.LogSoftmax(dim=2)(logits)
            plogp = target_p * out_logp
            loss = -torch.sum(position_mask * plogp, 2).mean()
            plosses.append(loss)
            with torch.no_grad():
                acces.append(((logits.argmax(-1) == target_p.argmax(-1)) * position_mask.squeeze(-1)).sum().item() / (
                        loss_mask.sum().item() + 1e-6))

            if not last:
                input_ids = padding(input_ids, left=False)
                target = padding(target, left=False)
                loss_mask = padding(loss_mask, left=False)



        return plosses, vlosses, acces




