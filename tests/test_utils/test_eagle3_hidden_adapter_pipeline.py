import importlib.util
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "specforge"
    / "core"
    / "eagle3_hidden_adapter_pipeline.py"
)
SPEC = importlib.util.spec_from_file_location(
    "eagle3_hidden_adapter_pipeline", MODULE_PATH
)
hidden_adapter_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = hidden_adapter_module
SPEC.loader.exec_module(hidden_adapter_module)

GatedResidualHiddenAdapter = hidden_adapter_module.GatedResidualHiddenAdapter
HiddenAdapterDraftWrapper = hidden_adapter_module.HiddenAdapterDraftWrapper
OnlineEagle3HiddenAdapterModel = hidden_adapter_module.OnlineEagle3HiddenAdapterModel


class DummyDraftModel(nn.Module):
    def __init__(self, vocab_size=16, hidden_size=12):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size * 3, hidden_size)
        self.backbone_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.register_buffer("t2d", torch.ones(vocab_size, dtype=torch.bool))
        self.register_buffer("d2t", torch.zeros(vocab_size, dtype=torch.long))

    def project_hidden_states(self, hidden_states):
        return self.proj(hidden_states)

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def prepare_decoder_attention_mask(
        self,
        attention_mask,
        hidden_states,
        batch_size,
        seq_length,
        past_key_values_length,
    ):
        del hidden_states, batch_size, seq_length, past_key_values_length
        return attention_mask

    def backbone(
        self,
        input_embeds,
        hidden_states,
        cache_hidden,
        attention_mask,
        position_ids,
        past_key_values=None,
        use_cache=True,
    ):
        del cache_hidden, attention_mask, position_ids, past_key_values, use_cache
        return torch.tanh(self.backbone_proj(input_embeds + hidden_states))

    def compute_logits(self, hidden_states):
        return self.lm_head(self.norm(hidden_states))


class TestEagle3HiddenAdapterPipeline(unittest.TestCase):
    def test_gated_adapter_starts_as_noop(self):
        torch.manual_seed(0)
        hidden = torch.randn(2, 3, 8)
        adapter = GatedResidualHiddenAdapter(
            hidden_size=8,
            bottleneck_size=4,
            gate_type="scalar",
            gate_init=0.0,
        )

        corrected = adapter(hidden)

        self.assertTrue(torch.equal(corrected, hidden))
        self.assertEqual(adapter.config_dict()["hidden_size"], 8)

    def test_interpolated_adapter_starts_close_to_draft_hidden(self):
        torch.manual_seed(0)
        hidden = torch.randn(2, 3, 8)
        adapter = GatedResidualHiddenAdapter(
            hidden_size=8,
            bottleneck_size=4,
            gate_type="scalar",
            gate_init=0.001,
            merge_type="interpolate",
        )

        corrected = adapter(hidden)

        self.assertEqual(adapter.config_dict()["merge_type"], "interpolate")
        self.assertLess((corrected - hidden).abs().mean().item(), 0.01)
        self.assertAlmostEqual(adapter.gate_abs_mean().item(), 0.001, places=5)

    def test_online_hidden_adapter_forward_and_backward(self):
        torch.manual_seed(1)
        batch_size, seq_len, hidden_size, vocab_size = 2, 5, 12, 16
        draft = DummyDraftModel(vocab_size=vocab_size, hidden_size=hidden_size)
        adapter = GatedResidualHiddenAdapter(
            hidden_size=hidden_size,
            bottleneck_size=6,
            gate_type="scalar",
            gate_init=0.0,
        )
        model = OnlineEagle3HiddenAdapterModel(
            draft_model=draft,
            hidden_adapter=adapter,
            length=3,
            attention_backend="sdpa",
            kl_weight=0.8,
            hidden_weight=0.2,
            hidden_loss_type="mse",
        )

        output = model(
            input_ids=torch.randint(0, vocab_size, (batch_size, seq_len)),
            attention_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
            loss_mask=torch.ones(batch_size, seq_len, 1),
            target=torch.randn(batch_size, seq_len, vocab_size),
            hidden_states=torch.randn(batch_size, seq_len, hidden_size * 3),
        )

        self.assertEqual(len(output.losses), 3)
        self.assertEqual(len(output.kl_losses), 3)
        self.assertEqual(len(output.hidden_losses), 3)
        self.assertGreater(output.loss.item(), 0)
        self.assertIn("acceptance_rate", output.metrics)
        output.loss.backward()

        adapter_grad = sum(
            p.grad.abs().sum().item()
            for p in adapter.parameters()
            if p.grad is not None
        )
        draft_grad = sum(
            p.grad.abs().sum().item() for p in draft.parameters() if p.grad is not None
        )
        self.assertGreater(adapter_grad, 0)
        self.assertEqual(draft_grad, 0)

    def test_state_adapter_forward_and_backward(self):
        torch.manual_seed(3)
        batch_size, seq_len, hidden_size, vocab_size = 2, 5, 12, 16
        draft = DummyDraftModel(vocab_size=vocab_size, hidden_size=hidden_size)
        adapter = GatedResidualHiddenAdapter(
            hidden_size=hidden_size,
            bottleneck_size=6,
            gate_type="scalar",
            gate_init=0.05,
            merge_type="interpolate",
        )
        model = OnlineEagle3HiddenAdapterModel(
            draft_model=draft,
            hidden_adapter=adapter,
            length=4,
            attention_backend="sdpa",
            kl_weight=0.8,
            hidden_weight=0.2,
            hidden_loss_type="mse",
            adapter_placement="state",
        )

        output = model(
            input_ids=torch.randint(0, vocab_size, (batch_size, seq_len)),
            attention_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
            loss_mask=torch.ones(batch_size, seq_len, 1),
            target=torch.randn(batch_size, seq_len, vocab_size),
            hidden_states=torch.randn(batch_size, seq_len, hidden_size * 3),
        )

        self.assertEqual(len(output.losses), 4)
        self.assertGreater(output.loss.item(), 0)
        output.loss.backward()

        adapter_grad = sum(
            p.grad.abs().sum().item()
            for p in adapter.parameters()
            if p.grad is not None
        )
        draft_grad = sum(
            p.grad.abs().sum().item() for p in draft.parameters() if p.grad is not None
        )
        self.assertGreater(adapter_grad, 0)
        self.assertEqual(draft_grad, 0)

    def test_inference_wrapper_changes_only_compute_logits(self):
        torch.manual_seed(2)
        hidden_size, vocab_size = 12, 16
        draft = DummyDraftModel(vocab_size=vocab_size, hidden_size=hidden_size)
        adapter = GatedResidualHiddenAdapter(
            hidden_size=hidden_size,
            bottleneck_size=6,
            gate_type="scalar",
            gate_init=0.0,
        )
        wrapper = HiddenAdapterDraftWrapper(draft_model=draft, hidden_adapter=adapter)
        hidden = torch.randn(2, 4, hidden_size)

        self.assertTrue(
            torch.equal(wrapper.compute_logits(hidden), draft.compute_logits(hidden))
        )
        with torch.no_grad():
            adapter.gate.fill_(0.5)

        self.assertFalse(
            torch.equal(wrapper.compute_logits(hidden), draft.compute_logits(hidden))
        )
        self.assertIs(wrapper.lm_head, draft.lm_head)

    def test_state_wrapper_keeps_logits_raw_and_adapts_next_hidden(self):
        torch.manual_seed(4)
        hidden_size, vocab_size = 12, 16
        draft = DummyDraftModel(vocab_size=vocab_size, hidden_size=hidden_size)
        adapter = GatedResidualHiddenAdapter(
            hidden_size=hidden_size,
            bottleneck_size=6,
            gate_type="scalar",
            gate_init=0.2,
            merge_type="interpolate",
        )
        wrapper = HiddenAdapterDraftWrapper(
            draft_model=draft,
            hidden_adapter=adapter,
            adapter_placement="state",
        )
        hidden = torch.randn(2, 4, hidden_size)

        self.assertTrue(
            torch.equal(wrapper.compute_logits(hidden), draft.compute_logits(hidden))
        )
        self.assertFalse(torch.equal(wrapper.adapt_next_hidden(hidden), hidden))


if __name__ == "__main__":
    unittest.main()
