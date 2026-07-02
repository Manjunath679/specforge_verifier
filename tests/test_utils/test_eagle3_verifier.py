import unittest

import torch
import torch.nn as nn

from specforge.core.eagle3_verifier import (
    Eagle3CandidateVerifier,
    OnlineEagle3VerifierModel,
)


class DummyDraftModel(nn.Module):
    def __init__(self, vocab_size=16, draft_vocab_size=8, hidden_size=12):
        super().__init__()
        self.vocab_size = vocab_size
        self.draft_vocab_size = draft_vocab_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size * 3, hidden_size)
        self.backbone_proj = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, draft_vocab_size)
        self.register_buffer("d2t", torch.zeros(draft_vocab_size, dtype=torch.long))
        self.register_buffer("t2d", torch.ones(vocab_size, dtype=torch.bool))

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
        return self.lm_head(hidden_states)


class TestEagle3Verifier(unittest.TestCase):
    def test_online_verifier_forward_and_backward(self):
        torch.manual_seed(0)
        batch_size, seq_len, hidden_size = 2, 5, 12
        vocab_size, draft_vocab_size = 16, 8
        draft = DummyDraftModel(
            vocab_size=vocab_size,
            draft_vocab_size=draft_vocab_size,
            hidden_size=hidden_size,
        )
        verifier = Eagle3CandidateVerifier(
            draft_hidden_size=hidden_size,
            draft_vocab_size=draft_vocab_size,
            max_depth=3,
            candidate_embed_dim=6,
            hidden_size=24,
        )
        model = OnlineEagle3VerifierModel(
            draft_model=draft,
            verifier=verifier,
            length=3,
            top_k=4,
            attention_backend="sdpa",
            label_type="target_top1",
        )

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        loss_mask = torch.ones(batch_size, seq_len, 1)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size * 3)
        target = torch.randn(batch_size, seq_len, vocab_size)
        target[..., 0] = 10.0

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            target=target,
            hidden_states=hidden_states,
        )

        self.assertEqual(len(output.losses), 3)
        self.assertGreater(output.loss.item(), 0)
        self.assertIn("verifier_recall_at_1", output.metrics)
        output.loss.backward()

        verifier_grad = sum(
            p.grad.abs().sum().item()
            for p in verifier.parameters()
            if p.grad is not None
        )
        draft_grad = sum(
            p.grad.abs().sum().item() for p in draft.parameters() if p.grad is not None
        )
        self.assertGreater(verifier_grad, 0)
        self.assertEqual(draft_grad, 0)


if __name__ == "__main__":
    unittest.main()
