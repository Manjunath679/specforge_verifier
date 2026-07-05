import unittest
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "specforge"
    / "core"
    / "eagle3_reranker_pipeline.py"
)
SPEC = importlib.util.spec_from_file_location("eagle3_reranker_pipeline", MODULE_PATH)
reranker_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reranker_module
SPEC.loader.exec_module(reranker_module)

LowRankTokenReranker = reranker_module.LowRankTokenReranker
OnlineEagle3RerankerModel = reranker_module.OnlineEagle3RerankerModel
RerankedEagleTreeBuilder = reranker_module.RerankedEagleTreeBuilder


class DummyDraftModel(nn.Module):
    def __init__(self, vocab_size=16, draft_vocab_size=8, hidden_size=12):
        super().__init__()
        self.vocab_size = vocab_size
        self.draft_vocab_size = draft_vocab_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size * 3, hidden_size)
        self.backbone_proj = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, draft_vocab_size, bias=False)
        self.register_buffer("d2t", torch.zeros(draft_vocab_size, dtype=torch.long))

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
        return self.lm_head(hidden_states)


class TestEagle3RerankerPipeline(unittest.TestCase):
    def test_tree_builder_preserves_parent_links_and_hidden_states(self):
        torch.manual_seed(0)
        hidden_size = 6
        vocab_size = 9
        token_rows = torch.randn(vocab_size, hidden_size)
        token_embeds = torch.randn(vocab_size, hidden_size)
        reranker = LowRankTokenReranker(
            hidden_size=hidden_size,
            token_hidden_size=hidden_size,
            rank_dim=4,
        )

        def draft_logits_fn(hidden):
            return hidden @ token_rows.T

        def token_row_fn(token_ids):
            return token_rows[token_ids]

        def advance_hidden_fn(parent_hidden, token_ids, depth):
            return torch.tanh(parent_hidden + token_embeds[token_ids] + depth * 0.01)

        builder = RerankedEagleTreeBuilder(
            reranker=reranker,
            token_row_fn=token_row_fn,
            draft_logits_fn=draft_logits_fn,
            advance_hidden_fn=advance_hidden_fn,
            top_k_per_branch=4,
            beam_size=2,
            max_depth=3,
        )

        output = builder.build_tree(root_hidden=torch.randn(1, hidden_size))

        self.assertEqual(output.token_ids.numel(), 1 + 2 * 3)
        self.assertEqual(output.hidden_states.shape, (7, hidden_size))
        self.assertEqual(output.active_indices.numel(), 2)
        self.assertTrue((output.parent_indices[1:] >= 0).all())
        self.assertTrue((output.depths[output.active_indices] == 3).all())
        self.assertEqual(output.tree_mask.shape, (7, 7))
        self.assertGreaterEqual(output.retrieve_indices.shape[0], 1)

    def test_online_reranker_forward_and_backward(self):
        torch.manual_seed(1)
        batch_size, seq_len, hidden_size = 2, 5, 12
        vocab_size, draft_vocab_size = 16, 8
        draft = DummyDraftModel(
            vocab_size=vocab_size,
            draft_vocab_size=draft_vocab_size,
            hidden_size=hidden_size,
        )
        reranker = LowRankTokenReranker(
            hidden_size=hidden_size,
            token_hidden_size=hidden_size,
            rank_dim=6,
        )
        model = OnlineEagle3RerankerModel(
            draft_model=draft,
            reranker=reranker,
            length=3,
            top_k=4,
            attention_backend="sdpa",
        )

        output = model(
            input_ids=torch.randint(0, vocab_size, (batch_size, seq_len)),
            attention_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
            loss_mask=torch.ones(batch_size, seq_len, 1),
            target=torch.randn(batch_size, seq_len, vocab_size),
            hidden_states=torch.randn(batch_size, seq_len, hidden_size * 3),
        )

        self.assertEqual(len(output.losses), 3)
        self.assertGreater(output.loss.item(), 0)
        self.assertIn("restricted_top1", output.metrics)
        output.loss.backward()

        reranker_grad = sum(
            p.grad.abs().sum().item()
            for p in reranker.parameters()
            if p.grad is not None
        )
        draft_grad = sum(
            p.grad.abs().sum().item() for p in draft.parameters() if p.grad is not None
        )
        self.assertGreater(reranker_grad, 0)
        self.assertEqual(draft_grad, 0)


if __name__ == "__main__":
    unittest.main()
