"""
reward_model.py – RewardModel class (Task 3).

Architecture (per assignment spec):
  - Load checkpoints/sft/  (the full CausalLM state-dict).
  - Take the transformer body  (model.model),  drop the LM head.
  - Add nn.Linear(hidden_size, 1).
  - Reward = that linear layer applied to the final hidden state of
    the LAST NON-PADDING token.

We build the class from an AutoModelForCausalLM so we can reuse
exactly the same weight tensors as SFT without any key remapping.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from config import MODEL_NAME


class RewardModel(nn.Module):
    """
    Wraps the transformer body of SmolLM2-135M and adds a scalar
    reward head.  Initialised from the SFT checkpoint.
    """

    def __init__(self, checkpoint_path: str, device: torch.device):
        super().__init__()

        # ── 1. Load the full causal-LM architecture (random weights) ──
        causal_lm = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

        # ── 2. Keep only the transformer body, drop the LM head ─────
        self.backbone = causal_lm.model   # LlamaModel
        del causal_lm                     # free the LM-head memory

        # ── 3. Add a scalar reward head ─────────────────────────────
        hidden_size = self.backbone.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1, bias=True)

        # Initialise head with small weights for stable early training
        nn.init.normal_(self.reward_head.weight, std=0.02)
        nn.init.zeros_(self.reward_head.bias)

        # Disable KV-cache (we always pass full sequences)
        self.backbone.config.use_cache = False

        # ── 4. Load weights — handle both checkpoint types ───────────
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        if any(k.startswith("backbone.") for k in state):
            # Reward model checkpoint (keys: backbone.*, reward_head.*)
            # Load directly into self
            self.load_state_dict(state, strict=True)
        else:
            # SFT checkpoint (keys: model.*, lm_head.*)
            # Remap model.* -> backbone.* and load only backbone weights
            backbone_state = {
                k.replace("model.", "", 1): v
                for k, v in state.items()
                if k.startswith("model.")
            }
            self.backbone.load_state_dict(backbone_state, strict=True)
            # reward_head stays randomly initialised (correct for training)

        self.to(device)

    # ----------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,       # (B, L)
        attention_mask: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:                 # (B,)
        """
        Returns one scalar reward per example.
        Uses the hidden state at the LAST non-padding position.
        We find that position via attention_mask (never via
        input_ids != pad_token_id, as instructed).
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # (B, L, H)

        # Index of the last real (non-padding) token in each row
        # attention_mask is 1 for real tokens, 0 for padding
        last_token_idx = attention_mask.sum(dim=1) - 1  # (B,)
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)

        # (B, H) → (B, 1) → (B,)
        last_hidden = hidden_states[batch_idx, last_token_idx]  # (B, H)
        # Cast to float32 to avoid dtype mismatch when backbone runs in bfloat16
        rewards = self.reward_head(last_hidden.float()).squeeze(-1)  # (B,)

        return rewards
