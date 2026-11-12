"""
train_ppo.py – Stage 3: PPO fine-tuning (Task 4).

Implements PPO with KL-penalty from scratch in PyTorch, following the
lecture notes (Sections 2–4).  No RLHF library is used.

Models in GPU memory:
  policy πθ + value head Vθ   (trainable, starts from SFT)
  reference πref               (frozen SFT copy)
  reward model rϕ              (frozen, from Stage 2)

Settings (must match the assignment):
  clip ε = 0.2          KL coeff βKL = 0.05
  value coeff cV = 0.1  entropy coeff cH = 0.01
  discount γ = 1.0      PPO epochs K = 4 per rollout
  AdamW lr = 1e-5       grad-norm clip = 1.0
  prompts per rollout = 16    total iterations = 200

Memory saving (for 6 GB VRAM):
  - bfloat16 autocast during forward passes
  - gradient checkpointing on policy backbone
  - rollout processed in two halves (B//2) during PPO update epochs

Run from project root:
  python train_ppo.py
"""

import os
import sys
import time
import json
import random
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

# ── allow imports from starter/ ──────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "starter"))

from config import SEED, MAX_SUMMARY_TOKENS
from models import load_tokenizer, load_causal_lm
from data import load_sft_splits
from batching import collate_prompts
from reward_model import RewardModel
from common import token_logprobs


# ─────────────────────────────────────────────────────────────────────
# Value head wrapper
# ─────────────────────────────────────────────────────────────────────

class PolicyWithValueHead(nn.Module):
    """
    Wraps AutoModelForCausalLM and adds a linear value head.
    The value head is stripped when exporting the policy checkpoint.
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model
        hidden_size = base_model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1, bias=True)
        nn.init.normal_(self.value_head.weight, std=0.02)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits        = outputs.logits                   # (B, L, V)
        hidden_states = outputs.hidden_states[-1]        # (B, L, H)
        values        = self.value_head(hidden_states.float()).squeeze(-1)  # (B, L)
        return logits, values

    def generate(self, **kwargs):
        return self.base.generate(**kwargs)


# ─────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────

def logits_to_logprobs(logits, token_ids):
    """(B,L,V),(B,L) → (B,L)  log-probs of the given tokens."""
    return F.log_softmax(logits, dim=-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)


def entropy_from_logits(logits):
    """(B,L,V) → (B,L)  token-level entropy."""
    p = F.softmax(logits, dim=-1)
    return -(p * F.log_softmax(logits, dim=-1)).sum(dim=-1)


@torch.no_grad()
def score_summaries(reward_model, prompt_ids_list, summary_ids_list, pad_id, device):
    """Score a batch of (prompt, summary) pairs with the reward model."""
    seqs = [p + s for p, s in zip(prompt_ids_list, summary_ids_list)]
    max_len = max(len(s) for s in seqs)
    B = len(seqs)
    ids  = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for i, seq in enumerate(seqs):
        ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        mask[i, :len(seq)] = 1
    return reward_model(ids, mask)   # (B,)


# ─────────────────────────────────────────────────────────────────────
# PPO update: process one half of the rollout at a time to save VRAM
# ─────────────────────────────────────────────────────────────────────

def ppo_update_half(
    policy, full_ids, full_mask, sum_mask,
    logprobs_old, returns, advantages,
    CLIP_EPS, C_VALUE, C_ENTROPY, device,
):
    """
    Forward pass + loss for one (B,L) slice of the rollout.
    Returns (ppo_loss, value_loss, entropy_loss, ratio, H) — all tensors.
    Does NOT call backward.
    """
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=(device.type == "cuda")):
        logits_new, values_new_full = policy(full_ids, full_mask)
    logprobs_new = logits_to_logprobs(logits_new[:, :-1], full_ids[:, 1:])
    values_new   = values_new_full[:, :-1]
    H            = entropy_from_logits(logits_new[:, :-1])

    sm_f = sum_mask.float()
    n_tokens = sm_f.sum().clamp(min=1)

    ratio  = torch.exp((logprobs_new - logprobs_old.detach()) * sm_f)
    surr1  = ratio * advantages.detach()
    surr2  = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages.detach()
    ppo_loss    = -(torch.min(surr1, surr2) * sm_f).sum() / n_tokens
    value_loss  = (F.mse_loss(values_new * sm_f, returns.detach() * sm_f,
                              reduction="sum") / n_tokens)
    entropy_loss = -(H * sm_f).sum() / n_tokens

    return ppo_loss, value_loss, entropy_loss, ratio.detach(), H.detach()


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tok    = load_tokenizer()
    pad_id = tok.pad_token_id

    print("\nLoading SFT training data …")
    sft_train, _ = load_sft_splits(tok)
    print(f"  SFT train examples: {len(sft_train):,}")

    # ── Load models ──────────────────────────────────────────────────
    sft_path    = os.path.join("checkpoints", "sft",    "best_model.pth")
    reward_path = os.path.join("checkpoints", "reward", "best_model.pth")

    sft_state = torch.load(sft_path, map_location=device, weights_only=True)

    print("\nLoading policy (SFT checkpoint) …")
    policy_base = load_causal_lm(device=device)
    policy_base.load_state_dict(sft_state, strict=True)
    policy_base.config.use_cache = False
    policy_base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    policy = PolicyWithValueHead(policy_base).to(device)
    policy.train()

    print("Loading frozen reference model (SFT checkpoint) …")
    ref_model = load_causal_lm(device=device)
    ref_model.load_state_dict(sft_state, strict=True)
    ref_model.config.use_cache = False
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    print("Loading frozen reward model …")
    reward_model = RewardModel(reward_path, device)
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad_(False)

    # ── PPO hyper-parameters (fixed by assignment) ───────────────────
    CLIP_EPS  = 0.2
    BETA_KL   = 0.05
    C_VALUE   = 0.1
    C_ENTROPY = 0.01
    GAMMA     = 1.0
    K_EPOCHS  = 4
    LR        = 1e-5
    MAX_NORM  = 1.0
    PROMPTS_PER_ROLLOUT = 16
    N_ITERATIONS        = 200

    optimizer = AdamW(policy.parameters(), lr=LR, weight_decay=0.01)

    os.makedirs("checkpoints/ppo", exist_ok=True)
    os.makedirs("results/training_logs", exist_ok=True)

    all_logs = []
    rng      = random.Random(SEED)
    sft_idx  = list(range(len(sft_train)))

    print(f"\n========== PPO TRAINING ({N_ITERATIONS} iters, B={PROMPTS_PER_ROLLOUT}) ==========")
    t0 = time.time()

    for iteration in range(1, N_ITERATIONS + 1):

        # ──────────────────────────────────────────────────────────────
        # 4.1  Rollout
        # ──────────────────────────────────────────────────────────────
        idxs    = rng.sample(sft_idx, PROMPTS_PER_ROLLOUT)
        prompts = [sft_train[i] for i in idxs]

        prompt_batch     = collate_prompts(prompts, pad_id)
        prompt_input_ids = prompt_batch["input_ids"].to(device)
        prompt_attn_mask = prompt_batch["attention_mask"].to(device)

        policy.eval()
        with torch.no_grad():
            generated = policy.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attn_mask,
                max_new_tokens=MAX_SUMMARY_TOKENS,
                do_sample=True,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                pad_token_id=pad_id,
                eos_token_id=tok.eos_token_id,
            )

        padded_prompt_len = prompt_input_ids.size(1)
        gen_tokens = generated[:, padded_prompt_len:]  # (B, T_max)
        B = gen_tokens.size(0)

        # Truncate each row at first EOS
        summary_ids_list = []
        summary_lens     = []
        for b in range(B):
            row     = gen_tokens[b]
            eos_pos = (row == tok.eos_token_id).nonzero(as_tuple=True)[0]
            end     = eos_pos[0].item() + 1 if len(eos_pos) > 0 else row.size(0)
            summary_ids_list.append(row[:end].tolist())
            summary_lens.append(end)

        prompt_ids_list = [prompts[b]["prompt_ids"] for b in range(B)]

        # ──────────────────────────────────────────────────────────────
        # 4.2  Build right-padded (prompt+summary) batch; store old vals
        # ──────────────────────────────────────────────────────────────
        full_seqs, sum_starts = [], []
        for b in range(B):
            seq = prompt_ids_list[b] + summary_ids_list[b]
            full_seqs.append(seq)
            sum_starts.append(len(prompt_ids_list[b]))

        max_full_len = max(len(s) for s in full_seqs)
        full_ids  = torch.full((B, max_full_len), pad_id,
                               dtype=torch.long, device=device)
        full_mask = torch.zeros((B, max_full_len), dtype=torch.long, device=device)
        for b, seq in enumerate(full_seqs):
            full_ids[b, :len(seq)]  = torch.tensor(seq, dtype=torch.long, device=device)
            full_mask[b, :len(seq)] = 1

        plen_t = torch.tensor(sum_starts, dtype=torch.long, device=device)

        L_m1 = max_full_len - 1
        pos   = torch.arange(1, max_full_len, device=device).unsqueeze(0)
        sum_mask = (pos >= plen_t.unsqueeze(1)).long() * full_mask[:, 1:]  # (B, L-1)

        with torch.no_grad():
            # Policy old: logprobs + values
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                logits_old, values_old_full = policy(full_ids, full_mask)
            logprobs_old = logits_to_logprobs(logits_old[:, :-1], full_ids[:, 1:])
            values_old   = values_old_full[:, :-1]

            # Reference logprobs
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                ref_out = ref_model(input_ids=full_ids, attention_mask=full_mask)
            logprobs_ref = logits_to_logprobs(ref_out.logits[:, :-1], full_ids[:, 1:])

            # Reward scores
            rm_scores = score_summaries(
                reward_model, prompt_ids_list, summary_ids_list, pad_id, device
            )

        # ──────────────────────────────────────────────────────────────
        # 4.3  Per-token reward
        # ──────────────────────────────────────────────────────────────
        kl_per_token     = (logprobs_old - logprobs_ref) * sum_mask.float()
        per_token_reward = -BETA_KL * kl_per_token.clone()
        for b in range(B):
            last_idx = sum_starts[b] + summary_lens[b] - 1
            if last_idx < L_m1:
                per_token_reward[b, last_idx] += rm_scores[b].item()

        # ──────────────────────────────────────────────────────────────
        # 4.4  Returns and advantages  (γ = 1 → cumsum from right)
        # ──────────────────────────────────────────────────────────────
        returns    = torch.zeros_like(per_token_reward)
        running    = torch.zeros(B, device=device)
        for t in reversed(range(L_m1)):
            m_t     = sum_mask[:, t].float()
            running = per_token_reward[:, t] + GAMMA * running * m_t
            returns[:, t] = running * m_t

        advantages = (returns - values_old.detach()) * sum_mask.float()
        valid_adv  = advantages[sum_mask.bool()]
        if valid_adv.numel() > 1:
            advantages = (advantages - valid_adv.mean()) / valid_adv.std().clamp(min=1e-8)
        advantages = advantages * sum_mask.float()

        # ──────────────────────────────────────────────────────────────
        # 4.5  PPO update (K epochs, process in two halves to save VRAM)
        # ──────────────────────────────────────────────────────────────
        half = B // 2
        policy.train()

        final_ppo_loss = final_val_loss = final_ent_loss = 0.0
        final_ratio = None

        for ppo_epoch in range(K_EPOCHS):
            optimizer.zero_grad(set_to_none=True)

            total_ppo = total_val = total_ent = 0.0
            all_ratios = []

            for h_start in [0, half]:
                h_end   = h_start + half
                sl      = slice(h_start, h_end)

                pl, vl, el, ratio_h, _ = ppo_update_half(
                    policy,
                    full_ids[sl],  full_mask[sl],  sum_mask[sl],
                    logprobs_old[sl], returns[sl],  advantages[sl],
                    CLIP_EPS, C_VALUE, C_ENTROPY, device,
                )
                n_tok_h = sum_mask[sl].float().sum().clamp(min=1)
                n_tok_B = sum_mask.float().sum().clamp(min=1)
                scale   = n_tok_h / n_tok_B   # weight by token count

                loss = (pl + C_VALUE * vl + C_ENTROPY * el) * scale
                loss.backward()

                total_ppo += pl.item() * scale.item()
                total_val += vl.item() * scale.item()
                total_ent += el.item() * scale.item()
                all_ratios.append(ratio_h)

            torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_NORM)
            optimizer.step()

            final_ppo_loss = total_ppo
            final_val_loss = total_val
            final_ent_loss = total_ent
            final_ratio    = torch.cat(all_ratios, dim=0)   # (B, L-1)

        # ──────────────────────────────────────────────────────────────
        # 4.6  Logging
        # ──────────────────────────────────────────────────────────────
        with torch.no_grad():
            mean_rm_score  = rm_scores.mean().item()
            kl_per_example = kl_per_token.sum(dim=1) / sum_mask.float().sum(dim=1).clamp(min=1)
            mean_kl        = kl_per_example.mean().item()
            mean_len       = float(sum(summary_lens)) / B

            clipped   = ((final_ratio < 1 - CLIP_EPS) | (final_ratio > 1 + CLIP_EPS)) & sum_mask.bool()
            clip_frac = clipped.sum().float().item() / sum_mask.sum().float().clamp(min=1).item()

        log = dict(
            iteration=iteration,
            mean_rm_score=mean_rm_score,
            mean_kl=mean_kl,
            mean_len=mean_len,
            clip_frac=clip_frac,
            ppo_loss=final_ppo_loss,
            value_loss=final_val_loss,
            entropy_loss=final_ent_loss,
            elapsed=time.time() - t0,
        )
        all_logs.append(log)

        print(
            f"Iter {iteration:3d} | "
            f"rm={mean_rm_score:+.3f} | "
            f"kl={mean_kl:.4f} | "
            f"len={mean_len:.1f} | "
            f"clip={clip_frac:.3f} | "
            f"ppo={final_ppo_loss:.4f} | "
            f"val={final_val_loss:.4f} | "
            f"ent={final_ent_loss:.4f}"
        )

    # ──────────────────────────────────────────────────────────────────
    # 4.7  Save policy (without value head)
    # ──────────────────────────────────────────────────────────────────
    os.makedirs("checkpoints/ppo", exist_ok=True)
    torch.save(policy.base.state_dict(), "checkpoints/ppo/best_model.pth")
    print("\nPPO policy saved to: checkpoints/ppo/best_model.pth")

    with open("results/training_logs/ppo_training.json", "w") as f:
        json.dump(all_logs, f, indent=2)

    print("========== PPO TRAINING COMPLETE ==========")


if __name__ == "__main__":
    main()
