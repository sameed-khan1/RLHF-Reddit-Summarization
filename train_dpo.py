"""
train_dpo.py – Stage 4: Direct Preference Optimisation (Task 5).

Implements DPO from scratch in PyTorch:
  LDPO(θ) = -E[ log σ(β * log[πθ(yw|x)/πref(yw|x)]
                        - β * log[πθ(yl|x)/πref(yl|x)]) ]

where log π(y|x) = Σ_{t∈summary} log πθ(yt|x,y<t)

Settings (from assignment):
  β = 0.1
  AdamW, lr = 2e-6
  batch size = 8 pairs
  1 epoch over 20,000 preference pairs
  Log every 250 steps: loss + implicit-reward accuracy on 1,000 val pairs
  Save to checkpoints/dpo/

Run from project root:
  python train_dpo.py
"""

import os
import sys
import time
import json
from functools import partial

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

# ── allow imports from starter/ ──────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "starter"))

from config import SEED
from models import load_tokenizer, load_causal_lm
from data import load_preference_splits
from batching import collate_pref
from common import token_logprobs


# ─────────────────────────────────────────────────────────────────────
# DPO loss
# ─────────────────────────────────────────────────────────────────────

def dpo_loss(
    log_pi_chosen:   torch.Tensor,  # (B,) sum of log-probs over summary tokens
    log_pi_rejected: torch.Tensor,  # (B,)
    log_ref_chosen:  torch.Tensor,  # (B,)
    log_ref_rejected:torch.Tensor,  # (B,)
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (loss, implicit_reward_margin).
    implicit_reward_margin[b] > 0  means chosen is preferred.
    """
    # Implicit log-reward = β * log[π(y|x) / π_ref(y|x)]
    implicit_chosen   = beta * (log_pi_chosen   - log_ref_chosen)
    implicit_rejected = beta * (log_pi_rejected - log_ref_rejected)
    margin = implicit_chosen - implicit_rejected
    loss   = -F.logsigmoid(margin).mean()
    return loss, margin


def sequence_logprob(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
    device_type: str = "cuda",
) -> torch.Tensor:
    """
    Returns the SUM of log-probs over summary tokens for each sequence.
    Shape: (B,)
    Uses bfloat16 autocast to reduce VRAM usage.
    """
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                        enabled=(device_type == "cuda")):
        logprobs, summary_mask = token_logprobs(
            model, input_ids, attention_mask, prompt_lens
        )
    # Sum over summary positions (not mean — matching the formula)
    return (logprobs.float() * summary_mask.float()).sum(dim=1)  # (B,)


# ─────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_dpo(policy, ref_model, loader, device, beta=0.1):
    policy.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    dt = device.type

    for batch in loader:
        chosen_ids  = batch["chosen_ids"].to(device)
        chosen_mask = batch["chosen_mask"].to(device)
        rej_ids     = batch["rejected_ids"].to(device)
        rej_mask    = batch["rejected_mask"].to(device)
        prompt_lens = batch["prompt_lens"].to(device)
        n = chosen_ids.size(0)

        # Policy log-probs
        lp_pi_ch  = sequence_logprob(policy,    chosen_ids,  chosen_mask,  prompt_lens, dt)
        lp_pi_rj  = sequence_logprob(policy,    rej_ids,     rej_mask,     prompt_lens, dt)
        # Reference log-probs
        lp_ref_ch = sequence_logprob(ref_model, chosen_ids,  chosen_mask,  prompt_lens, dt)
        lp_ref_rj = sequence_logprob(ref_model, rej_ids,     rej_mask,     prompt_lens, dt)

        loss, margin = dpo_loss(lp_pi_ch, lp_pi_rj, lp_ref_ch, lp_ref_rj, beta)

        total_loss    += loss.item() * n
        total_correct += (margin > 0).sum().item()
        total_n       += n

    policy.train()
    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


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

    tok = load_tokenizer()
    pad_id = tok.pad_token_id

    print("\nLoading preference datasets …")
    train_data, val_data = load_preference_splits(tok)
    print(f"  train: {len(train_data):,}  val: {len(val_data):,}")

    train_loader = DataLoader(
        train_data,
        batch_size=2,          # micro-batch; accumulate 4 → effective 8
        shuffle=True,
        collate_fn=partial(collate_pref, pad_id=pad_id),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=2,
        shuffle=False,
        collate_fn=partial(collate_pref, pad_id=pad_id),
        pin_memory=(device.type == "cuda"),
    )

    sft_path = os.path.join("checkpoints", "sft", "best_model.pth")
    sft_state = torch.load(sft_path, map_location=device, weights_only=True)

    # ── Trainable policy πθ ──────────────────────────────────────────
    print("\nLoading trainable policy from SFT …")
    policy = load_causal_lm(device=device)
    policy.load_state_dict(sft_state, strict=True)
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    policy.train()

    # ── Frozen reference πref ────────────────────────────────────────
    print("Loading frozen reference from SFT …")
    ref_model = load_causal_lm(device=device)
    ref_model.load_state_dict(sft_state, strict=True)
    ref_model.config.use_cache = False
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    dt = device.type  # shorthand for autocast device_type arg

    # Verify correctness: first step loss should equal log(2) ≈ 0.6931
    print("\nRunning DPO first-step sanity check (expect loss ≈ 0.6931) …")
    with torch.no_grad():
        sample = next(iter(train_loader))
        ch_ids  = sample["chosen_ids"][:2].to(device)
        ch_mask = sample["chosen_mask"][:2].to(device)
        rj_ids  = sample["rejected_ids"][:2].to(device)
        rj_mask = sample["rejected_mask"][:2].to(device)
        pl      = sample["prompt_lens"][:2].to(device)

        lp_pi_ch  = sequence_logprob(policy,    ch_ids,  ch_mask,  pl, dt)
        lp_pi_rj  = sequence_logprob(policy,    rj_ids,  rj_mask,  pl, dt)
        lp_ref_ch = sequence_logprob(ref_model, ch_ids,  ch_mask,  pl, dt)
        lp_ref_rj = sequence_logprob(ref_model, rj_ids,  rj_mask,  pl, dt)
        loss0, _ = dpo_loss(lp_pi_ch, lp_pi_rj, lp_ref_ch, lp_ref_rj, beta=0.1)
        print(f"  First-step loss = {loss0.item():.6f}  (expected ≈ 0.6931)")

    # ── Optimizer ────────────────────────────────────────────────────
    optimizer = AdamW(policy.parameters(), lr=2e-6, weight_decay=0.01)

    ACCUM_STEPS = 4  # effective batch = 2 × 4 = 8
    beta = 0.1
    os.makedirs("checkpoints/dpo", exist_ok=True)
    os.makedirs("results/training_logs", exist_ok=True)

    best_val_acc = 0.0
    logs = []
    micro_step   = 0
    opt_step     = 0
    running_loss = 0.0
    running_correct = 0
    running_n = 0
    t0 = time.time()

    print(f"\n========== DPO TRAINING ==========")
    print(f"Settings: β=0.1, lr=2e-6, micro_batch=2, accum={ACCUM_STEPS} → eff_batch=8, 1 epoch")

    optimizer.zero_grad(set_to_none=True)

    for batch in train_loader:
        micro_step += 1

        chosen_ids  = batch["chosen_ids"].to(device)
        chosen_mask = batch["chosen_mask"].to(device)
        rej_ids     = batch["rejected_ids"].to(device)
        rej_mask    = batch["rejected_mask"].to(device)
        prompt_lens = batch["prompt_lens"].to(device)
        n = chosen_ids.size(0)

        # Policy log-probs
        lp_pi_ch = sequence_logprob(policy, chosen_ids, chosen_mask, prompt_lens, dt)
        lp_pi_rj = sequence_logprob(policy, rej_ids,    rej_mask,    prompt_lens, dt)

        # Reference log-probs (no grad)
        with torch.no_grad():
            lp_ref_ch = sequence_logprob(ref_model, chosen_ids, chosen_mask, prompt_lens, dt)
            lp_ref_rj = sequence_logprob(ref_model, rej_ids,    rej_mask,    prompt_lens, dt)

        loss, margin = dpo_loss(lp_pi_ch, lp_pi_rj, lp_ref_ch, lp_ref_rj, beta)

        loss_scaled = loss / ACCUM_STEPS
        loss_scaled.backward()

        running_loss    += loss.item() * n
        running_correct += (margin.detach() > 0).sum().item()
        running_n       += n

        if micro_step % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1

            if opt_step % 250 == 0:
                avg_loss    = running_loss / max(running_n, 1)
                train_acc   = running_correct / max(running_n, 1)
                val_loss, val_acc = evaluate_dpo(policy, ref_model, val_loader, device, beta)
                elapsed = time.time() - t0

                print(
                    f"OptStep {opt_step:4d} | "
                    f"train_loss={avg_loss:.4f} | "
                    f"train_acc={train_acc*100:.2f}% | "
                    f"val_loss={val_loss:.4f} | "
                    f"val_acc={val_acc*100:.2f}% | "
                    f"t={elapsed:.0f}s"
                )

                logs.append(dict(
                    opt_step=opt_step, train_loss=avg_loss, train_acc=train_acc,
                    val_loss=val_loss, val_acc=val_acc, elapsed=elapsed,
                ))

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(policy.state_dict(), "checkpoints/dpo/best_model.pth")
                    print(f"  >> New best val_acc={val_acc*100:.2f}% -- model saved.")

    # Handle any trailing micro-batches
    if micro_step % ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Final save
    torch.save(policy.state_dict(), "checkpoints/dpo/last_model.pth")
    if not os.path.exists("checkpoints/dpo/best_model.pth"):
        torch.save(policy.state_dict(), "checkpoints/dpo/best_model.pth")

    with open("results/training_logs/dpo_training.json", "w") as f:
        json.dump(logs, f, indent=2)

    print(f"\nBest val implicit-reward accuracy: {best_val_acc*100:.2f}%")
    print("Checkpoint saved to: checkpoints/dpo/best_model.pth")
    print("========== DPO TRAINING COMPLETE ==========")


if __name__ == "__main__":
    main()
