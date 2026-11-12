"""
train_sft.py – Stage 1: Supervised Fine-Tuning (Task 2).

Trains SmolLM2-135M to generate TL;DR summaries from Reddit posts
by minimising the mean negative log-likelihood over summary tokens only:

  LSFT(θ) = -1/|summary| Σ_{t∈summary} log πθ(xt | x<t)

Settings (from assignment):
  AdamW, lr = 5e-5
  batch size = 16
  gradient-norm clipping = 1.0
  1 epoch over 20,000 SFT pairs
  Log train+val loss every 250 steps
  Save to checkpoints/sft/best_model.pth

Run from project root:
  python train_sft.py
"""

import os
import sys
import time
import json
from functools import partial

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

# ── allow imports from starter/ ──────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "starter"))

from config import SEED, MAX_SUMMARY_TOKENS
from models import load_tokenizer, load_causal_lm
from data import load_sft_splits, load_eval_posts
from batching import collate_sft, collate_prompts
from common import token_logprobs


# ─────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────

def sft_loss(logprobs, summary_mask):
    """Mean NLL over summary tokens."""
    return -(logprobs * summary_mask.float()).sum() / summary_mask.sum().clamp(min=1)


@torch.no_grad()
def evaluate_sft(model, loader, device):
    model.eval()
    total_nll    = 0.0
    total_tokens = 0

    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        pl   = batch["prompt_lens"].to(device)

        lp, sm = token_logprobs(model, ids, mask, pl)
        total_nll    -= (lp * sm.float()).sum().item()
        total_tokens += sm.sum().item()

    model.train()
    return total_nll / max(total_tokens, 1)


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

    print("\nLoading SFT datasets …")
    train_data, val_data = load_sft_splits(tok)
    print(f"  train: {len(train_data):,}  val: {len(val_data):,}")

    # Report dataset stats (Task 1.2)
    prompt_lens  = [len(d["prompt_ids"])  for d in train_data]
    summary_lens = [len(d["summary_ids"]) for d in train_data]
    print(f"  SFT train prompt  length — mean={sum(prompt_lens)/len(prompt_lens):.1f} max={max(prompt_lens)}")
    print(f"  SFT train summary length — mean={sum(summary_lens)/len(summary_lens):.1f} max={max(summary_lens)}")

    train_loader = DataLoader(
        train_data,
        batch_size=16,
        shuffle=True,
        collate_fn=partial(collate_sft, pad_id=pad_id),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=8,
        shuffle=False,
        collate_fn=partial(collate_sft, pad_id=pad_id),
        pin_memory=(device.type == "cuda"),
    )

    print("\nLoading base model …")
    model = load_causal_lm(device=device)
    model.config.use_cache = False
    model.train()

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    os.makedirs("checkpoints/sft", exist_ok=True)
    os.makedirs("results/training_logs", exist_ok=True)

    best_val_loss = float("inf")
    logs = []
    step = 0
    running_nll    = 0.0
    running_tokens = 0
    t0 = time.time()

    print(f"\n========== SFT TRAINING ==========")

    for batch in train_loader:
        step += 1

        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        pl   = batch["prompt_lens"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            lp, sm = token_logprobs(model, ids, mask, pl)
            loss   = sft_loss(lp, sm)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_nll    -= (lp.detach() * sm.float()).sum().item()
        running_tokens += sm.sum().item()

        if step % 250 == 0:
            avg_train_loss = running_nll / max(running_tokens, 1)
            val_loss       = evaluate_sft(model, val_loader, device)
            elapsed        = time.time() - t0

            print(
                f"Step {step:4d} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"t={elapsed:.0f}s"
            )

            logs.append(dict(step=step, train_loss=avg_train_loss, val_loss=val_loss, elapsed=elapsed))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "checkpoints/sft/best_model.pth")
                print(f"  ✓ New best val_loss={val_loss:.4f} — model saved.")

    # Final save if never saved during training
    if not os.path.exists("checkpoints/sft/best_model.pth"):
        torch.save(model.state_dict(), "checkpoints/sft/best_model.pth")

    with open("results/training_logs/sft_training.json", "w") as f:
        json.dump(logs, f, indent=2)

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print("Checkpoint saved to: checkpoints/sft/best_model.pth")
    print("========== SFT TRAINING COMPLETE ==========")


if __name__ == "__main__":
    main()
