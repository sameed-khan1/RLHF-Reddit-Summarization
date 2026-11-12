"""
train_reward.py – Stage 2: Reward Model Training (Task 3).

Implements Bradley-Terry preference learning exactly as specified:
  Loss = -E[ log σ( r(x, y_w) - r(x, y_l) ) ]

Training settings (from assignment):
  - AdamW, lr = 1e-5
  - effective batch size = 8 pairs  (micro_batch=2, accumulate 4 steps)
  - 1 epoch over 20,000 preference pairs
  - Log pairwise accuracy every 250 optimiser steps on 1,000 val pairs
  - Save to checkpoints/reward/

Memory-saving techniq
ues (for 6 GB VRAM):
  - bfloat16 autocast
  - gradient checkpointing on the backbone
  - micro-batch accumulation

Run from project root:
  python starter/train_reward.py
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

# ── allow imports from starter/ when running from project root ────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from config import SEED
from models import load_tokenizer
from data import load_preference_splits
from batching import collate_pref
from reward_model import RewardModel


# ─────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────

def bradley_terry_loss(chosen_rewards, rejected_rewards):
    """L = -E[ log σ( r(x,y_w) - r(x,y_l) ) ]  (Eq. 40)"""
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    """Mean BT-loss and pairwise accuracy on the validation loader."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for batch in loader:
        chosen_ids  = batch["chosen_ids"].to(device)
        chosen_mask = batch["chosen_mask"].to(device)
        rej_ids     = batch["rejected_ids"].to(device)
        rej_mask    = batch["rejected_mask"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            r_ch = model(chosen_ids,  chosen_mask)
            r_rj = model(rej_ids,     rej_mask)
            loss = bradley_terry_loss(r_ch, r_rj)

        n = chosen_ids.size(0)
        total_loss    += loss.item() * n
        total_correct += (r_ch > r_rj).sum().item()
        total_n       += n

    model.train()
    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


# ─────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: {total_mem:.1f} GB")

    # ── Tokenizer & data ─────────────────────────────────────────────
    print("\nLoading tokenizer and preference datasets …")
    tok = load_tokenizer()
    train_data, val_data = load_preference_splits(tok)
    print(f"  train pairs : {len(train_data):,}")
    print(f"  val   pairs : {len(val_data):,}")

    pad_id = tok.pad_token_id

    # Effective batch = 8 pairs; micro-batch = 2 to fit in 6 GB VRAM
    MICRO_BATCH   = 2
    ACCUM_STEPS   = 4   # 2 × 4 = 8 effective

    train_loader = DataLoader(
        train_data,
        batch_size=MICRO_BATCH,
        shuffle=True,
        collate_fn=partial(collate_pref, pad_id=pad_id),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=MICRO_BATCH,
        shuffle=False,
        collate_fn=partial(collate_pref, pad_id=pad_id),
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ────────────────────────────────────────────────────────
    sft_path = os.path.join("checkpoints", "sft", "best_model.pth")
    print(f"\nLoading reward model from SFT checkpoint: {sft_path}")
    model = RewardModel(sft_path, device)

    # Enable gradient checkpointing to save VRAM
    model.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.backbone.config.use_cache = False
    model.train()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── Optimizer (spec: AdamW, lr=1e-5) ────────────────────────────
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

    os.makedirs("checkpoints/reward", exist_ok=True)
    os.makedirs("results/training_logs", exist_ok=True)

    # ── Training ─────────────────────────────────────────────────────
    print(f"\n========== REWARD MODEL TRAINING ==========")
    print(f"Settings: lr=1e-5, effective_batch=8 (micro=2, accum={ACCUM_STEPS}), 1 epoch")

    best_val_acc  = 0.0
    best_val_loss = float("inf")
    logs          = []

    # Counts for running metrics
    opt_step        = 0          # number of optimiser.step() calls
    running_loss    = 0.0
    running_correct = 0
    running_n       = 0

    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for micro_step, batch in enumerate(train_loader, start=1):

        chosen_ids  = batch["chosen_ids"].to(device)
        chosen_mask = batch["chosen_mask"].to(device)
        rej_ids     = batch["rejected_ids"].to(device)
        rej_mask    = batch["rejected_mask"].to(device)
        n = chosen_ids.size(0)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            r_ch = model(chosen_ids, chosen_mask)
            r_rj = model(rej_ids,    rej_mask)
            loss = bradley_terry_loss(r_ch, r_rj) / ACCUM_STEPS  # scale for accum

        loss.backward()

        with torch.no_grad():
            running_loss    += loss.item() * ACCUM_STEPS * n   # unscale for logging
            running_correct += (r_ch > r_rj).sum().item()
            running_n       += n

        # ── Optimiser step every ACCUM_STEPS micro-batches ───────────
        if micro_step % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1

            # ── Log every 250 optimiser steps ────────────────────────
            if opt_step % 250 == 0:
                avg_train_loss = running_loss / max(running_n, 1)
                train_acc      = running_correct / max(running_n, 1)

                val_loss, val_acc = evaluate(model, val_loader, device)
                elapsed = time.time() - t0

                print(
                    f"OptStep {opt_step:4d} | "
                    f"train_loss={avg_train_loss:.4f} | "
                    f"train_acc={train_acc*100:.2f}% | "
                    f"val_loss={val_loss:.4f} | "
                    f"val_acc={val_acc*100:.2f}% | "
                    f"t={elapsed:.0f}s"
                )

                logs.append(dict(
                    opt_step=opt_step,
                    train_loss=avg_train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    elapsed=elapsed,
                ))

                if val_acc > best_val_acc:
                    best_val_acc  = val_acc
                    best_val_loss = val_loss
                    torch.save(model.state_dict(),
                               "checkpoints/reward/best_model.pth")
                    print(f"  >> New best val_acc={val_acc*100:.2f}% -- model saved.")

                model.train()

    # Handle any remaining micro-batches that didn't complete an accum cycle
    if micro_step % ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # ── Final evaluation ─────────────────────────────────────────────
    val_loss, val_acc = evaluate(model, val_loader, device)
    print(f"\nFinal val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}%")

    torch.save(model.state_dict(), "checkpoints/reward/last_model.pth")
    if not os.path.exists("checkpoints/reward/best_model.pth"):
        torch.save(model.state_dict(), "checkpoints/reward/best_model.pth")

    with open("results/training_logs/reward_training.json", "w") as f:
        json.dump(logs, f, indent=2)

    print("\n========== REWARD MODEL TRAINING COMPLETE ==========")
    print(f"Best val pairwise accuracy : {best_val_acc*100:.2f}%")
    print(f"Best val loss              : {best_val_loss:.4f}")
    print("Checkpoint saved to        : checkpoints/reward/best_model.pth")

    return best_val_acc


if __name__ == "__main__":
    main()
