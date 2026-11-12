
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from config import *
from models import load_tokenizer, load_causal_lm
from data import load_sft_splits
from batching import collate_sft
from common import token_logprobs


def calculate_loss(model, batch):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    prompt_lens = batch["prompt_lens"]

    logprobs, summary_mask = token_logprobs(
        model,
        input_ids,
        attention_mask,
        prompt_lens
    )

    loss = -(logprobs.float() * summary_mask).sum()
    loss = loss / summary_mask.sum().clamp(min=1)

    return loss


def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():

        for batch in loader:

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda")
            ):

                logprobs, summary_mask = token_logprobs(
                    model,
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["prompt_lens"]
                )

            total_loss += (
                -(logprobs.float() * summary_mask).sum().item()
            )

            total_tokens += summary_mask.sum().item()

    model.train()

    return total_loss / max(total_tokens, 1)


def main():

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nLoading tokenizer...")

    tokenizer = load_tokenizer()

    print("\nLoading SFT datasets...")

    train_data, val_data = load_sft_splits(tokenizer)

    print("Training examples:", len(train_data))
    print("Validation examples:", len(val_data))

    train_loader = DataLoader(
        train_data,
        batch_size=16,
        shuffle=True,
        collate_fn=lambda batch: collate_sft(
            batch,
            tokenizer.pad_token_id
        )
    )

    val_loader = DataLoader(
        val_data,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda batch: collate_sft(
            batch,
            tokenizer.pad_token_id
        )
    )

    print("\nLoading language model...")

    model = load_causal_lm(device=device)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=5e-5
    )

    micro_batch_size = 2
    gradient_accumulation_steps = 8

    os.makedirs("checkpoints/sft", exist_ok=True)

    best_val_loss = float("inf")

    print("\n========== SFT TRAINING ==========")

    for epoch in range(1):

        model.train()

        optimizer.zero_grad(set_to_none=True)

        optimizer_steps = 0
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            batch_size = batch["input_ids"].size(0)

            accumulated_loss = 0.0

            for start in range(0, batch_size, micro_batch_size):

                micro_batch = {
                    key: value[start:start + micro_batch_size]
                    for key, value in batch.items()
                }

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=(device.type == "cuda")
                ):

                    loss = calculate_loss(
                        model,
                        micro_batch
                    )

                    scaled_loss = (
                        loss / gradient_accumulation_steps
                    )

                scaled_loss.backward()

                accumulated_loss += loss.detach().item()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            optimizer_steps += 1

            running_loss += (
                accumulated_loss / gradient_accumulation_steps
            )

            if optimizer_steps % 250 == 0:

                print(
                    f"Epoch {epoch + 1} | "
                    f"Step {optimizer_steps} | "
                    f"Training Loss: "
                    f"{running_loss / optimizer_steps:.4f}"
                )

        train_loss = running_loss / max(optimizer_steps, 1)

        print("\nRunning validation...")

        val_loss = evaluate(
            model,
            val_loader,
            device
        )

        print(f"\nEpoch {epoch + 1} Results")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                "checkpoints/sft/best_model.pth"
            )

            print("Best SFT model saved.")

    print("\n========== SFT COMPLETE ==========")

    print("Best validation loss:", best_val_loss)

    print(
        "Checkpoint: checkpoints/sft/best_model.pth"
    )


if __name__ == "__main__":
    main()
