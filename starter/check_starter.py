
from functools import partial

from torch.utils.data import DataLoader

from models import load_tokenizer
from data import (
    load_sft_splits,
    load_preference_splits,
    load_eval_posts
)
from batching import (
    collate_sft,
    collate_pref,
    collate_prompts
)


def print_statistics(name, values):
    print(f"\n{name}")
    print(f"Mean: {sum(values) / len(values):.2f}")
    print(f"Maximum: {max(values)}")


def main():
    print("Loading tokenizer...")

    tok = load_tokenizer()

    print("\nLoading SFT dataset...")
    sft_train, sft_val = load_sft_splits(tok)

    print("\nLoading preference dataset...")
    pref_train, pref_val = load_preference_splits(tok)

    print("\nLoading evaluation dataset...")
    eval_posts = load_eval_posts(tok)

    print("\n========== DATASET SIZES ==========")

    print("SFT train:", len(sft_train))
    print("SFT validation:", len(sft_val))

    print("Preference train:", len(pref_train))
    print("Preference validation:", len(pref_val))

    print("Evaluation posts:", len(eval_posts))

    if not sft_train:
        raise RuntimeError("SFT training subset is empty.")

    if not pref_train:
        raise RuntimeError("Preference training subset is empty.")

    print("\n========== SFT TOKEN STATISTICS ==========")

    prompt_lengths = [
        len(item["prompt_ids"])
        for item in sft_train
    ]

    summary_lengths = [
        len(item["summary_ids"])
        for item in sft_train
    ]

    print_statistics("Prompt lengths", prompt_lengths)
    print_statistics("Summary lengths (including EOS)", summary_lengths)

    print("\n========== ONE SFT EXAMPLE ==========")

    example = sft_train[0]

    print(example["prompt"][-300:])
    print("SUMMARY:", repr(example["summary"]))

    pad_id = tok.pad_token_id

    print("\n========== SFT BATCH ==========")

    sft_loader = DataLoader(
        sft_train,
        batch_size=8,
        shuffle=True,
        collate_fn=partial(collate_sft, pad_id=pad_id)
    )

    batch = next(iter(sft_loader))

    print("Input shape:", tuple(batch["input_ids"].shape))
    print("Prompt lengths:", batch["prompt_lens"].tolist())

    print("\n========== PREFERENCE BATCH ==========")

    pref_loader = DataLoader(
        pref_train,
        batch_size=8,
        shuffle=True,
        collate_fn=partial(collate_pref, pad_id=pad_id)
    )

    batch = next(iter(pref_loader))

    print("Chosen shape:", tuple(batch["chosen_ids"].shape))
    print("Rejected shape:", tuple(batch["rejected_ids"].shape))

    print("\n========== GENERATION PROMPT BATCH ==========")

    eval_loader = DataLoader(
        eval_posts,
        batch_size=8,
        collate_fn=partial(collate_prompts, pad_id=pad_id)
    )

    batch = next(iter(eval_loader))

    print("Input shape:", tuple(batch["input_ids"].shape))
    print("Attention mask shape:", tuple(batch["attention_mask"].shape))
    print("Left-padded prompt batch created successfully.")

    print("\n========== STARTER CHECK COMPLETE ==========")


if __name__ == "__main__":
    main()
