
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models import load_tokenizer, load_causal_lm
from data import load_sft_splits
from batching import collate_sft
from common import token_logprobs


def main():
    print("Loading tokenizer and model...")

    tokenizer = load_tokenizer()
    model = load_causal_lm()

    model.eval()

    print("Loading one SFT example...")

    train_data, _ = load_sft_splits(tokenizer)

    batch = collate_sft(train_data[:2], tokenizer.pad_token_id)

    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    prompt_lens = batch["prompt_lens"].to(DEVICE)

    print("Calculating token log probabilities...")

    with torch.no_grad():
        logprobs, summary_mask = token_logprobs(
            model,
            input_ids,
            attention_mask,
            prompt_lens
        )

    print("\n========== TEST RESULTS ==========")

    print("Input shape:", tuple(input_ids.shape))
    print("Log probabilities shape:", tuple(logprobs.shape))
    print("Summary mask shape:", tuple(summary_mask.shape))

    print("\nSummary token counts:")

    for i in range(len(prompt_lens)):
        print(
            f"Example {i + 1}:",
            int(summary_mask[i].sum().item())
        )

    print("\nDecoded summary tokens:")

    for i in range(len(prompt_lens)):
        selected = torch.cat([
            input_ids[i, 1:][summary_mask[i].bool()],
            torch.tensor(
                [tokenizer.pad_token_id],
                device=DEVICE
            )
        ])[:-1]

        print(
            f"Example {i + 1}:",
            tokenizer.decode(selected.tolist())
        )

    assert logprobs.shape == (input_ids.size(0), input_ids.size(1) - 1)
    assert summary_mask.shape == logprobs.shape
    assert torch.all(summary_mask.sum(dim=1) > 0)

    print("\nALL TESTS PASSED!")


if __name__ == "__main__":
    main()
