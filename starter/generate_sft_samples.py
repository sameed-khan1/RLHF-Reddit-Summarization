
import os
import gc
import json
import torch

from config import *
from models import load_tokenizer, load_causal_lm
from data import load_eval_posts
from batching import collate_prompts


def generate_summary(model, tokenizer, item, device):

    batch = collate_prompts(
        [item],
        tokenizer.pad_token_id
    )

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.inference_mode():

        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_SUMMARY_TOKENS,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_tokens = output[0, input_ids.shape[1]:]

    summary = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return summary.strip()


def main():

    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    tokenizer = load_tokenizer()

    print("\nLoading evaluation posts...")

    eval_data = load_eval_posts(tokenizer)[:5]

    print("Number of examples:", len(eval_data))

    results = []

    print("\n========== ORIGINAL MODEL ==========")

    base_model = load_causal_lm(device=device)
    base_model.eval()

    for i, item in enumerate(eval_data, start=1):

        summary = generate_summary(
            base_model,
            tokenizer,
            item,
            device
        )

        results.append({
            "example": i,
            "prompt": item["prompt"],
            "reference_summary": item["summary"],
            "original_summary": summary
        })

        print(f"\nExample {i}")
        print("Original summary:", summary)

    del base_model
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\n========== SFT MODEL ==========")

    sft_model = load_causal_lm(device=device)

    checkpoint_path = "checkpoints/sft/best_model.pth"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"SFT checkpoint not found: {checkpoint_path}"
        )

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    sft_model.load_state_dict(state_dict)

    sft_model.eval()

    for i, item in enumerate(eval_data):

        summary = generate_summary(
            sft_model,
            tokenizer,
            item,
            device
        )

        results[i]["sft_summary"] = summary

        print(f"\nExample {i + 1}")
        print("SFT summary:", summary)

    os.makedirs("results", exist_ok=True)

    output_path = "results/sft_comparison.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\n========== COMPARISON COMPLETE ==========")

    print("Total examples:", len(results))
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
