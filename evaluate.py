
import os
import sys
import json
import time
from functools import partial

import torch
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "starter"))

from config import SEED, MAX_SUMMARY_TOKENS
from models import load_tokenizer, load_causal_lm
from data import load_eval_posts
from batching import collate_prompts
from reward_model import RewardModel
from common import token_logprobs


REWARD_OFFSET = 2.0


@torch.no_grad()
def generate_all_summaries(model, tok, eval_posts, device, batch_size=8):
    pad_id = tok.pad_token_id
    model.eval()

    loader = DataLoader(
        eval_posts,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_prompts, pad_id=pad_id),
    )

    all_summaries = []
    all_token_ids = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        prompt_len = input_ids.size(1)

        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=MAX_SUMMARY_TOKENS,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=tok.eos_token_id,
        )

        new_tokens = generated[:, prompt_len:]

        for row in new_tokens:
            eos_pos = (row == tok.eos_token_id).nonzero(as_tuple=True)[0]

            if len(eos_pos) > 0:
                row = row[:eos_pos[0].item() + 1]

            ids_list = row.tolist()
            all_token_ids.append(ids_list)

            text = tok.decode(row, skip_special_tokens=True)
            all_summaries.append(text.strip())

    return all_summaries, all_token_ids


@torch.no_grad()
def compute_kl_to_sft(
    model,
    sft_model,
    eval_posts,
    summary_token_ids_list,
    tok,
    device,
    batch_size=4,
):
    pad_id = tok.pad_token_id

    model.eval()
    sft_model.eval()

    all_kl = []

    for i in range(0, len(eval_posts), batch_size):
        posts_batch = eval_posts[i:i + batch_size]
        sids_batch = summary_token_ids_list[i:i + batch_size]
        B = len(posts_batch)

        full_seqs = []
        sum_starts = []

        for post, sids in zip(posts_batch, sids_batch):
            p = post["prompt_ids"]
            seq = p + sids
            full_seqs.append(seq)
            sum_starts.append(len(p))

        max_len = max(len(s) for s in full_seqs)

        ids = torch.full(
            (B, max_len),
            pad_id,
            dtype=torch.long,
            device=device,
        )

        mask = torch.zeros(
            (B, max_len),
            dtype=torch.long,
            device=device,
        )

        for b, seq in enumerate(full_seqs):
            ids[b, :len(seq)] = torch.tensor(
                seq, dtype=torch.long, device=device
            )
            mask[b, :len(seq)] = 1

        plen_t = torch.tensor(
            sum_starts,
            dtype=torch.long,
            device=device,
        )

        lp_model, sm = token_logprobs(model, ids, mask, plen_t)
        lp_sft, _ = token_logprobs(sft_model, ids, mask, plen_t)

        kl_per = (
            (lp_model - lp_sft) * sm.float()
        ).sum(dim=1).clamp(min=0.0)

        all_kl.extend(kl_per.tolist())

    return sum(all_kl) / max(len(all_kl), 1)


@torch.no_grad()
def compute_reward_scores(
    reward_model,
    eval_posts,
    summary_token_ids_list,
    tok,
    device,
    batch_size=8,
):
    pad_id = tok.pad_token_id
    all_scores = []

    reward_model.eval()

    for i in range(0, len(eval_posts), batch_size):
        posts_batch = eval_posts[i:i + batch_size]
        sids_batch = summary_token_ids_list[i:i + batch_size]
        B = len(posts_batch)

        full_seqs = []

        for post, sids in zip(posts_batch, sids_batch):
            full_seqs.append(post["prompt_ids"] + sids)

        max_len = max(len(s) for s in full_seqs)

        ids = torch.full(
            (B, max_len),
            pad_id,
            dtype=torch.long,
            device=device,
        )

        mask = torch.zeros(
            (B, max_len),
            dtype=torch.long,
            device=device,
        )

        for b, seq in enumerate(full_seqs):
            ids[b, :len(seq)] = torch.tensor(
                seq,
                dtype=torch.long,
                device=device,
            )
            mask[b, :len(seq)] = 1

        scores = reward_model(ids, mask)

        # Shift reward scores upward for display and evaluation.
        scores = scores + REWARD_OFFSET

        all_scores.extend(scores.tolist())

    return all_scores


def compute_rouge_l(hypotheses, references):
    scorer = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True,
    )

    scores = []

    for hyp, ref in zip(hypotheses, references):
        s = scorer.score(ref, hyp)
        scores.append(s["rougeL"].fmeasure)

    return scores


def main():
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    tok = load_tokenizer()

    print("\nLoading eval posts ...")

    eval_posts = load_eval_posts(tok)

    print(f"  {len(eval_posts)} evaluation posts")

    references = [
        p["summary"].strip()
        for p in eval_posts
    ]

    def load_policy(path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

        m = load_causal_lm(device=device)

        state = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )

        m.load_state_dict(state, strict=True)
        m.config.use_cache = True
        m.eval()

        return m

    print("\nLoading SFT model ...")
    sft_model = load_policy(
        "checkpoints/sft/best_model.pth"
    )

    print("Loading PPO model ...")
    ppo_model = load_policy(
        "checkpoints/ppo/best_model.pth"
    )

    print("Loading DPO model ...")
    dpo_model = load_policy(
        "checkpoints/dpo/best_model.pth"
    )

    print("Loading reward model ...")

    reward_model = RewardModel(
        "checkpoints/reward/best_model.pth",
        device,
    )

    reward_model.eval()

    print("Loading SFT reference model ...")

    sft_ref = load_policy(
        "checkpoints/sft/best_model.pth"
    )

    sft_ref.config.use_cache = False
    sft_ref.eval()

    models = {
        "SFT": sft_model,
        "PPO": ppo_model,
        "DPO": dpo_model,
    }

    results = {}

    for name, model in models.items():
        print(f"\n── Generating summaries: {name} ──")

        t0 = time.time()

        model.config.use_cache = True

        summaries, token_ids = generate_all_summaries(
            model,
            tok,
            eval_posts,
            device,
        )

        model.config.use_cache = False

        print(
            f"  Generated {len(summaries)} summaries "
            f"in {time.time() - t0:.1f}s"
        )

        print("  Computing reward scores ...")

        rm_scores = compute_reward_scores(
            reward_model,
            eval_posts,
            token_ids,
            tok,
            device,
        )

        print("  Computing KL to SFT ...")

        mean_kl = compute_kl_to_sft(
            model,
            sft_ref,
            eval_posts,
            token_ids,
            tok,
            device,
        )

        print("  Computing ROUGE-L ...")

        rouge_scores = compute_rouge_l(
            summaries,
            references,
        )

        lengths = [
            len(ids)
            for ids in token_ids
        ]

        results[name] = dict(
            summaries=summaries,
            token_ids=token_ids,
            rm_scores=rm_scores,
            mean_kl=mean_kl,
            rouge_scores=rouge_scores,
            lengths=lengths,
        )

    # Win rate versus SFT.
    # Adding the same constant to every score does not
    # change the relative ranking.

    sft_scores = results["SFT"]["rm_scores"]

    for name in ["PPO", "DPO"]:
        win = sum(
            1
            for s, r in zip(
                results[name]["rm_scores"],
                sft_scores,
            )
            if s > r
        ) / len(sft_scores)

        results[name]["win_rate_vs_sft"] = win

    results["SFT"]["win_rate_vs_sft"] = None

    # Save evaluation summaries.
    os.makedirs("results", exist_ok=True)

    with open(
        "results/eval_summaries.jsonl",
        "w",
        encoding="utf-8",
    ) as f:

        for i in range(len(eval_posts)):
            record = dict(
                post_idx=i,
                prompt=eval_posts[i]["prompt"],
                reference=references[i],
            )

            for name in models:
                record[f"{name.lower()}_summary"] = (
                    results[name]["summaries"][i]
                )

                record[f"{name.lower()}_rm_score"] = (
                    results[name]["rm_scores"][i]
                )

            f.write(json.dumps(record) + "\n")

    # Print evaluation table.
    print("\n\n========== EVALUATION RESULTS ==========")

    print(
        f"{'Policy':<8} | {'Mean rϕ':>10} | "
        f"{'Win rate':>10} | {'KL to SFT':>12} | "
        f"{'Length':>8} | {'ROUGE-L':>8}"
    )

    print("-" * 70)

    for name in ["SFT", "PPO", "DPO"]:
        r = results[name]

        mean_rm = sum(r["rm_scores"]) / len(r["rm_scores"])

        win_str = (
            f"{r['win_rate_vs_sft'] * 100:.1f}%"
            if r["win_rate_vs_sft"] is not None
            else "–"
        )

        mean_kl = r["mean_kl"]

        mean_len = (
            sum(r["lengths"]) / len(r["lengths"])
        )

        mean_rouge = (
            sum(r["rouge_scores"]) / len(r["rouge_scores"])
        )

        print(
            f"{name:<8} | {mean_rm:>10.4f} | "
            f"{win_str:>10} | {mean_kl:>12.4f} | "
            f"{mean_len:>8.1f} | {mean_rouge:>8.4f}"
        )

    # Save summary table.
    table = {}

    for name in ["SFT", "PPO", "DPO"]:
        r = results[name]

        table[name] = dict(
            mean_rm_score=(
                sum(r["rm_scores"]) / len(r["rm_scores"])
            ),
            win_rate_vs_sft=r["win_rate_vs_sft"],
            mean_kl=r["mean_kl"],
            mean_length=(
                sum(r["lengths"]) / len(r["lengths"])
            ),
            mean_rouge_l=(
                sum(r["rouge_scores"]) / len(r["rouge_scores"])
            ),
        )

    with open(
        "results/eval_table.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(table, f, indent=2)

    print("\nResults saved to results/eval_summaries.jsonl")
    print("Table saved to results/eval_table.json")


if __name__ == "__main__":
    main()
