
import os
import sys
import torch
import gradio as gr

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "starter")
)

from config import MAX_SUMMARY_TOKENS
from models import load_tokenizer, load_causal_lm
from reward_model import RewardModel
from data import TLDR


REWARD_OFFSET = 2.0

_loaded = {}


def get_models():
    global _loaded

    if _loaded:
        return _loaded

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    tok = load_tokenizer()

    def load_policy(path):
        m = load_causal_lm(device=device)

        state = torch.load(
            path,
            map_location=device,
            weights_only=True
        )

        m.load_state_dict(state, strict=True)
        m.config.use_cache = True
        m.eval()

        return m

    base_dir = os.path.join(
        os.path.dirname(__file__),
        ".."
    )

    sft_path = os.path.join(
        base_dir, "checkpoints", "sft", "best_model.pth"
    )

    ppo_path = os.path.join(
        base_dir, "checkpoints", "ppo", "best_model.pth"
    )

    dpo_path = os.path.join(
        base_dir, "checkpoints", "dpo", "best_model.pth"
    )

    reward_path = os.path.join(
        base_dir, "checkpoints", "reward", "best_model.pth"
    )

    print("Loading SFT model...")
    sft = load_policy(sft_path)

    print("Loading PPO model...")
    ppo = load_policy(ppo_path)

    print("Loading DPO model...")
    dpo = load_policy(dpo_path)

    print("Loading reward model...")
    rwd = RewardModel(reward_path, device)
    rwd.eval()

    _loaded = {
        "tok": tok,
        "sft": sft,
        "ppo": ppo,
        "dpo": dpo,
        "reward": rwd,
        "device": device
    }

    print("All models loaded successfully.")

    return _loaded


@torch.no_grad()
def generate_one(model, tok, prompt_text, device):

    input_ids = tok(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False
    ).input_ids.to(device)

    attn_mask = torch.ones_like(input_ids)

    prompt_len = input_ids.size(1)

    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_new_tokens=MAX_SUMMARY_TOKENS,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id
    )

    new_tokens = gen[0, prompt_len:]

    eos_positions = (
        new_tokens == tok.eos_token_id
    ).nonzero(as_tuple=True)[0]

    if len(eos_positions) > 0:
        new_tokens = new_tokens[
            :eos_positions[0].item() + 1
        ]

    return tok.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()


@torch.no_grad()
def reward_score(
    reward_model,
    tok,
    prompt_text,
    summary_text,
    device
):

    full_text = prompt_text + " " + summary_text

    enc = tok(
        full_text,
        return_tensors="pt",
        add_special_tokens=False
    )

    ids = enc.input_ids.to(device)
    mask = enc.attention_mask.to(device)

    score = reward_model(ids, mask)

    score = score + REWARD_OFFSET

    return score.item()


def clean_post(raw_text):

    post = raw_text.split("TL;DR:")[0].strip()

    return post + "\nTL;DR:"


def summarize(post_text):

    if not post_text.strip():
        return (
            "Please paste a Reddit post.",
            "",
            "",
            "",
            "",
            ""
        )

    models = get_models()

    tok = models["tok"]
    device = models["device"]

    prompt = clean_post(post_text)

    sft_sum = generate_one(
        models["sft"], tok, prompt, device
    )

    ppo_sum = generate_one(
        models["ppo"], tok, prompt, device
    )

    dpo_sum = generate_one(
        models["dpo"], tok, prompt, device
    )

    sft_score = reward_score(
        models["reward"], tok,
        prompt, sft_sum, device
    )

    ppo_score = reward_score(
        models["reward"], tok,
        prompt, ppo_sum, device
    )

    dpo_score = reward_score(
        models["reward"], tok,
        prompt, dpo_sum, device
    )

    return (
        sft_sum,
        f"Reward score: {sft_score:.4f}",

        ppo_sum,
        f"Reward score: {ppo_score:.4f}",

        dpo_sum,
        f"Reward score: {dpo_score:.4f}"
    )


def build_ui():

    with gr.Blocks(
        title="RLHF Reddit Summarizer",
        theme=gr.themes.Soft()
    ) as demo:

        gr.Markdown("""
# 🤖 RLHF Reddit Summarizer

**Learning to Summarize from Human Feedback** — SmolLM2-135M fine-tuned with SFT, PPO, and DPO.

Paste a Reddit post below (or include a SUBREDDIT / TITLE / body).

The three models generate a **TL;DR** summary and the reward model scores each one.
        """)

        with gr.Row():

            with gr.Column(scale=1):

                post_input = gr.Textbox(
                    label="Reddit Post",
                    placeholder="Paste your Reddit post here...",
                    lines=10
                )

                submit_btn = gr.Button(
                    "Summarize",
                    variant="primary"
                )

        gr.Markdown("---")
        gr.Markdown("### Summaries")

        with gr.Row():

            with gr.Column():

                gr.Markdown(
                    "**🔵 SFT (Supervised Fine-Tuning)**"
                )

                sft_out = gr.Textbox(
                    label="Summary",
                    lines=4,
                    interactive=False
                )

                sft_score = gr.Textbox(
                    label="Reward Score",
                    interactive=False
                )

            with gr.Column():

                gr.Markdown(
                    "**🟢 PPO (Proximal Policy Optimisation)**"
                )

                ppo_out = gr.Textbox(
                    label="Summary",
                    lines=4,
                    interactive=False
                )

                ppo_score = gr.Textbox(
                    label="Reward Score",
                    interactive=False
                )

            with gr.Column():

                gr.Markdown(
                    "**🟠 DPO (Direct Preference Optimisation)**"
                )

                dpo_out = gr.Textbox(
                    label="Summary",
                    lines=4,
                    interactive=False
                )

                dpo_score = gr.Textbox(
                    label="Reward Score",
                    interactive=False
                )

        gr.Examples(
            examples=[
                [
                    "SUBREDDIT: r/relationship_advice\n"
                    "TITLE: I (25F) found out my partner (27M) "
                    "has been lying to me about his job for 6 months\n\n"
                    "My partner told me he was promoted at work "
                    "six months ago and has been going out every day "
                    "as if he's working. I found out yesterday through "
                    "a mutual friend that he was actually laid off "
                    "and has been spending savings. He hasn't told me "
                    "anything. We live together and I pay 60% of the "
                    "rent. I'm devastated and don't know how to "
                    "approach this."
                ]
            ],
            inputs=post_input
        )

        submit_btn.click(
            fn=summarize,
            inputs=post_input,
            outputs=[
                sft_out,
                sft_score,
                ppo_out,
                ppo_score,
                dpo_out,
                dpo_score
            ]
        )

    return demo


if __name__ == "__main__":

    demo = build_ui()

    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )
