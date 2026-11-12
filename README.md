# RLHF Reddit Summarization

**Learning to Summarize from Human Feedback** — Generative AI Assignment 3

Reproduces the RLHF pipeline from [Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)
using `HuggingFaceTB/SmolLM2-135M` as the base model.

---

## Pipeline

| Stage | Script | What it does |
|-------|--------|--------------|
| 1 SFT | `train_sft.py` | Fine-tune on 20k human summaries |
| 2 Reward | `starter/train_reward.py` | Bradley-Terry reward model from 20k preference pairs |
| 3 PPO | `train_ppo.py` | RL fine-tuning with KL-constrained objective |
| 4 DPO | `train_dpo.py` | Direct Preference Optimisation from preference pairs |
| Eval | `evaluate.py` | Compare all three policies on 300 held-out posts |
| App | `app/app.py` | Gradio web interface |

---

## Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers datasets rouge-score matplotlib gradio
```

---

## Training Order

Run from the **project root** (`RLHF_Reddit_Summarization/`):

```bash
# 1. Supervised Fine-Tuning
python train_sft.py

# 2. Reward Model
python starter/train_reward.py

# 3. PPO
python train_ppo.py

# 4. DPO
python train_dpo.py

# 5. Evaluation
python evaluate.py

# 6. Front-end
python app/app.py
```

> **Note:** Use Colab or Kaggle with a T4/A100 GPU for stages 1–4.
> Checkpoints are saved to `checkpoints/`.

---

## Checkpoints

Download pre-trained checkpoints from [Google Drive / Hugging Face link]:

```
checkpoints/
  sft/best_model.pth
  reward/best_model.pth
  ppo/best_model.pth
  dpo/best_model.pth
```

Place them in the project root before running `evaluate.py` or `app/app.py`.

---

## Correctness Checks

### SFT Mask
Decoding tokens where `summary_mask == 1` gives exactly the summary text + EOS:
```
✓ Verified in starter/test_common.py
```

### Frozen Models
All parameters of πref and rϕ have `requires_grad = False` during Stages 3 and 4:
```
✓ Verified: ref_model.parameters() all have requires_grad=False
✓ Verified: reward_model.parameters() all have requires_grad=False
```

### PPO First Iteration
KL to reference ≈ 0 at iteration 1 (policy still equals SFT):
```
✓ Verified: mean_kl ≈ 0.0 at iteration 1
```

### PPO First Epoch
rt(θ) = 1 for every token and clip fraction = 0 in first PPO epoch of each rollout:
```
✓ Verified: ratio = exp(logprobs_new - logprobs_old) ≈ 1.0 (up to float error)
```

### DPO First Step
Loss ≈ log(2) ≈ 0.6931 when πθ = πref (argument of σ is 0):
```
✓ Verified: first-step loss printed in train_dpo.py sanity check
```

### Padding
All padding uses `attention_mask`, never `input_ids != pad_token_id`:
```
✓ reward_model.py: last_token_idx = attention_mask.sum(dim=1) - 1
✓ collate_prompts uses left-padding for model.generate()
```

---

## Results

### Table 1 – Data

| Subset | Size |
|--------|------|
| SFT train / val | 20,000 / 500 |
| Preference train / val | 20,000 / 1,000 |
| Evaluation posts | 300 |
| Mean / max prompt length (SFT train) | XX / 448 tokens |
| Mean / max summary length (SFT train) | XX / 48 tokens |

### Table 2 – Training

| Stage | Final train loss | Validation metric | Time / GPU |
|-------|-----------------|-------------------|------------|
| SFT | X.XX | val loss: X.XX | ~XX min / T4 |
| Reward model | X.XX | pairwise acc: XX% | ~XX min / T4 |
| PPO | – | final mean rϕ: X.XX | ~XX min / T4 |
| DPO | X.XX | implicit reward acc: XX% | ~XX min / T4 |

### Table 3 – Final Comparison (300 eval posts)

| Policy | Mean rϕ | Win rate vs SFT | KL to SFT | Length | ROUGE-L |
|--------|---------|-----------------|-----------|--------|---------|
| SFT | X.XX | – | 0 | XX | X.XX |
| PPO | X.XX | XX% | X.XX | XX | X.XX |
| DPO | X.XX | XX% | X.XX | XX | X.XX |

### Table 4 – Human Evaluation (30 posts)

| | SFT picked | PPO picked | DPO picked |
|---|---|---|---|
| Member 1 | X | X | X |
| Member 2 | X | X | X |
| Agreement | X/30 | | |

---

## Repository Layout

```
starter/           # Given starter code (unchanged config, data, batching, models)
  config.py
  data.py
  batching.py
  models.py
  check_starter.py
  common.py        # token_logprobs, generate_summaries (Task 1.3)
  train_sft.py     # original SFT script
  train_reward.py  # Task 3: reward model training
  reward_model.py  # Task 3: RewardModel class
train_sft.py       # Task 2: SFT from project root
train_ppo.py       # Task 4: PPO
train_dpo.py       # Task 5: DPO
evaluate.py        # Task 6: comparison on 300 posts
results/
  eval_summaries.jsonl
  eval_table.json
  analysis.md
  samples.md
  human_eval.csv
  training_logs/
  figures/
app/
  app.py           # Task 8: Gradio front-end
README.md
```

---

## Links

- 📄 [Medium Blog](LINK_HERE)
- 💼 [LinkedIn Post](LINK_HERE)
- 🤗 [Model Checkpoints](LINK_HERE)
