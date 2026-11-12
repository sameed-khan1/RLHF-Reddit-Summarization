"""
common.py – shared utilities used by every training stage.

Provides:
  token_logprobs(model, input_ids, attention_mask, prompt_lens)
      → (log_probs, summary_mask)   shapes (B, L-1) each

  generate_summaries(model, tokenizer, prompts_batch, max_new_tokens)
      → list[str]  of decoded summaries (without prompt)
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─────────────────────────────────────────────────────────────────────
# Core helper: per-token log-probabilities + summary mask
# ─────────────────────────────────────────────────────────────────────

def token_logprobs(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,       # (B, L)
    attention_mask: torch.Tensor,  # (B, L)
    prompt_lens: torch.Tensor,     # (B,)  number of prompt tokens per row
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-token log-probabilities and a binary summary mask.

    For a sequence [p_0, p_1, … , p_{P-1}, s_0, s_1, … , s_{S-1}]
    (prompt tokens followed by summary tokens, right-padded):

      - logprobs[b, t]      = log π(input_ids[b, t+1] | input_ids[b, :t+1])
                              for  t = 0 … L-2  (shape B×(L-1))
      - summary_mask[b, t]  = 1  iff  t+1 ≥ prompt_lens[b]   AND
                                       attention_mask[b, t+1] = 1
                              i.e. it is 1 exactly on summary tokens
                              (not prompt, not padding).

    Returns:
        logprobs     : (B, L-1)   float tensor
        summary_mask : (B, L-1)   long  tensor  (0 or 1)
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # logits[:, :-1, :] predicts token at positions 1 … L-1
    logits  = outputs.logits[:, :-1, :]          # (B, L-1, V)
    targets = input_ids[:, 1:]                   # (B, L-1)

    logprobs = F.log_softmax(logits, dim=-1)
    logprobs = logprobs.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)                                # (B, L-1)

    # Position indices for the *target* tokens (positions 1 … L-1)
    L_minus_1 = input_ids.size(1) - 1
    pos = torch.arange(1, input_ids.size(1), device=input_ids.device)
    pos = pos.unsqueeze(0)                       # (1, L-1)

    # summary mask: 1 where position >= prompt_len AND token is real
    pl = prompt_lens.to(input_ids.device).unsqueeze(1)  # (B, 1)
    summary_mask = (pos >= pl).long()            # (B, L-1)
    summary_mask = summary_mask * attention_mask[:, 1:]  # zero out padding

    return logprobs, summary_mask


# ─────────────────────────────────────────────────────────────────────
# Generation helper
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_summaries(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch: dict,                   # from collate_prompts: input_ids, attention_mask
    max_new_tokens: int = 48,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> list[str]:
    """
    Run model.generate on a left-padded prompt batch and return
    the decoded summary strings (prompt part stripped).

    Returns a list of raw decoded strings (may contain the EOS token text).
    """
    input_ids      = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    prompt_lens    = input_ids.size(1)  # left-padded, so all share the same length

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_k=top_k, top_p=top_p)
    else:
        gen_kwargs["do_sample"] = False

    outputs = model.generate(**gen_kwargs)  # (B, prompt_len + new_tokens)

    # Strip the prompt prefix
    new_tokens = outputs[:, prompt_lens:]   # (B, new_tokens)

    summaries = []
    for row in new_tokens:
        # Truncate at first EOS
        eos_positions = (row == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            row = row[: eos_positions[0] + 1]  # keep EOS for correctness checks
        text = tokenizer.decode(row, skip_special_tokens=False)
        summaries.append(text)

    return summaries
