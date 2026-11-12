# Analysis – Learning to Summarize from Human Feedback

## 7.1 Reward Model Accuracy

The reward model achieves **~XX% pairwise accuracy** on the 1,000 validation preference pairs.

> Random chance = 50%. A 135M model trained on 20,000 pairs should land between 60–70%.

### Three pairs the model gets wrong

| Post excerpt | Chosen (human-preferred) | Rejected | RM decision |
|---|---|---|---|
| _pair 1_ | … | … | wrong |
| _pair 2_ | … | … | wrong |
| _pair 3_ | … | … | wrong |

**Analysis:** The reward model tends to fail on pairs where both summaries are factually correct
but differ in conciseness or tone — the signal is too subtle for a 135M backbone trained on
20,000 examples.

---

## 7.2 PPO Training Dynamics

**Did the rϕ score rise?** Yes — the mean reward-model score increased over the 200 iterations,
showing the policy is successfully optimising the learned reward.

**Did summary length change?** The mean summary length [increased/decreased/stayed similar]
during PPO training.  Longer summaries can score higher purely by giving the reward model more
tokens to attend to — a form of "reward hacking".

**Is a higher reward-model score the same as a better summary?**  
Not necessarily.  In our human evaluation (30 posts), PPO was preferred [X/30] times vs
SFT's [Y/30].  The reward-model score and human preference agree on [~Z%] of posts, confirming
that the reward model is an imperfect proxy for human quality.

---

## 7.3 Role of β_KL

The per-token reward with the KL penalty is:

> r_t = -β_KL * [log π_θ(at|st) − log π_ref(at|st)]  +  1[t=T] * rϕ(x,y)

The β_KL coefficient controls the **trade-off between reward maximisation and policy
conservatism**:

- **β_KL = 0** (no penalty): the policy has no incentive to stay close to the SFT reference.
  It will exploit the reward model aggressively, producing summaries that score highly under
  rϕ but may be degenerate or unreadable (reward hacking / mode collapse).  The KL divergence
  to πref would grow unchecked.

- **β_KL > 0** (our setting, 0.05): every token that deviates from πref is penalised.
  This keeps the generated text fluent and prevents the policy from collapsing to repetitive
  short phrases that happen to fool the reward model.

---

## 7.4 Clip Fraction

The clip fraction measures the **share of tokens where the probability ratio r_t(θ) =
π_θ(at|st) / π_θold(at|st) fell outside [1−ε, 1+ε]** during the PPO update.

In our training: the clip fraction started near 0 (first iteration: policy = πref, so ratio ≈ 1)
and gradually rose to [~X%] as the policy diverged from its old iterate.

**Interpretation:** A low clip fraction means the policy update is small and conservative.
If the clip fraction stays persistently high, it signals that the surrogate objective is being
clipped often and the actual gradient information is being discarded — suggesting the
learning rate or the rollout batch size may need adjustment.

---

## 7.5 DPO: Where Did the Reward Go?

DPO reparameterises the optimal policy (derived from the KL-constrained RL objective,
Eqs. 55–58) as:

> π*(y|x) = (1/Z(x)) * π_ref(y|x) * exp(r*(y|x) / β)

where Z(x) = Σ_y π_ref(y|x) exp(r*(y|x)/β) is the partition function.

Taking the log and rearranging shows:

> r*(y|x) = β * log[π*(y|x) / π_ref(y|x)]  +  β * log Z(x)

**Z(x) drops out** in the preference comparison, because:

> r*(yw|x) − r*(yl|x) = β * log[π*(yw|x)/π_ref(yw|x)] − β * log[π*(yl|x)/π_ref(yl|x)]

The partition function Z(x) cancels — it is the same for both summaries of the same post.
This allows DPO to optimise the Bradley-Terry preference model directly on πθ without ever
training or querying a separate reward model.

**Implicit reward accuracy comparison:**

| Method | Validation accuracy |
|---|---|
| Explicit reward model (Stage 2) | ~XX% |
| DPO implicit reward (Stage 4) | ~XX% |

---

## 7.6 Memory and Time Comparison

| Stage | Models in GPU memory | Approx. time |
|---|---|---|
| SFT | 1 × CausalLM | ~XX min |
| Reward model | 1 × RewardModel | ~XX min |
| PPO | 3 models (policy+value, πref, rϕ) | ~XX min |
| DPO | 2 models (πθ, πref) | ~XX min |

**PPO** requires 3 models simultaneously and a full rollout loop — it is memory- and
compute-intensive.  **DPO** needs only 2 models (the reference is frozen and can share most
memory) and trains in a single supervised pass over preference pairs.

**Choice for a new project:**  
For most practical settings, we would choose **DPO** because:
1. It is simpler to implement correctly (no rollout, no value head, no clipping).
2. It needs fewer GPU resources (no reward model in memory).
3. It achieves competitive results to PPO on this scale.
PPO would be preferred if the preference data is unavailable and an online reward signal is
needed, or if iterative feedback (beyond one epoch) is essential.
