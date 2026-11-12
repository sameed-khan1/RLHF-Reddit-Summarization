
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"

SFT_DATA = "CarperAI/openai_summarize_tldr"
PREF_DATA = "CarperAI/openai_summarize_comparisons"

MAX_PROMPT_TOKENS = 448
MAX_SUMMARY_TOKENS = 48

SEED = 42

N_SFT_TRAIN = 20000
N_SFT_VAL = 500

N_PREF_TRAIN = 20000
N_PREF_VAL = 1000

N_EVAL_POSTS = 300
