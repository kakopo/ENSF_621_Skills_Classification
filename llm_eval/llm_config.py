"""All settings for the LLM comparison (RQ3). The other files only read from here.

To add or change an LLM, edit the LLMS dictionary below. Nothing else needs to change.

!! Model IDs and prices change often. Check them on the providers' pages before a run, then set
!! PRICES_VERIFIED = True (the code prints a warning until you do):
!!   OpenAI:   https://platform.openai.com/docs/pricing
!!   Gemini:   https://ai.google.dev/gemini-api/docs/pricing
!!   DeepSeek: https://api-docs.deepseek.com/quick_start/pricing
"""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

# ---------------- which skills are classified ----------------
# "sample": a stratified sample of SAMPLE_SIZE skills (default)
# "full":   every skill in SAMPLE_FROM
DATASET = "sample"
SAMPLE_SIZE = 1000
# "test": the ML test split (1,100 skills). The ML models never saw these, so ML and LLMs are compared fairly.
# "all":  all 5,498 cleaned skills. About 80% of these were ML TRAINING data, so the ML scores on them are inflated
#         (evaluate.py then compares ML and LLMs on the test-split rows only).
SAMPLE_FROM = "test"
# The sample keeps the share of every label AND of every attack pattern inside each label.
# primary_pattern is only used to pick the sample. It is never sent to the LLM.
STRATIFY_BY = ["judgment", "primary_pattern"]
RANDOM_STATE = 42  # same seed as the ML notebook

# ---------------- what the LLM sees ----------------
# The same input as the ML models: cleaned text (full_clean) plus the 9 binary flags. Written by Section 10 of
# skill_vulnerability_classifier_with_flags_v2.ipynb.
DATA_FILE = PROJECT_DIR / "llm_eval" / "data" / "skills_for_llm.jsonl"
DATA_META = PROJECT_DIR / "llm_eval" / "data" / "skills_for_llm_meta.json"
INCLUDE_FLAGS = True
# Longer skills are cut in the middle (start and end are kept). 120k characters is about 30k tokens and affects
# roughly the longest 3% of skills. The predictions file records which skills were cut.
MAX_INPUT_CHARS = 120_000

# ---------------- output ----------------
RESULTS_DIR = PROJECT_DIR / "results" / "llm"

# ---------------- the LLMs ----------------
# provider: "openai" (OpenAI Batch API), "gemini" (Gemini Batch API), or "openai_compatible" (DeepSeek and any other
#           API that speaks the OpenAI chat format. No batch API, so requests are sent in parallel)
# api_key_env: name of the environment variable (put it in the .env file in the project folder)
# prices: US$ per 1M tokens at the STANDARD (non-batch) rate. cached_input = price of input tokens served from the
#         provider's prompt cache (None = same as input).
# batch_discount: multiplier applied to all prices when the provider's batch API is used (0.5 = half price)
# json_mode: "schema" (strict JSON schema), "object" (any JSON object, for DeepSeek), or "none" (parse the text)
# temperature: None = do not send it (reasoning models such as OpenAI's reject temperature)
# max_output_tokens: includes hidden reasoning tokens for reasoning models, so keep some room
# reasoning_effort (OpenAI only): "minimal"/"low"/"medium"/"high" or None (model default). Lower = cheaper.
# batch_chunk_size: requests per batch job. OpenAI limits how many tokens can be queued per model (depends on
#                   your usage tier), so a big run is split into several jobs.
PRICES_VERIFIED = False

LLMS = {
    "gpt": {
        "provider": "openai",
        "model": "gpt-6.1-terra",          # VERIFY
        "api_key_env": "OPENAI_API_KEY",
        "prices": {"input": 1.00, "cached_input": 0.10, "output": 5.00},   # VERIFY
        "batch_discount": 0.5,
        "json_mode": "schema",
        "temperature": None,
        "reasoning_effort": "low",
        "max_output_tokens": 4000,
        "batch_chunk_size": 250,
    },
    "gemini": {
        "provider": "gemini",
        "model": "gemini-3.5-flash-lite",  # VERIFY
        "api_key_env": "GEMINI_API_KEY",
        "prices": {"input": 0.30, "cached_input": None, "output": 2.50},   # VERIFY
        "batch_discount": 0.5,
        "json_mode": "schema",
        "temperature": 0.0,
        "max_output_tokens": 4000,
        "batch_chunk_size": 1000,
    },
    "deepseek": {
        "provider": "openai_compatible",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-flash",         # VERIFY
        "api_key_env": "DEEPSEEK_API_KEY",
        # DeepSeek's peak rates. Off-peak (UTC night) is cheaper, so this is an upper bound. VERIFY
        "prices": {"input": 0.30, "cached_input": None, "output": 1.20},
        "batch_discount": 1.0,             # no batch API, so no discount
        "json_mode": "object",
        "temperature": 0.0,
        "max_output_tokens": 4000,
        "max_workers": 8,                  # parallel requests
    },
}

# realtime mode (all providers): one request per skill, sent in parallel. Full price. Use it for quick tests,
# e.g. `python -m llm_eval.runner run --llm gpt --mode realtime --limit 5`
REALTIME_MAX_WORKERS = 8
MAX_RETRIES = 5
POLL_SECONDS = 60
