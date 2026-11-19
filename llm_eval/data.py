"""Loads the skills exported by the ML notebook and picks the full set or a stratified sample."""
import json

import numpy as np
import pandas as pd

from . import llm_config as cfg

LABELS = ["normal", "suspicious", "malicious"]


def load_export():
    """The cleaned skills (same text and flags as the ML models) + the ML test predictions. Returns (df, meta)."""
    if not cfg.DATA_FILE.exists():
        raise FileNotFoundError(f"{cfg.DATA_FILE} not found. Run Section 10 of "
                                "skill_vulnerability_classifier_with_flags_v2.ipynb first.")
    df = pd.read_json(cfg.DATA_FILE, lines=True, dtype={"id": str})
    meta = json.loads(cfg.DATA_META.read_text())
    return df, meta


def _largest_remainder(sizes: pd.Series, n: int) -> pd.Series:
    """Split n across groups in proportion to their sizes. Whole numbers that add up to exactly n."""
    exact = sizes / sizes.sum() * n
    alloc = np.floor(exact).astype(int)
    leftover = n - alloc.sum()
    # the groups with the largest fractional part get the remaining places (ties: bigger group first)
    order = (exact - alloc).to_frame("frac").assign(size=sizes).sort_values(["frac", "size"], ascending=False).index
    alloc[order[:leftover]] += 1
    return alloc


def stratified_sample(df: pd.DataFrame, n: int = 1000, by=("judgment", "primary_pattern"),
                      random_state: int = 42) -> pd.DataFrame:
    """A sample of n skills with the same proportions as df.

    Level 1: n is split across by[0] (the label) in proportion, so the label counts are exact.
    Level 2: inside each label, its share is split across by[1] (e.g. attack pattern) in proportion.
    Groups too small to get one place may be left out (that is unavoidable for any sample).
    """
    if n >= len(df):
        return df.copy()
    by = list(by)
    # most normal skills have no attack pattern: "(none)" becomes its own group instead of being dropped
    df = df.assign(**{c: df[c].fillna("(none)") for c in by})
    rng = np.random.RandomState(random_state)
    parts = []
    label_alloc = _largest_remainder(df.groupby(by[0]).size(), n)
    for label, n_label in label_alloc.items():
        sub = df[df[by[0]] == label]
        if len(by) > 1:
            sub_alloc = _largest_remainder(sub.groupby(by[1]).size(), n_label)
            for value, k in sub_alloc.items():
                group = sub[sub[by[1]] == value]
                parts.append(group.loc[rng.choice(group.index, size=k, replace=False)])
        else:
            parts.append(sub.loc[rng.choice(sub.index, size=n_label, replace=False)])
    return pd.concat(parts).sort_index()


def representativeness(population: pd.DataFrame, sample: pd.DataFrame, col: str) -> pd.DataFrame:
    """Share (%) of each value of col in the population and in the sample."""
    pop, samp = population[col].fillna("(none)"), sample[col].fillna("(none)")
    out = pd.DataFrame({"population %": pop.value_counts(normalize=True) * 100,
                        "sample %": samp.value_counts(normalize=True) * 100}).fillna(0)
    out["difference"] = out["sample %"] - out["population %"]
    return out.round(2).sort_values("population %", ascending=False)


def select_skills(dataset: str = None, n: int = None, sample_from: str = None) -> pd.DataFrame:
    """The skills to classify, chosen with the settings in llm_config (each can be overridden here)."""
    dataset = dataset or cfg.DATASET
    n = n or cfg.SAMPLE_SIZE
    sample_from = sample_from or cfg.SAMPLE_FROM
    df, _ = load_export()
    if sample_from == "test":
        pool = df[df["split"] == "test"]
    elif sample_from == "all":
        pool = df
    else:
        raise ValueError(f"SAMPLE_FROM must be 'test' or 'all', not {sample_from!r}")
    if dataset == "full":
        chosen = pool
    elif dataset == "sample":
        chosen = stratified_sample(pool, n, cfg.STRATIFY_BY, cfg.RANDOM_STATE)
    else:
        raise ValueError(f"DATASET must be 'sample' or 'full', not {dataset!r}")
    return chosen.reset_index(drop=True)


def run_name(dataset: str = None, n: int = None, sample_from: str = None) -> str:
    """Folder name of a run, e.g. 'test_sample1000_seed42'. The same settings always give the same folder."""
    dataset = dataset or cfg.DATASET
    sample_from = sample_from or cfg.SAMPLE_FROM
    if dataset == "full":
        return f"{sample_from}_full"
    return f"{sample_from}_sample{n or cfg.SAMPLE_SIZE}_seed{cfg.RANDOM_STATE}"
