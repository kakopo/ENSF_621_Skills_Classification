"""Scores the LLMs and the ML models on exactly the same skills, and saves the comparison tables."""
import json

import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score, precision_score,
                             recall_score)

from . import data
from . import llm_config as cfg

LABELS = data.LABELS


def scores(y_true, y_pred):
    """Same metrics as Section 6 of the ML notebook. An unreadable LLM answer ('unparsed') counts as wrong."""
    kw = dict(labels=LABELS, average="macro", zero_division=0)
    out = {"accuracy": accuracy_score(y_true, y_pred),
           "macro precision": precision_score(y_true, y_pred, **kw),
           "macro recall": recall_score(y_true, y_pred, **kw),
           "macro F1": f1_score(y_true, y_pred, **kw),
           "weighted F1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)}
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    out.update({f"F1 {lab}": v for lab, v in zip(LABELS, per_class)})
    return out


def load_predictions(run_dir):
    """{llm: predictions DataFrame} for every LLM folder in a run."""
    preds = {}
    for p in sorted(run_dir.glob("*/predictions.csv")):
        preds[p.parent.name] = pd.read_csv(p, dtype={"id": str})
    return preds


def compare(run_name, ml_models=None, save=True):
    """Comparison table for one run (folder name under results/llm/).

    Every model is scored on the SAME skills: the skills of the run that are in the ML test split and were answered
    by every LLM that took part. A skill an LLM failed on stays in and counts as wrong (fair to the ML models,
    which always answer).
    """
    run_dir = cfg.RESULTS_DIR / run_name
    llm_preds = load_predictions(run_dir)
    if not llm_preds:
        raise FileNotFoundError(f"No predictions.csv under {run_dir}")
    export, meta = data.load_export()
    export = export.set_index("id")
    ml_cols = [c for c in export.columns if c.startswith("ml_")]
    if ml_models:
        ml_cols = [f"ml_{m}" for m in ml_models]
        missing = [c[3:] for c in ml_cols if c not in export.columns]
        if missing:
            raise KeyError(f"No predictions for {missing} in {cfg.DATA_FILE.name}. For the stacked models, run "
                           "Section 6.5 and then Section 10 of the ML notebook again. "
                           f"Available: {[c[3:] for c in export.columns if c.startswith('ml_')]}")
    stack_names = set(meta.get("stack_models", []))

    # common skills: answered (or attempted) by every LLM, and in the ML test split
    common = set.intersection(*(set(p["id"]) for p in llm_preds.values()))
    common = sorted(cid for cid in common if export.loc[cid, "split"] == "test")
    y_true = export.loc[common, "judgment"]
    n_run = len(pd.read_csv(run_dir / "selected_skills.csv"))

    rows = {}
    for llm, p in llm_preds.items():
        p = p.set_index("id")
        conf = json.loads((run_dir / llm / "config_used.json").read_text())
        cost = p["cost_usd"].sum()
        rows[f"LLM: {llm}"] = {
            "type": "LLM", "model": conf["model"], **scores(y_true, p.loc[common, "prediction"]),
            "skills scored": len(common), "failed answers": int((p.loc[common, "prediction"] == "unparsed").sum()),
            "cost (USD)": cost, "cost per 1k skills (USD)": 1000 * cost / len(p),
            "est. cost all 5,498 skills (USD)": len(export) * cost / len(p),
            "input tokens": int(p["input_tokens"].sum()), "output tokens": int(p["output_tokens"].sum()),
            "reasoning tokens": int(p["reasoning_tokens"].sum()),
            "mean latency (s)": p["latency_s"].mean() if p["latency_s"].notna().any() else None,
            "via": "/".join(sorted(p["via"].dropna().unique())),
        }
    for col in ml_cols:
        name = col[3:]
        kind = "ML (stack)" if name in stack_names else "ML (best)" if name == meta.get("best_model") else "ML"
        rows[f"ML: {name}"] = {"type": kind, "model": name,
                               **scores(y_true, export.loc[common, col]), "skills scored": len(common),
                               "failed answers": 0, "cost (USD)": 0.0}
    table = pd.DataFrame(rows).T
    table = table.sort_values("macro F1", ascending=False)

    reports, cms = {}, {}
    all_preds = {**{f"LLM: {k}": v.set_index("id").loc[common, "prediction"] for k, v in llm_preds.items()},
                 **{f"ML: {c[3:]}": export.loc[common, c] for c in ml_cols}}
    for name, pred in all_preds.items():
        reports[name] = classification_report(y_true, pred, labels=LABELS, digits=3, zero_division=0)
        cms[name] = pd.DataFrame(confusion_matrix(y_true, pred, labels=LABELS),
                                 index=[f"true {l}" for l in LABELS], columns=[f"pred {l}" for l in LABELS])

    if save:
        table.to_csv(run_dir / "comparison_llm_vs_ml.csv")
        with open(run_dir / "classification_reports.txt", "w") as f:
            for name, rep in reports.items():
                f.write(f"==================== {name} ====================\n{rep}\n")
        pd.concat(cms, names=["model"]).to_csv(run_dir / "confusion_matrices.csv")
        print(f"Saved comparison of {len(rows)} models on {len(common)} skills -> {run_dir.relative_to(cfg.PROJECT_DIR)}")
    if len(common) < n_run:
        print(f"Note: {n_run - len(common)} skills of this run are left out (not answered by every LLM yet, "
              "or not in the ML test split).")
    return table, reports, cms
