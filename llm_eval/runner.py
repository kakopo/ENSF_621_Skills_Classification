"""Sends the skills to the LLMs, collects the answers, and saves predictions, tokens and cost.

Use it from the notebook (llm_comparison.ipynb) or from a terminal in the project folder:

    python3 -m llm_eval.runner estimate                       # expected tokens and cost, no API calls
    python3 -m llm_eval.runner run --llm gpt --mode realtime --limit 5   # quick test on 5 skills
    python3 -m llm_eval.runner submit --llm gpt gemini        # start the batch jobs (results within 24 h)
    python3 -m llm_eval.runner status                         # are the jobs done?
    python3 -m llm_eval.runner collect                        # download finished jobs -> predictions.csv
    python3 -m llm_eval.runner run --llm deepseek             # DeepSeek: parallel requests, finishes in one go

Options for every command: --dataset sample|full, --n 1000, --from test|all, --limit N (first N skills only).
Everything is saved under results/llm/<run>/<llm>/. Running a command again continues where it stopped:
skills that already have an answer are skipped, and a batch is never submitted twice.
"""
import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from . import data
from . import llm_config as cfg
from .prompt import SYSTEM_PROMPT, build_user_prompt, parse_answer
from .providers import PROVIDERS, empty_result, make_provider

PROMPT_VERSION = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:10]


def load_env():
    """Reads API keys from the .env file in the project folder (if python-dotenv is installed)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(cfg.PROJECT_DIR / ".env")
    except ImportError:
        pass


def _warn_prices():
    if not cfg.PRICES_VERIFIED:
        print("WARNING: model IDs and prices in llm_config.py are not marked as verified. Check them on the "
              "providers' pricing pages, then set PRICES_VERIFIED = True.\n")


# ------------------------------------------------------------------ choosing the skills
class Run:
    """One experiment: the chosen skills (same for every LLM) and the folder its results go to."""

    def __init__(self, dataset=None, n=None, sample_from=None, limit=None):
        self.dataset = dataset or cfg.DATASET
        self.n = n or cfg.SAMPLE_SIZE
        self.sample_from = sample_from or cfg.SAMPLE_FROM
        self.name = data.run_name(self.dataset, self.n, self.sample_from) + (f"_first{limit}" if limit else "")
        self.dir = cfg.RESULTS_DIR / self.name
        self.dir.mkdir(parents=True, exist_ok=True)

        _, self.meta = data.load_export()
        skills = data.select_skills(self.dataset, self.n, self.sample_from)
        if limit:   # a small test run that still has all three labels
            skills = data.stratified_sample(skills, limit, ["judgment"], cfg.RANDOM_STATE).reset_index(drop=True)
        built = [build_user_prompt(r, self.meta["text_col"], self.meta["flag_cols"]) for _, r in skills.iterrows()]
        skills["prompt"] = [p for p, _ in built]
        skills["truncated"] = [cut for _, cut in built]
        self.skills = skills.set_index("id", drop=False)
        self._save_selection()

    def _save_selection(self):
        cols = ["id", "judgment", "split", *[c for c in cfg.STRATIFY_BY if c != "judgment"], "truncated"]
        path = self.dir / "selected_skills.csv"
        sel = self.skills[cols].reset_index(drop=True)
        if path.exists():
            old = pd.read_csv(path, dtype={"id": str})
            if set(old["id"]) != set(sel["id"]):
                raise RuntimeError(f"{path} lists different skills than the current settings choose. "
                                   "Delete that run folder or change the settings back.")
            return
        sel.to_csv(path, index=False)
        if self.dataset == "sample":   # how well the sample represents the pool it was drawn from
            full, _ = data.load_export()
            pool = full if self.sample_from == "all" else full[full["split"] == "test"]
            tables = [data.representativeness(pool, self.skills, c).assign(column=c) for c in cfg.STRATIFY_BY]
            pd.concat(tables).to_csv(self.dir / "sample_vs_population.csv")

    def requests(self, ids=None):
        rows = self.skills if ids is None else self.skills.loc[list(ids)]
        return [(cid, SYSTEM_PROMPT, p) for cid, p in zip(rows["id"], rows["prompt"])]


# ------------------------------------------------------------------ saved answers
def _llm_dir(run, llm):
    d = run.dir / llm
    d.mkdir(exist_ok=True)
    return d


_lock = threading.Lock()


def _append_results(run, llm, results, via):
    with _lock, open(_llm_dir(run, llm) / "raw_results.jsonl", "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({**r, "via": via, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                               ensure_ascii=False) + "\n")


def _load_results(run, llm):
    """The latest saved answer per skill (a retry replaces an earlier failed answer)."""
    path = _llm_dir(run, llm) / "raw_results.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def _ok(r):
    return r.get("error") is None and parse_answer(r.get("text"))[0] is not None


def _pending_ids(run, llm, retry_failed=False):
    saved = _load_results(run, llm)
    return [cid for cid in run.skills.index if cid not in saved or (retry_failed and not _ok(saved[cid]))]


def _jobs_path(run, llm):
    return _llm_dir(run, llm) / "batch_jobs.json"


def _load_jobs(run, llm):
    p = _jobs_path(run, llm)
    return json.loads(p.read_text()) if p.exists() else []


def _save_jobs(run, llm, jobs):
    _jobs_path(run, llm).write_text(json.dumps(jobs, indent=2))


def _save_config(run, llm, conf):
    info = {"llm": llm, **conf, "prompt_version": PROMPT_VERSION, "include_flags": cfg.INCLUDE_FLAGS,
            "max_input_chars": cfg.MAX_INPUT_CHARS, "prices_verified": cfg.PRICES_VERIFIED,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (_llm_dir(run, llm) / "config_used.json").write_text(json.dumps(info, indent=2))


# ------------------------------------------------------------------ cost
def cost_usd(conf, input_tokens, cached_input_tokens, output_tokens, batch):
    p = conf["prices"]
    cached_price = p["input"] if p.get("cached_input") is None else p["cached_input"]
    usd = ((input_tokens - cached_input_tokens) * p["input"] + cached_input_tokens * cached_price
           + output_tokens * p["output"]) / 1e6
    return usd * (conf.get("batch_discount", 1.0) if batch else 1.0)


def estimate(llms=None, mode="batch", **run_args):
    """Expected tokens and cost before sending anything. Input tokens ~ characters / 4 (a rough rule of thumb);
    output assumes ~80 answer tokens plus a guess for hidden reasoning (300 for models with reasoning)."""
    _warn_prices()
    run = Run(**run_args)
    in_tokens = (run.skills["prompt"].str.len().sum() + len(SYSTEM_PROMPT) * len(run.skills)) / 4
    rows = []
    for llm in llms or list(cfg.LLMS):
        conf = cfg.LLMS[llm]
        reasoning = 300 if conf.get("reasoning_effort") or conf["provider"] == "gemini" else 0
        out_tokens = (80 + reasoning) * len(run.skills)
        batch = mode == "batch" and conf["provider"] in ("openai", "gemini")
        rows.append({"llm": llm, "model": conf["model"], "skills": len(run.skills), "via": "batch" if batch else "realtime",
                     "input tokens (M)": in_tokens / 1e6, "output tokens (M)": out_tokens / 1e6,
                     "estimated cost (USD)": cost_usd(conf, in_tokens, 0, out_tokens, batch)})
    table = pd.DataFrame(rows).set_index("llm")
    print(f"Run {run.name}: {len(run.skills)} skills, {run.skills['truncated'].sum()} cut to {cfg.MAX_INPUT_CHARS:,} chars")
    return table.round(3)


# ------------------------------------------------------------------ realtime (parallel requests)
def run_realtime(llm, run, retry_failed=False):
    conf = cfg.LLMS[llm]
    ids = _pending_ids(run, llm, retry_failed)
    if not ids:
        print(f"{llm}: all {len(run.skills)} skills already answered")
        return
    _save_config(run, llm, conf)
    provider = make_provider(llm, conf)
    workers = conf.get("max_workers", cfg.REALTIME_MAX_WORKERS)
    print(f"{llm}: sending {len(ids)} requests ({workers} in parallel) ...")
    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(provider.realtime, cid, sys_p, user_p) for cid, sys_p, user_p in run.requests(ids)]
        for fut in as_completed(futures):
            _append_results(run, llm, [fut.result()], via="realtime")   # saved one by one: safe to interrupt
            done += 1
            if done % 50 == 0 or done == len(ids):
                print(f"  {done}/{len(ids)}  ({time.time() - t0:.0f}s)")
    build_predictions(llm, run)


# ------------------------------------------------------------------ batch
def submit(llm, run, retry_failed=False):
    conf = cfg.LLMS[llm]
    if not PROVIDERS[conf["provider"]].supports_batch:
        print(f"{llm}: {conf['provider']} has no batch API, sending parallel requests instead")
        return run_realtime(llm, run, retry_failed)
    jobs = _load_jobs(run, llm)
    in_flight = {cid for j in jobs if not j.get("collected") for cid in j["ids"]}
    ids = [cid for cid in _pending_ids(run, llm, retry_failed) if cid not in in_flight]
    if not ids:
        print(f"{llm}: nothing to submit ({len(in_flight)} skills already in unfinished jobs)")
        return
    _save_config(run, llm, conf)
    provider = make_provider(llm, conf)
    size = conf.get("batch_chunk_size", 1000)
    for start in range(0, len(ids), size):
        chunk = ids[start:start + size]
        tag = f"{run.name}_{llm}_{len(jobs) + 1:02d}"
        job_id = provider.submit_batch(run.requests(chunk), _llm_dir(run, llm), tag)
        jobs.append({"job_id": job_id, "tag": tag, "ids": chunk, "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "submitted_ts": time.time(), "collected": False})
        _save_jobs(run, llm, jobs)    # saved after each job, so nothing is submitted twice
        print(f"{llm}: submitted job {job_id} with {len(chunk)} skills")


def status(llm, run, quiet=False):
    """Returns True when every job of this LLM has finished."""
    conf = cfg.LLMS[llm]
    jobs = [j for j in _load_jobs(run, llm) if not j.get("collected")]
    if not jobs:
        if not quiet:
            print(f"{llm}: no unfinished batch jobs")
        return True
    provider = make_provider(llm, conf)
    all_done = True
    for j in jobs:
        state, done, info = provider.batch_status(j["job_id"])
        all_done &= done
        if not quiet:
            print(f"{llm}: {j['job_id']}  {state}  {info}")
    return all_done


def collect(llm, run):
    """Downloads every finished job and writes predictions.csv."""
    conf = cfg.LLMS[llm]
    jobs = _load_jobs(run, llm)
    if any(not j.get("collected") for j in jobs):
        provider = make_provider(llm, conf)
        for j in jobs:
            if j.get("collected"):
                continue
            state, done, _ = provider.batch_status(j["job_id"])
            if not done:
                print(f"{llm}: {j['job_id']} is still {state}")
                continue
            results = provider.batch_results(j["job_id"])
            returned = {r["id"] for r in results}
            results += [empty_result(cid, f"no result (job {state})") for cid in j["ids"] if cid not in returned]
            _append_results(run, llm, results, via="batch")
            j.update(collected=True, final_state=state, collected_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                     wall_time_s=round(time.time() - j["submitted_ts"]))
            _save_jobs(run, llm, jobs)
            print(f"{llm}: collected {len(results)} results from {j['job_id']} ({state})")
    if _load_results(run, llm):
        return build_predictions(llm, run)


def wait_and_collect(llm, run):
    while not status(llm, run, quiet=True):
        time.sleep(cfg.POLL_SECONDS)
    return collect(llm, run)


# ------------------------------------------------------------------ predictions file
def build_predictions(llm, run):
    """predictions.csv: one row per skill with the true label, the LLM's label, tokens and cost."""
    conf = cfg.LLMS[llm]
    saved = _load_results(run, llm)
    rows = []
    for cid, skill in run.skills.iterrows():
        r = saved.get(cid)
        if r is None:
            continue
        label, conf_score, reason, parse_error = parse_answer(r.get("text"))
        batch = r.get("via") == "batch"
        rows.append({"id": cid, "judgment": skill["judgment"], "prediction": label or "unparsed",
                     "confidence": conf_score, "reason": reason, "error": r.get("error") or parse_error,
                     "truncated": skill["truncated"], "via": r.get("via"),
                     "input_tokens": r["input_tokens"], "cached_input_tokens": r["cached_input_tokens"],
                     "output_tokens": r["output_tokens"], "reasoning_tokens": r["reasoning_tokens"],
                     "cost_usd": cost_usd(conf, r["input_tokens"], r["cached_input_tokens"], r["output_tokens"], batch),
                     "latency_s": r.get("latency_s"), "raw_answer": r.get("text")})
    preds = pd.DataFrame(rows)
    path = _llm_dir(run, llm) / "predictions.csv"
    preds.to_csv(path, index=False)
    n_bad = (preds["prediction"] == "unparsed").sum()
    print(f"{llm}: {len(preds)}/{len(run.skills)} skills answered, {n_bad} failed or unreadable, "
          f"cost ${preds['cost_usd'].sum():.4f} -> {path.relative_to(cfg.PROJECT_DIR)}")
    if n_bad:
        print(f"  (re-send the failed ones with --retry-failed; first error: {preds.loc[preds['error'].notna(), 'error'].iloc[0]})")
    return preds


# ------------------------------------------------------------------ command line
def main(argv=None):
    ap = argparse.ArgumentParser(description="Classify skills with LLMs (see the top of this file).")
    ap.add_argument("command", choices=["estimate", "run", "submit", "status", "collect"])
    ap.add_argument("--llm", nargs="+", default=list(cfg.LLMS), help=f"default: all ({', '.join(cfg.LLMS)})")
    ap.add_argument("--mode", choices=["batch", "realtime"], default="batch",
                    help="batch = provider batch API where available (cheaper); realtime = parallel requests")
    ap.add_argument("--dataset", choices=["sample", "full"], default=None, help=f"default: {cfg.DATASET}")
    ap.add_argument("--n", type=int, default=None, help=f"sample size (default {cfg.SAMPLE_SIZE})")
    ap.add_argument("--from", dest="sample_from", choices=["test", "all"], default=None,
                    help=f"pool to use (default {cfg.SAMPLE_FROM})")
    ap.add_argument("--limit", type=int, default=None, help="only the first N chosen skills (for testing)")
    ap.add_argument("--retry-failed", action="store_true", help="re-send skills whose answer failed")
    a = ap.parse_args(argv)
    unknown = set(a.llm) - set(cfg.LLMS)
    if unknown:
        ap.error(f"unknown LLM(s) {sorted(unknown)}; defined in llm_config.py: {list(cfg.LLMS)}")

    load_env()
    run_args = dict(dataset=a.dataset, n=a.n, sample_from=a.sample_from, limit=a.limit)
    if a.command == "estimate":
        print(estimate(a.llm, a.mode, **run_args).to_string())
        return
    _warn_prices()
    run = Run(**run_args)
    print(f"Run {run.name}: {len(run.skills)} skills -> {run.dir.relative_to(cfg.PROJECT_DIR)}")
    for llm in a.llm:
        if a.command == "run":
            if a.mode == "realtime" or not PROVIDERS[cfg.LLMS[llm]["provider"]].supports_batch:
                run_realtime(llm, run, a.retry_failed)
            else:
                submit(llm, run, a.retry_failed)
                print(f"{llm}: waiting for the batch (checking every {cfg.POLL_SECONDS}s; Ctrl+C is safe, "
                      "use `collect` later) ...")
                wait_and_collect(llm, run)
        elif a.command == "submit":
            if a.mode == "realtime":
                run_realtime(llm, run, a.retry_failed)
            else:
                submit(llm, run, a.retry_failed)
        elif a.command == "status":
            status(llm, run)
        elif a.command == "collect":
            collect(llm, run)


if __name__ == "__main__":
    main()
