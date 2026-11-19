"""One class per API provider. All of them return results in the same format, so runner.py does not care which LLM
it is talking to.

A result is a dict:
    id, text (the model's answer), input_tokens, cached_input_tokens, output_tokens (includes reasoning),
    reasoning_tokens, error (None if OK), latency_s (realtime only)
"""
import json
import os
import time
from pathlib import Path

from .prompt import LABELS, RESPONSE_SCHEMA


def _get(d, *keys, default=None):
    """d[key] for the first key present (the APIs mix snake_case and camelCase)."""
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return default


def _api_key(conf):
    key = os.environ.get(conf["api_key_env"])
    if not key:
        raise RuntimeError(f"Environment variable {conf['api_key_env']} is not set (add it to the .env file).")
    return key


def empty_result(custom_id, error=None):
    return {"id": custom_id, "text": None, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0,
            "reasoning_tokens": 0, "error": error, "latency_s": None}


class Provider:
    supports_batch = False

    def __init__(self, name, conf):
        self.name, self.conf, self.model = name, conf, conf["model"]

    def realtime(self, custom_id, system, user):
        raise NotImplementedError

    def submit_batch(self, requests, work_dir: Path, tag: str):
        """requests = [(id, system, user)]. Returns a job id."""
        raise NotImplementedError(f"{self.name}: provider has no batch API, use mode='realtime'")

    def batch_status(self, job_id):
        """Returns (state, finished: bool, info dict)."""
        raise NotImplementedError

    def batch_results(self, job_id):
        """List of result dicts for a finished job."""
        raise NotImplementedError


# ------------------------------------------------------------------ OpenAI and OpenAI-compatible (DeepSeek)
class OpenAICompatibleProvider(Provider):
    """Any API that speaks the OpenAI chat-completions format (DeepSeek, ...). Realtime requests only."""
    max_tokens_field = "max_tokens"

    def __init__(self, name, conf):
        super().__init__(name, conf)
        from openai import OpenAI
        self.client = OpenAI(api_key=_api_key(conf), base_url=conf.get("base_url"), timeout=600,
                             max_retries=conf.get("max_retries", 5))

    def request_body(self, system, user):
        c = self.conf
        body = {"model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                self.max_tokens_field: c.get("max_output_tokens", 4000)}
        if c.get("temperature") is not None:
            body["temperature"] = c["temperature"]
        if c.get("reasoning_effort"):
            body["reasoning_effort"] = c["reasoning_effort"]
        if c.get("json_mode") == "schema":
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "skill_label", "strict": True, "schema": RESPONSE_SCHEMA}}
        elif c.get("json_mode") == "object":
            body["response_format"] = {"type": "json_object"}
        body.update(c.get("extra_body", {}))   # any other provider-specific parameter
        return body

    @staticmethod
    def parse_completion(custom_id, completion: dict):
        r = empty_result(custom_id)
        choices = completion.get("choices") or []
        if choices:
            r["text"] = (choices[0].get("message") or {}).get("content")
            if choices[0].get("finish_reason") == "length" and not r["text"]:
                r["error"] = "output token limit reached before the answer (raise max_output_tokens)"
        usage = completion.get("usage") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        out_details = usage.get("completion_tokens_details") or {}
        r["input_tokens"] = usage.get("prompt_tokens", 0) or 0
        # OpenAI: prompt_tokens_details.cached_tokens; DeepSeek: prompt_cache_hit_tokens
        r["cached_input_tokens"] = (prompt_details.get("cached_tokens") or usage.get("prompt_cache_hit_tokens") or 0)
        r["output_tokens"] = usage.get("completion_tokens", 0) or 0       # already includes reasoning tokens
        r["reasoning_tokens"] = out_details.get("reasoning_tokens", 0) or 0
        return r

    def realtime(self, custom_id, system, user):
        t0 = time.time()
        try:
            completion = self.client.chat.completions.create(**self.request_body(system, user))
            r = self.parse_completion(custom_id, completion.model_dump())
        except Exception as e:  # noqa: BLE001  (record the error and continue with the other skills)
            r = empty_result(custom_id, f"{type(e).__name__}: {e}")
        r["latency_s"] = round(time.time() - t0, 2)
        return r


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI, with the Batch API (half price, results within 24 hours)."""
    supports_batch = True
    max_tokens_field = "max_completion_tokens"
    DONE = {"completed", "failed", "expired", "cancelled"}

    def submit_batch(self, requests, work_dir, tag):
        path = work_dir / f"batch_input_{tag}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for cid, system, user in requests:
                f.write(json.dumps({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
                                    "body": self.request_body(system, user)}, ensure_ascii=False) + "\n")
        with open(path, "rb") as fh:
            uploaded = self.client.files.create(file=fh, purpose="batch")
        job = self.client.batches.create(input_file_id=uploaded.id, endpoint="/v1/chat/completions",
                                         completion_window="24h", metadata={"run": tag[:500]})
        return job.id

    def batch_status(self, job_id):
        job = self.client.batches.retrieve(job_id)
        counts = job.request_counts.model_dump() if job.request_counts else {}
        info = {"status": job.status, **counts}
        if job.errors and job.errors.data:
            info["errors"] = [e.message for e in job.errors.data][:5]
        return job.status, job.status in self.DONE, info

    def batch_results(self, job_id):
        job = self.client.batches.retrieve(job_id)
        results = []
        for file_id in (job.output_file_id, job.error_file_id):
            if not file_id:
                continue
            for line in self.client.files.content(file_id).text.splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                cid, resp = item.get("custom_id"), item.get("response") or {}
                if resp.get("status_code") == 200:
                    results.append(self.parse_completion(cid, resp.get("body") or {}))
                else:
                    err = item.get("error") or (resp.get("body") or {}).get("error") or resp
                    results.append(empty_result(cid, f"batch error: {json.dumps(err)[:300]}"))
        return results


# ------------------------------------------------------------------ Google Gemini
GEMINI_SCHEMA = {"type": "OBJECT",
                 "properties": {"label": {"type": "STRING", "enum": LABELS},
                                "confidence": {"type": "NUMBER"},
                                "reason": {"type": "STRING"}},
                 "required": ["label", "confidence", "reason"]}


class GeminiProvider(Provider):
    """Google Gemini, with the Gemini Batch API (half price, results within 24 hours)."""
    supports_batch = True
    DONE = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
            "JOB_STATE_PARTIALLY_SUCCEEDED"}

    def __init__(self, name, conf):
        super().__init__(name, conf)
        from google import genai
        self.client = genai.Client(api_key=_api_key(conf))

    def generation_config(self):
        c = self.conf
        g = {"max_output_tokens": c.get("max_output_tokens", 4000)}
        if c.get("temperature") is not None:
            g["temperature"] = c["temperature"]
        if c.get("json_mode") == "schema":
            g["response_mime_type"] = "application/json"
            g["response_schema"] = GEMINI_SCHEMA
        elif c.get("json_mode") == "object":
            g["response_mime_type"] = "application/json"
        g.update(c.get("extra_generation_config", {}))   # e.g. {"thinking_config": {"thinking_budget": 0}}
        return g

    @staticmethod
    def parse_response(custom_id, resp: dict):
        """resp = a GenerateContentResponse as a dict (snake_case from the SDK, camelCase from batch files)."""
        r = empty_result(custom_id)
        candidates = resp.get("candidates") or []
        if candidates:
            parts = (candidates[0].get("content") or {}).get("parts") or []
            r["text"] = "".join(p.get("text", "") for p in parts if not p.get("thought")) or None
            finish = _get(candidates[0], "finish_reason", "finishReason")
            if not r["text"]:
                r["error"] = f"no answer (finish reason {finish})"
        else:
            feedback = _get(resp, "prompt_feedback", "promptFeedback")
            r["error"] = f"no candidates (blocked?): {json.dumps(feedback)[:200]}"
        usage = _get(resp, "usage_metadata", "usageMetadata", default={})
        r["input_tokens"] = _get(usage, "prompt_token_count", "promptTokenCount", default=0)
        r["cached_input_tokens"] = _get(usage, "cached_content_token_count", "cachedContentTokenCount", default=0)
        thoughts = _get(usage, "thoughts_token_count", "thoughtsTokenCount", default=0)
        answer = _get(usage, "candidates_token_count", "candidatesTokenCount", default=0)
        r["output_tokens"] = answer + thoughts     # Gemini bills thinking tokens as output
        r["reasoning_tokens"] = thoughts
        return r

    def realtime(self, custom_id, system, user):
        from google.genai import types
        t0 = time.time()
        r = None
        for attempt in range(self.conf.get("max_retries", 5)):
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=user,
                    config=types.GenerateContentConfig(system_instruction=system, **self.generation_config()))
                r = self.parse_response(custom_id, resp.model_dump(mode="json", exclude_none=True))
                break
            except Exception as e:  # noqa: BLE001
                r = empty_result(custom_id, f"{type(e).__name__}: {e}")
                time.sleep(min(60, 2 ** attempt * 2))
        r["latency_s"] = round(time.time() - t0, 2)
        return r

    def submit_batch(self, requests, work_dir, tag):
        from google.genai import types
        path = work_dir / f"batch_input_{tag}.jsonl"
        gen = self.generation_config()
        with open(path, "w", encoding="utf-8") as f:
            for cid, system, user in requests:
                f.write(json.dumps({"key": cid, "request": {
                    "system_instruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                    "generation_config": gen}}, ensure_ascii=False) + "\n")
        uploaded = self.client.files.upload(file=str(path),
                                            config=types.UploadFileConfig(display_name=path.stem, mime_type="jsonl"))
        job = self.client.batches.create(model=self.model, src=uploaded.name, config={"display_name": tag[:100]})
        return job.name

    def batch_status(self, job_id):
        job = self.client.batches.get(name=job_id)
        state = job.state.name if job.state else "UNKNOWN"
        info = {"status": state}
        if job.error:
            info["error"] = str(job.error)[:300]
        return state, state in self.DONE, info

    def batch_results(self, job_id):
        job = self.client.batches.get(name=job_id)
        results = []
        if job.dest and job.dest.file_name:
            content = self.client.files.download(file=job.dest.file_name).decode("utf-8")
            for line in content.splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                cid = item.get("key")
                if item.get("response"):
                    results.append(self.parse_response(cid, item["response"]))
                else:
                    results.append(empty_result(cid, f"batch error: {json.dumps(item.get('error', item))[:300]}"))
        elif job.dest and job.dest.inlined_responses:
            for i, ir in enumerate(job.dest.inlined_responses):
                d = ir.model_dump(mode="json", exclude_none=True)
                cid = (d.get("metadata") or {}).get("key", str(i))
                results.append(self.parse_response(cid, d["response"]) if d.get("response")
                               else empty_result(cid, f"batch error: {json.dumps(d.get('error'))[:300]}"))
        return results


PROVIDERS = {"openai": OpenAIProvider, "openai_compatible": OpenAICompatibleProvider, "gemini": GeminiProvider}


def make_provider(name, conf):
    if conf["provider"] not in PROVIDERS:
        raise ValueError(f"{name}: unknown provider {conf['provider']!r}, choose from {list(PROVIDERS)}")
    return PROVIDERS[conf["provider"]](name, conf)
