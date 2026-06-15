#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kv_workload_builder.py

Open-source friendly workload builder for Large Language Model KV Cache testing.

Input:
  - ShareGPT-style raw data directory/file downloaded by:
      git clone https://www.modelscope.cn/datasets/huangjintao/sharegpt.git

Output:
  - JSONL workload files. Each line is a model-ready request object with a unified schema.
  - Optional manifest summary JSON.

Design goals:
  - Standard multi-turn session workloads for KV incremental append and context reuse.
  - Long-context workloads for KV capacity, paging, eviction, offload and reload testing.
  - Concurrency workloads for batching, memory allocation and throughput testing.
  - Interleaved traffic workloads for realistic long/short, hot/cold, single/multi-turn traffic.
  - Multilingual workloads for Chinese/English/code/mixed prompts.
  - Boundary and fault-like workloads for empty, ultra-short, near-limit, abort/reconnect sessions.
  - Prefix-sharing and continuous context growth workloads for prefix cache and session lifecycle tests.
  - Lightweight multimodal request schema that can combine ShareGPT text with image-text datasets.

The script intentionally depends only on Python standard library by default.
For accurate token counting, install transformers and pass --tokenizer:
  pip install transformers
  python kv_workload_builder.py --sharegpt-path ./sharegpt --tokenizer Qwen/Qwen2.5-7B-Instruct --out ./workloads

Author: Your Name / Lab
License: MIT recommended
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------
# Unified output schema
# -----------------------------

@dataclass
class WorkloadRequest:
    """One model-ready request.

    The schema is intentionally compatible with most inference clients:
      - `prompt`: plain text prompt for text-only engines.
      - `messages`: OpenAI/vLLM chat-compatible message list.
      - `images`: optional image metadata for multimodal engines.
      - `session_id`: stable session key for KV reuse experiments.
      - `reuse_group`: logical group for prefix/session reuse analysis.
      - `metadata`: KV-related labels and measurement hints.
    """

    request_id: str
    scenario: str
    prompt: str
    messages: List[Dict[str, Any]]
    max_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    session_id: Optional[str] = None
    reuse_group: Optional[str] = None
    arrival_time: Optional[float] = None
    images: List[Dict[str, Any]] = field(default_factory=list)
    phase: str = "main"
    phase_order: int = 0
    sequence_no: int = 0
    execution_mode: str = "batch"
    action: str = "generate"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedConversation:
    conv_id: str
    source_file: str
    category: str
    turns: List[Dict[str, str]]
    text: str
    char_len: int
    approx_token_len: int
    language: str
    has_code: bool


# -----------------------------
# Token counting
# -----------------------------

class TokenCounter:
    """Tokenizer wrapper.

    If a HuggingFace tokenizer is available, use it.
    Otherwise use a conservative approximate estimator.
    """

    def __init__(self, tokenizer_name: Optional[str] = None):
        self.tokenizer_name = tokenizer_name
        self.tokenizer = None
        if tokenizer_name:
            try:
                from transformers import AutoTokenizer  # type: ignore
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name,
                    trust_remote_code=True,
                    use_fast=True,
                )
                print(f"[INFO] Loaded tokenizer: {tokenizer_name}", file=sys.stderr)
            except Exception as exc:
                print(
                    f"[WARN] Failed to load tokenizer {tokenizer_name}: {exc}. "
                    "Falling back to approximate token counting.",
                    file=sys.stderr,
                )
                self.tokenizer = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        return approx_token_count(text)

    def truncate_to_tokens(self, text: str, target_tokens: int) -> str:
        """Truncate text to about target_tokens.

        Accurate if HF tokenizer is available, approximate otherwise.
        """
        if target_tokens <= 0:
            return ""
        if self.tokenizer is not None:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            ids = ids[:target_tokens]
            return self.tokenizer.decode(ids, skip_special_tokens=False)

        # Approximate fallback: binary search by char length.
        lo, hi = 0, len(text)
        best = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = text[:mid]
            n = self.count(candidate)
            if n <= target_tokens:
                best = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return best


def approx_token_count(text: str) -> int:
    """A simple mixed CJK/English/code token estimator.

    For KV tests, relative length bucketing is often enough before using
    a real tokenizer. Chinese chars roughly count as 1 token, English words
    and punctuation/code chunks are estimated separately.
    """
    if not text:
        return 0
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    ascii_words = len(re.findall(r"[A-Za-z0-9_]+", text))
    punct = len(re.findall(r"[^\w\s\u4e00-\u9fff]", text))
    whitespace_blocks = len(re.findall(r"\s+", text))
    # Code/markdown has many separators; add a small penalty.
    return max(1, int(cjk + ascii_words * 1.25 + punct * 0.35 + whitespace_blocks * 0.05))


# -----------------------------
# Loading and normalization
# -----------------------------

ROLE_MAP = {
    "human": "user",
    "user": "user",
    "prompter": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "bot": "assistant",
    "system": "system",
}


def iter_candidate_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    exts = {".json", ".jsonl", ".ndjson", ".txt", ".gz"}
    excluded = {".git", "__pycache__", ".venv", "venv", "kv_workloads", "workloads", "outputs", "results"}
    for p in path.rglob("*"):
        if p.is_file() and not any(x in excluded for x in p.parts):
            if p.suffix.lower() in exts or p.name.endswith(".jsonl.gz"):
                yield p


def load_json_records(file_path: Path) -> Iterable[Dict[str, Any]]:
    opener = gzip.open if str(file_path).endswith(".gz") else open
    name = file_path.name.lower()
    is_jsonl = name.endswith(".jsonl") or name.endswith(".ndjson") or name.endswith(".jsonl.gz")
    try:
        with opener(file_path, "rt", encoding="utf-8", errors="ignore") as f:  # type: ignore[arg-type]
            if is_jsonl:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as exc:
                        print(f"[WARN] Skip malformed line {file_path}:{line_no}: {exc}", file=sys.stderr)
                        continue
                    if isinstance(obj, dict):
                        yield obj
                return
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                for line in f:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict):
                        yield obj
                return
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        yield item
            elif isinstance(data, dict):
                yield data
    except Exception as exc:
        print(f"[WARN] Skip file {file_path}: {exc}", file=sys.stderr)


def normalize_one_record(obj: Dict[str, Any], source_file: str, idx: int, token_counter: TokenCounter) -> Optional[NormalizedConversation]:
    """Normalize common ShareGPT-like schemas into role/content turns."""
    raw_turns: Any = None
    for key in ("conversation", "conversations", "messages", "dialog", "turns"):
        if key in obj and isinstance(obj[key], list):
            raw_turns = obj[key]
            break
    if raw_turns is None:
        return None

    turns: List[Dict[str, str]] = []
    for t in raw_turns:
        if not isinstance(t, dict):
            continue
        if "human" in t or "assistant" in t:
            human = str(t.get("human") or "").strip()
            assistant = str(t.get("assistant") or "").strip()
            if human:
                turns.append({"role": "user", "content": human})
            if assistant:
                turns.append({"role": "assistant", "content": assistant})
            continue
        raw_role = t.get("from") or t.get("role") or t.get("speaker") or t.get("author") or t.get("name")
        role = ROLE_MAP.get(str(raw_role).lower())
        if role is None:
            continue
        content = t.get("value") or t.get("content") or t.get("text") or t.get("message")
        if content is not None and str(content).strip():
            turns.append({"role": role, "content": str(content).strip()})

    if not turns:
        return None

    # Remove duplicate consecutive roles by merging; KV session tests prefer valid chat alternation.
    merged: List[Dict[str, str]] = []
    for t in turns:
        if merged and merged[-1]["role"] == t["role"]:
            merged[-1]["content"] += "\n\n" + t["content"]
        else:
            merged.append(t)
    turns = merged

    text = render_messages_as_prompt(turns)
    conv_id = str(
        obj.get("conversation_id")
        or obj.get("id")
        or obj.get("uid")
        or stable_id(f"{source_file}:{idx}:{text[:256]}")
    )
    category = str(obj.get("category") or obj.get("source") or "unknown")
    lang = detect_language(text)
    has_code = looks_like_code(text)
    return NormalizedConversation(
        conv_id=conv_id,
        source_file=source_file,
        category=category,
        turns=turns,
        text=text,
        char_len=len(text),
        approx_token_len=token_counter.count(text),
        language=lang,
        has_code=has_code,
    )


def load_sharegpt(path: Path, token_counter: TokenCounter, limit: Optional[int] = None) -> List[NormalizedConversation]:
    records: List[NormalizedConversation] = []
    seen = set()
    file_count = 0
    for fp in iter_candidate_files(path):
        file_count += 1
        for idx, obj in enumerate(load_json_records(fp)):
            conv = normalize_one_record(obj, str(fp), idx, token_counter)
            if conv is None:
                continue
            # Deduplicate by content hash.
            h = stable_id(conv.text[:4096])
            if h in seen:
                continue
            seen.add(h)
            records.append(conv)
            if limit and len(records) >= limit:
                print(f"[INFO] Loaded {len(records)} conversations from {file_count} files", file=sys.stderr)
                return records
    multi_turn = sum(1 for c in records if len(c.turns) >= 3)
    multi_user = sum(1 for c in records if sum(1 for t in c.turns if t["role"] == "user") >= 2)
    print(f"[INFO] Loaded {len(records)} conversations from {file_count} files", file=sys.stderr)
    print(f"[INFO] Conversation stats: >=3 turns={multi_turn}, >=2 user turns={multi_user}", file=sys.stderr)
    return records


# -----------------------------
# Rendering and utilities
# -----------------------------

def stable_id(text: str, n: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def render_messages_as_prompt(messages: Sequence[Dict[str, str]]) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            lines.append(f"System: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"User: {content}")
    return "\n\n".join(lines).strip()


def final_user_prompt(prefix: str) -> Dict[str, str]:
    return {"role": "user", "content": prefix}


def detect_language(text: str) -> str:
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    total = max(1, cjk + latin)
    has_code = looks_like_code(text)
    if has_code:
        return "code"
    if cjk / total > 0.25 and latin / total > 0.25:
        return "mixed"
    if cjk / total > 0.25:
        return "zh"
    return "en"


def looks_like_code(text: str) -> bool:
    patterns = [
        r"```", r"\bdef\s+\w+\(", r"\bclass\s+\w+", r"#include\s*<",
        r"\bfunction\s+\w+\(", r"\bimport\s+\w+", r"\breturn\b", r";\s*$",
    ]
    return any(re.search(p, text, re.MULTILINE) for p in patterns)


def choose(rng: random.Random, items: Sequence[Any], k: int) -> List[Any]:
    if not items:
        return []
    if len(items) <= k:
        return list(items)
    return rng.sample(list(items), k)


def write_jsonl(path: Path, rows: Iterable[WorkloadRequest]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_image_csv(path: Optional[Path]) -> List[Dict[str, str]]:
    """Read optional multimodal image metadata CSV.

    CSV columns accepted:
      image_path, question, answer, source
    Minimal:
      image_path
    """
    if not path:
        return []
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("image_path"):
                rows.append({k: str(v or "") for k, v in row.items()})
    return rows


# -----------------------------
# Workload builders
# -----------------------------

class WorkloadBuilder:
    def __init__(self, conversations: List[NormalizedConversation], token_counter: TokenCounter, seed: int = 42):
        self.convs = conversations
        self.tc = token_counter
        self.rng = random.Random(seed)

    def filter_convs(
        self,
        min_turns: int = 1,
        min_tokens: int = 1,
        max_tokens: Optional[int] = None,
        languages: Optional[Sequence[str]] = None,
        require_code: Optional[bool] = None,
    ) -> List[NormalizedConversation]:
        out = []
        for c in self.convs:
            if len(c.turns) < min_turns:
                continue
            if c.approx_token_len < min_tokens:
                continue
            if max_tokens is not None and c.approx_token_len > max_tokens:
                continue
            if languages and c.language not in set(languages):
                continue
            if require_code is not None and c.has_code != require_code:
                continue
            out.append(c)
        return out

    def annotate_executable_workload(self, scenario: str, rows: List[WorkloadRequest]) -> List[WorkloadRequest]:
        for seq, row in enumerate(rows):
            row.sequence_no = seq
            row.action = str(row.metadata.get("action") or row.metadata.get("phase") or "generate")
            if scenario in {"multi_turn", "continuous_growth"}:
                row.phase = row.session_id or scenario
                row.phase_order = int(row.metadata.get("turn_no", row.metadata.get("step", seq)))
                row.execution_mode = "sequential"
            elif scenario == "boundary":
                row.phase = str(row.metadata.get("case", "boundary"))
                row.phase_order = seq
                row.execution_mode = "sequential"
            elif scenario == "session_chaos":
                row.phase = "session_lifecycle"
                row.phase_order = seq
                row.execution_mode = "sequential"
            elif scenario == "prefix_shared":
                row.phase = "warmup_shared_prefix" if seq == 0 else "replay_shared_prefix"
                row.phase_order = 0 if seq == 0 else 1
                row.execution_mode = "batch"
                row.action = "warmup" if seq == 0 else "replay"
            elif scenario == "interleaved":
                row.phase = "interleaved_traffic"
                row.execution_mode = "arrival"
            elif scenario == "concurrency":
                row.phase = "concurrent_batch"
                row.execution_mode = "batch"
            elif scenario == "long_context":
                target = row.metadata.get("target_tokens", 0)
                row.phase = f"long_context_{target}"
                row.phase_order = int(target) if isinstance(target, int) else 0
                row.execution_mode = "batch"
            elif scenario == "multilingual":
                row.phase = "multilingual_batch"
            elif scenario == "multimodal":
                row.phase = "multimodal_batch"
            else:
                row.phase = scenario
        return rows

    def build_all(
        self,
        outdir: Path,
        per_scenario: int,
        max_model_len: int,
        target_long_tokens: Sequence[int],
        image_rows: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, int]:
        outdir.mkdir(parents=True, exist_ok=True)
        manifest: Dict[str, int] = {}

        scenarios = {
            "multi_turn": self.build_multi_turn(per_scenario=per_scenario, max_model_len=max_model_len),
            "long_context": self.build_long_context(per_scenario=per_scenario, target_tokens=target_long_tokens, max_model_len=max_model_len),
            "concurrency": self.build_concurrency(per_scenario=per_scenario, max_model_len=max_model_len),
            "interleaved": self.build_interleaved(per_scenario=per_scenario, max_model_len=max_model_len),
            "multilingual": self.build_multilingual(per_scenario=per_scenario, max_model_len=max_model_len),
            "boundary": self.build_boundary(per_scenario=per_scenario, max_model_len=max_model_len),
            "prefix_shared": self.build_prefix_shared(per_scenario=per_scenario, max_model_len=max_model_len),
            "continuous_growth": self.build_continuous_growth(per_scenario=per_scenario, max_model_len=max_model_len),
            "session_chaos": self.build_session_chaos(per_scenario=per_scenario, max_model_len=max_model_len),
        }
        if image_rows:
            scenarios["multimodal"] = self.build_multimodal(
                image_rows=image_rows,
                per_scenario=per_scenario,
                max_model_len=max_model_len,
            )

        for name, rows in scenarios.items():
            rows = self.annotate_executable_workload(name, rows)
            scenarios[name] = rows
            path = outdir / f"{name}.jsonl"
            count = write_jsonl(path, rows)
            manifest[name] = count
            print(f"[INFO] Wrote {count:6d} rows -> {path}", file=sys.stderr)

        all_rows: List[WorkloadRequest] = []
        for name in scenarios:
            with (outdir / f"{name}.jsonl").open("r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    all_rows.append(WorkloadRequest(**obj))
        # Sort by scenario then arrival_time if present.
        all_rows.sort(key=lambda r: (r.scenario, r.arrival_time if r.arrival_time is not None else 0.0, r.request_id))
        manifest["all"] = write_jsonl(outdir / "all_scenarios.jsonl", all_rows)

        summary = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_source_conversations": len(self.convs),
            "max_model_len": max_model_len,
            "target_long_tokens": list(target_long_tokens),
            "files": manifest,
            "schema": list(WorkloadRequest.__dataclass_fields__.keys()),
        }
        with (outdir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return manifest

    # 1. Standard multi-turn dialogue
    def build_multi_turn(self, per_scenario: int, max_model_len: int) -> List[WorkloadRequest]:
        candidates = self.filter_convs(min_turns=3, min_tokens=64, max_tokens=max_model_len)
        self.rng.shuffle(candidates)
        rows: List[WorkloadRequest] = []
        for c in candidates:
            user_indices = [i for i, t in enumerate(c.turns) if t["role"] == "user"]
            if len(user_indices) < 2:
                continue
            session = f"mt_{stable_id(c.conv_id)}"
            # Build rolling prompts ending at each user turn. This simulates multi-turn continuation.
            for turn_no, idx in enumerate(user_indices[:6], 1):
                msgs = c.turns[: idx + 1]
                prompt = render_messages_as_prompt(msgs)
                n_tok = self.tc.count(prompt)
                if n_tok > max_model_len - 128:
                    continue
                rows.append(WorkloadRequest(
                    request_id=f"multi_turn_{stable_id(c.conv_id)}_{turn_no:02d}",
                    scenario="multi_turn",
                    prompt=prompt,
                    messages=msgs,
                    session_id=session,
                    reuse_group=session,
                    max_tokens=64,
                    metadata={
                        "source_conv_id": c.conv_id,
                        "turn_no": turn_no,
                        "token_len": n_tok,
                        "kv_goal": "incremental_append_and_context_reuse",
                        "expected_behavior": "later turns should reuse earlier session/prefix KV when engine supports it",
                    },
                ))
                if len(rows) >= per_scenario:
                    return rows
        return rows

    # 2. Long context: long conversations + automatic concatenation
    def build_long_context(self, per_scenario: int, target_tokens: Sequence[int], max_model_len: int) -> List[WorkloadRequest]:
        pool = self.filter_convs(min_turns=1, min_tokens=64)
        pool.sort(key=lambda c: c.approx_token_len, reverse=True)
        if not pool:
            return []
        rows: List[WorkloadRequest] = []
        target_list = list(target_tokens) or [4096, 8192, 16384]
        target_list = [t for t in target_list if t > 0]
        i = 0
        while len(rows) < per_scenario:
            target = target_list[len(rows) % len(target_list)]
            target = min(target, max_model_len - 64)
            segments = []
            cur_tokens = 0
            # Prefer long samples first, but rotate to avoid all prompts being identical.
            start = i % len(pool)
            for j in range(len(pool)):
                c = pool[(start + j) % len(pool)]
                seg = f"\n\n### Source Conversation {j + 1}\n{c.text}"
                seg_tokens = self.tc.count(seg)
                if cur_tokens + seg_tokens > target and segments:
                    break
                segments.append(seg)
                cur_tokens += seg_tokens
                if cur_tokens >= target:
                    break
            i += 1
            body = "".join(segments)
            body = self.tc.truncate_to_tokens(body, max(1, target - 32))
            msgs = [
                {"role": "user", "content": body + "\n\n请仅用一句话总结以上所有对话的核心主题。"}
            ]
            prompt = render_messages_as_prompt(msgs)
            n_tok = self.tc.count(prompt)
            rows.append(WorkloadRequest(
                request_id=f"long_context_{len(rows):06d}_{n_tok}tok",
                scenario="long_context",
                prompt=prompt,
                messages=msgs,
                session_id=None,
                reuse_group=f"longctx_{target}",
                max_tokens=32,
                metadata={
                    "target_tokens": target,
                    "token_len": n_tok,
                    "construction": "sorted_long_conversations_plus_concatenation",
                    "kv_goal": "capacity_paging_eviction_offload_reload",
                },
            ))
        return rows

    # 3. High concurrency: independent requests with arrival queue metadata
    def build_concurrency(self, per_scenario: int, max_model_len: int) -> List[WorkloadRequest]:
        pool = self.filter_convs(min_turns=1, min_tokens=16, max_tokens=max_model_len - 128)
        self.rng.shuffle(pool)
        rows: List[WorkloadRequest] = []
        t = 0.0
        for idx, c in enumerate(pool[:per_scenario]):
            # Use first user turn if possible; otherwise full text.
            user_turns = [m for m in c.turns if m["role"] == "user"]
            msgs = [user_turns[0]] if user_turns else [final_user_prompt(c.text)]
            prompt = render_messages_as_prompt(msgs)
            n_tok = self.tc.count(prompt)
            t += self.rng.expovariate(10.0)  # synthetic Poisson-like arrivals, 10 req/s by default
            rows.append(WorkloadRequest(
                request_id=f"concurrency_{idx:06d}",
                scenario="concurrency",
                prompt=prompt,
                messages=msgs,
                session_id=f"conc_{idx:06d}",
                reuse_group=None,
                arrival_time=round(t, 6),
                max_tokens=self.rng.choice([16, 32, 64, 128]),
                metadata={
                    "source_conv_id": c.conv_id,
                    "token_len": n_tok,
                    "arrival_model": "poisson_like_expovariate_rate_10",
                    "kv_goal": "batching_allocation_throughput_tail_latency",
                },
            ))
        return rows

    # 4. Interleaved traffic: long/short, hot/cold, single/multi-turn
    def build_interleaved(self, per_scenario: int, max_model_len: int) -> List[WorkloadRequest]:
        short_pool = self.filter_convs(min_turns=1, min_tokens=1, max_tokens=256)
        med_pool = self.filter_convs(min_turns=2, min_tokens=256, max_tokens=min(2048, max_model_len - 128))
        long_pool = self.filter_convs(min_turns=2, min_tokens=2048, max_tokens=max_model_len - 128)
        if not long_pool:
            long_pool = self.filter_convs(min_turns=2, min_tokens=512, max_tokens=max_model_len - 128)
        hot_short = choose(self.rng, short_pool or med_pool, max(1, per_scenario // 20))
        hot_long = choose(self.rng, long_pool or med_pool, max(1, per_scenario // 20))
        cold_pool = choose(self.rng, (short_pool + med_pool + long_pool), per_scenario)

        rows: List[WorkloadRequest] = []
        t = 0.0
        for i in range(per_scenario):
            r = self.rng.random()
            label = ""
            if r < 0.30 and short_pool:
                c = self.rng.choice(short_pool)
                msgs = self._first_user_messages(c)
                label = "short_cold"
                session = f"cold_{i:06d}"
            elif r < 0.50 and hot_short:
                c = self.rng.choice(hot_short)
                msgs = self._first_user_messages(c)
                label = "short_hot"
                session = f"hot_short_{stable_id(c.conv_id)}"
            elif r < 0.70 and cold_pool:
                c = self.rng.choice(cold_pool)
                msgs = [final_user_prompt(self.tc.truncate_to_tokens(c.text, min(max_model_len - 128, max(512, c.approx_token_len))))]
                label = "long_cold"
                session = f"cold_long_{i:06d}"
            else:
                c = self.rng.choice(hot_long or cold_pool)
                user_indices = [j for j, m in enumerate(c.turns) if m["role"] == "user"]
                if user_indices:
                    upto = user_indices[min(i % len(user_indices), len(user_indices) - 1)]
                    msgs = c.turns[: upto + 1]
                else:
                    msgs = [final_user_prompt(c.text)]
                label = "long_hot_multiturn"
                session = f"hot_long_{stable_id(c.conv_id)}"

            prompt = render_messages_as_prompt(msgs)
            if self.tc.count(prompt) > max_model_len - 64:
                prompt = self.tc.truncate_to_tokens(prompt, max_model_len - 64)
                msgs = [final_user_prompt(prompt)]
            n_tok = self.tc.count(prompt)
            t += self.rng.expovariate(8.0)
            rows.append(WorkloadRequest(
                request_id=f"interleaved_{i:06d}_{label}",
                scenario="interleaved",
                prompt=prompt,
                messages=msgs,
                session_id=session,
                reuse_group=session if "hot" in label else None,
                arrival_time=round(t, 6),
                max_tokens=self.rng.choice([16, 32, 64]),
                metadata={
                    "traffic_class": label,
                    "token_len": n_tok,
                    "kv_goal": "hot_cold_long_short_eviction_and_tail_latency",
                },
            ))
        return rows

    # 5. Multilingual mixed: zh/en/code/mixed alternating prompts
    def build_multilingual(self, per_scenario: int, max_model_len: int) -> List[WorkloadRequest]:
        zh = self.filter_convs(min_turns=1, min_tokens=16, max_tokens=max_model_len // 3, languages=["zh", "mixed"])
        en = self.filter_convs(min_turns=1, min_tokens=16, max_tokens=max_model_len // 3, languages=["en"])
        code = self.filter_convs(min_turns=1, min_tokens=16, max_tokens=max_model_len // 3, languages=["code"])
        rows: List[WorkloadRequest] = []
        if not (zh or en or code):
            return rows
        for i in range(per_scenario):
            parts = []
            if zh:
                parts.append("请阅读下面中文对话片段：\n" + self.rng.choice(zh).text)
            if en:
                parts.append("Now read the following English dialogue snippet:\n" + self.rng.choice(en).text)
            if code:
                parts.append("Finally inspect the following code-related dialogue:\n" + self.rng.choice(code).text)
            if not parts:
                continue
            content = "\n\n---\n\n".join(parts)
            content += "\n\n请先用中文总结，再用 English summarize the key point."
            content = self.tc.truncate_to_tokens(content, max_model_len - 128)
            msgs = [{"role": "user", "content": content}]
            prompt = render_messages_as_prompt(msgs)
            rows.append(WorkloadRequest(
                request_id=f"multilingual_{i:06d}",
                scenario="multilingual",
                prompt=prompt,
                messages=msgs,
                session_id=f"ml_{i:06d}",
                reuse_group="multilingual_mix",
                max_tokens=96,
                metadata={
                    "languages_used": [x for x, pool in [("zh", zh), ("en", en), ("code", code)] if pool],
                    "token_len": self.tc.count(prompt),
                    "kv_goal": "tokenizer_language_mix_unicode_and_kv_consistency",
                },
            ))
        return rows

    # 6. Boundary and abnormal cases
    def build_boundary(self, per_scenario: int, max_model_len: int) -> List[WorkloadRequest]:
        rows: List[WorkloadRequest] = []
        short_inputs = ["", "hi", "?", "继续", "continue", "请继续", "OK", "1"]
        for i, s in enumerate(short_inputs):
            rows.append(WorkloadRequest(
                request_id=f"boundary_short_{i:02d}",
                scenario="boundary",
                prompt=s,
                messages=[{"role": "user", "content": s}],
                session_id=f"boundary_short_{i:02d}",
                max_tokens=8,
                metadata={
                    "case": "empty_or_ultra_short_input",
                    "token_len": self.tc.count(s),
                    "kv_goal": "scheduler_overhead_and_empty_input_handling",
                },
            ))

        # Near-limit contexts.
        pool = self.filter_convs(min_turns=1, min_tokens=64)
        base = "\n\n".join(c.text for c in choose(self.rng, pool, min(64, len(pool))))
        for offset in [512, 128, 1, 0, -1]:
            target = max_model_len - offset
            if target <= 0:
                continue
            content = self.tc.truncate_to_tokens(base, target)
            msgs = [{"role": "user", "content": content + "\n\n请回答：边界测试完成。"}]
            prompt = render_messages_as_prompt(msgs)
            rows.append(WorkloadRequest(
                request_id=f"boundary_near_limit_{target}tok",
                scenario="boundary",
                prompt=prompt,
                messages=msgs,
                session_id=f"boundary_limit_{target}",
                max_tokens=16,
                metadata={
                    "case": "near_or_over_max_model_len",
                    "target_tokens": target,
                    "token_len": self.tc.count(prompt),
                    "kv_goal": "limit_check_oom_truncation_error_handling",
                },
            ))

        # Abort/reconnect lifecycle markers.
        mt = self.build_multi_turn(per_scenario=10, max_model_len=max_model_len)
        for j, r in enumerate(mt[: max(1, min(10, per_scenario))]):
            sid = f"abort_reconnect_{j:04d}"
            for phase in ["start", "abort_after_prefill", "reconnect_continue", "destroy_session"]:
                obj = WorkloadRequest(
                    request_id=f"boundary_{sid}_{phase}",
                    scenario="boundary",
                    prompt=r.prompt,
                    messages=r.messages,
                    session_id=sid,
                    reuse_group=sid,
                    max_tokens=32,
                    metadata={
                        "case": "session_interrupt_reconnect",
                        "phase": phase,
                        "token_len": self.tc.count(r.prompt),
                        "kv_goal": "abort_cleanup_stale_kv_and_reconnect_consistency",
                    },
                )
                rows.append(obj)
                if len(rows) >= per_scenario:
                    return rows
        return rows[:per_scenario]

    # 7. Multimodal: combine image metadata with ShareGPT text.
    def build_multimodal(self, image_rows: List[Dict[str, str]], per_scenario: int, max_model_len: int) -> List[WorkloadRequest]:
        rows: List[WorkloadRequest] = []
        if not image_rows:
            return rows
        text_pool = self.filter_convs(min_turns=1, min_tokens=16, max_tokens=max_model_len // 2)
        for i in range(per_scenario):
            img = self.rng.choice(image_rows)
            text_context = self.rng.choice(text_pool).text if text_pool else ""
            question = img.get("question") or "请描述这张图片，并结合下面的文本上下文回答。"
            answer_hint = img.get("answer", "")
            content = (
                "<image>\n"
                f"图像问题：{question}\n\n"
                "文本上下文来自 ShareGPT：\n"
                f"{text_context}\n\n"
                "请综合图像和文本上下文，给出简洁回答。"
            )
            content = self.tc.truncate_to_tokens(content, max_model_len - 128)
            msgs = [{"role": "user", "content": content}]
            prompt = render_messages_as_prompt(msgs)
            rows.append(WorkloadRequest(
                request_id=f"multimodal_{i:06d}",
                scenario="multimodal",
                prompt=prompt,
                messages=msgs,
                images=[{
                    "image_path": img.get("image_path", ""),
                    "source": img.get("source", ""),
                }],
                session_id=f"mm_{i:06d}",
                reuse_group="multimodal_text_image",
                max_tokens=64,
                metadata={
                    "token_len": self.tc.count(prompt),
                    "answer_hint": answer_hint[:256],
                    "kv_goal": "image_token_plus_text_kv_allocation_multimodal_context",
                    "note": "Use with a VLM engine that understands <image> and image_path metadata.",
                },
            ))
        return rows

    # 8a. Prefix sharing: many requests share a long common prefix, differ only at tail.
    def build_prefix_shared(self, per_scenario: int, max_model_len: int) -> List[WorkloadRequest]:
        pool = self.filter_convs(min_turns=1, min_tokens=64)
        if not pool:
            return []
        prefix_text = "\n\n".join(c.text for c in choose(self.rng, pool, min(8, len(pool))))
        prefix_text = self.tc.truncate_to_tokens(prefix_text, max(64, max_model_len // 2))
        suffixes = [
            "请总结第一段内容。",
            "Please summarize the above conversation in English.",
            "请提取其中的技术关键词。",
            "List three possible follow-up questions.",
            "请判断上述内容是否包含代码问题。",
        ]
        rows = []
        for i in range(per_scenario):
            suffix = suffixes[i % len(suffixes)]
            content = f"{prefix_text}\n\n### New User Request\n{suffix}"
            msgs = [{"role": "user", "content": content}]
            prompt = render_messages_as_prompt(msgs)
            rows.append(WorkloadRequest(
                request_id=f"prefix_shared_{i:06d}",
                scenario="prefix_shared",
                prompt=prompt,
                messages=msgs,
                session_id=f"prefix_session_{i:06d}",
                reuse_group="shared_prefix_group_0",
                max_tokens=48,
                metadata={
                    "token_len": self.tc.count(prompt),
                    "shared_prefix_tokens": self.tc.count(prefix_text),
                    "kv_goal": "prefix_cache_hit_partial_hit_and_deduplication",
                },
            ))
        return rows

    # 8b. Continuous growth: one session keeps appending new turns until near limit.
    def build_continuous_growth(self, per_scenario: int, max_model_len: int) -> List[WorkloadRequest]:
        pool = self.filter_convs(min_turns=2, min_tokens=64)
        self.rng.shuffle(pool)
        rows: List[WorkloadRequest] = []
        for sidx, c in enumerate(pool):
            session = f"growth_{stable_id(c.conv_id)}"
            msgs: List[Dict[str, str]] = []
            step = 0
            for t in c.turns:
                msgs.append(t)
                if t["role"] != "user":
                    continue
                prompt = render_messages_as_prompt(msgs)
                n_tok = self.tc.count(prompt)
                if n_tok > max_model_len - 128:
                    break
                rows.append(WorkloadRequest(
                    request_id=f"continuous_growth_{sidx:04d}_{step:02d}",
                    scenario="continuous_growth",
                    prompt=prompt,
                    messages=list(msgs),
                    session_id=session,
                    reuse_group=session,
                    max_tokens=32,
                    metadata={
                        "step": step,
                        "token_len": n_tok,
                        "kv_goal": "monotonic_context_growth_incremental_kv_append_until_limit",
                    },
                ))
                step += 1
                if len(rows) >= per_scenario:
                    return rows
        return rows

    # 8c. Session chaos: create/destroy/revisit many sessions randomly.
    def build_session_chaos(self, per_scenario: int, max_model_len: int) -> List[WorkloadRequest]:
        base = self.build_multi_turn(per_scenario=max(per_scenario, 50), max_model_len=max_model_len)
        rows: List[WorkloadRequest] = []
        live_sessions: List[str] = []
        for i in range(per_scenario):
            if base:
                r = self.rng.choice(base)
                msgs = r.messages
                prompt = r.prompt
            else:
                msgs = [{"role": "user", "content": "hello"}]
                prompt = "hello"
            action = self.rng.choices(
                ["create", "revisit", "abort", "destroy"],
                weights=[0.45, 0.35, 0.10, 0.10],
            )[0]
            if action == "create" or not live_sessions:
                sid = f"chaos_{i:06d}_{stable_id(prompt)}"
                live_sessions.append(sid)
            else:
                sid = self.rng.choice(live_sessions)
                if action == "destroy" and sid in live_sessions:
                    live_sessions.remove(sid)
            rows.append(WorkloadRequest(
                request_id=f"session_chaos_{i:06d}_{action}",
                scenario="session_chaos",
                prompt=prompt,
                messages=msgs,
                session_id=sid,
                reuse_group=sid,
                arrival_time=round(i * 0.05, 6),
                max_tokens=32,
                metadata={
                    "action": action,
                    "token_len": self.tc.count(prompt),
                    "kv_goal": "session_lifecycle_cleanup_stale_kv_memory_leak_detection",
                },
            ))
        return rows

    def _first_user_messages(self, c: NormalizedConversation) -> List[Dict[str, str]]:
        for m in c.turns:
            if m["role"] == "user":
                return [m]
        return [final_user_prompt(c.text)]


# -----------------------------
# CLI and validation
# -----------------------------

def parse_int_list(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build multi-scenario ShareGPT workloads for LLM KV Cache benchmarking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sharegpt-path", type=Path, required=True, help="Path to cloned ShareGPT dataset directory or file.")
    p.add_argument("--out", type=Path, default=Path("./kv_workloads"), help="Output directory.")
    p.add_argument("--tokenizer", type=str, default=None, help="Optional HF tokenizer name/path for accurate token counting.")
    p.add_argument("--limit-source", type=int, default=None, help="Optional max number of source conversations to load.")
    p.add_argument("--per-scenario", type=int, default=200, help="Target number of requests per scenario.")
    p.add_argument("--max-model-len", type=int, default=8192, help="Model context limit used for construction/truncation.")
    p.add_argument("--target-long-tokens", type=str, default="4096,8192,16384", help="Comma-separated target lengths for long-context prompts.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--image-csv", type=Path, default=None, help="Optional CSV with image_path,question,answer,source for multimodal workloads.")
    p.add_argument("--scenario", type=str, default="all", choices=[
        "all", "multi_turn", "long_context", "concurrency", "interleaved", "multilingual",
        "boundary", "multimodal", "prefix_shared", "continuous_growth", "session_chaos",
    ], help="Build only one scenario or all.")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    token_counter = TokenCounter(args.tokenizer)
    conversations = load_sharegpt(args.sharegpt_path, token_counter, args.limit_source)
    if not conversations:
        print("[ERROR] No valid ShareGPT conversations found. Check --sharegpt-path and file schema.", file=sys.stderr)
        return 2

    outdir: Path = args.out
    builder = WorkloadBuilder(conversations, token_counter, seed=args.seed)
    target_long_tokens = parse_int_list(args.target_long_tokens)
    image_rows = read_image_csv(args.image_csv)

    if args.scenario == "all":
        builder.build_all(
            outdir=outdir,
            per_scenario=args.per_scenario,
            max_model_len=args.max_model_len,
            target_long_tokens=target_long_tokens,
            image_rows=image_rows,
        )
    else:
        fn_map = {
            "multi_turn": lambda: builder.build_multi_turn(args.per_scenario, args.max_model_len),
            "long_context": lambda: builder.build_long_context(args.per_scenario, target_long_tokens, args.max_model_len),
            "concurrency": lambda: builder.build_concurrency(args.per_scenario, args.max_model_len),
            "interleaved": lambda: builder.build_interleaved(args.per_scenario, args.max_model_len),
            "multilingual": lambda: builder.build_multilingual(args.per_scenario, args.max_model_len),
            "boundary": lambda: builder.build_boundary(args.per_scenario, args.max_model_len),
            "multimodal": lambda: builder.build_multimodal(image_rows, args.per_scenario, args.max_model_len),
            "prefix_shared": lambda: builder.build_prefix_shared(args.per_scenario, args.max_model_len),
            "continuous_growth": lambda: builder.build_continuous_growth(args.per_scenario, args.max_model_len),
            "session_chaos": lambda: builder.build_session_chaos(args.per_scenario, args.max_model_len),
        }
        rows = fn_map[args.scenario]()
        rows = builder.annotate_executable_workload(args.scenario, rows)
        count = write_jsonl(outdir / f"{args.scenario}.jsonl", rows)
        print(f"[INFO] Wrote {count} rows -> {outdir / f'{args.scenario}.jsonl'}", file=sys.stderr)

    print("[DONE] Workload generation finished.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
