# KV Cache Workload Bench

A ShareGPT-based multi-scenario workload generation and benchmarking suite for evaluating KV Cache behavior in large language model inference systems.

This project transforms real ShareGPT conversations into structured workloads for testing KV Cache allocation, incremental context growth, prefix reuse, long-context pressure, request concurrency, cache eviction, session lifecycle behavior, multilingual inputs, and external KV offloading to CPU, SSD, USB-attached storage, or remote devices.

The project is designed for systems research and engineering evaluation rather than language-model quality assessment. Its main goal is to provide reproducible and realistic workloads that expose the behavior of KV Cache management policies under different inference conditions.

---
# Usage
Generate:
```bash
python kv_workload_builder.py --sharegpt-path ../ --tokenizer Qwen/Qwen2.5-7B-Instruct --out ./kv_workloads --per-scenario 200 --max-model-len 16384 --target-long-tokens 4096,8192,12000,16000
```


## Motivation

KV Cache is one of the most important runtime data structures in large language model inference.

During autoregressive generation, each request continuously produces key and value tensors that must remain available for later decoding steps. As context lengths and concurrent request counts increase, KV Cache may consume a significant portion of GPU memory. Modern inference systems therefore introduce techniques such as:

* prefix caching;
* multi-turn context reuse;
* paged KV Cache allocation;
* cache eviction;
* CPU KV offloading;
* SSD or remote KV storage;
* asynchronous prefetching;
* cache compression;
* session-aware scheduling.

However, many KV Cache experiments rely only on synthetic prompts with fixed lengths. Synthetic prompts are useful for controlled microbenchmarks, but they cannot fully represent the irregular structure of real conversational workloads.

Real applications contain:

* short and long prompts;
* single-turn and multi-turn conversations;
* repeated prefixes;
* hot and cold sessions;
* multilingual text;
* code and Markdown;
* bursty arrivals;
* session interruptions;
* continuously growing contexts.

This benchmark uses ShareGPT as the primary text source because it contains diverse human-assistant conversations with realistic language, topic, length, and turn distributions.

The generated workloads preserve these characteristics while adding explicit metadata for KV Cache experiments.

---

## Why ShareGPT

ShareGPT is well suited for KV Cache workload construction because it contains real multi-turn conversations collected from interactions with large language models.

Compared with manually generated synthetic prompts, ShareGPT provides several useful properties.

### Real conversational structure

Many records contain multiple alternating user and assistant messages. This makes it possible to construct workloads where each new request extends a previous context.

Such workloads are essential for evaluating:

* incremental prefill;
* multi-turn KV reuse;
* prefix matching;
* session continuation;
* context growth.

### Natural length variation

ShareGPT contains conversations ranging from very short questions to long technical discussions.

This variation allows the benchmark to construct:

* short requests;
* medium-length requests;
* long-context requests;
* mixed-length request streams.

### Diverse content

The dataset includes:

* general conversation;
* technical questions;
* programming code;
* Markdown;
* Chinese;
* English;
* mixed-language content.

This diversity is useful because tokenization behavior affects prompt length, KV Cache size, and prefill cost.

### Reproducibility

ShareGPT is publicly available and can be downloaded independently. Workloads generated from it can therefore be reproduced by other researchers.

---

## Benchmark Scope

The benchmark focuses on system-level KV Cache behavior.

It is intended to evaluate:

* prompt token processing;
* KV Cache allocation;
* prefix cache hit rate;
* external KV hit rate;
* cache eviction;
* KV transfer volume;
* KV load and store latency;
* GPU memory pressure;
* request throughput;
* token throughput;
* time to first token;
* tail latency;
* session cleanup;
* long-context stability.

The benchmark is not intended to evaluate:

* factual correctness;
* reasoning quality;
* instruction-following quality;
* model alignment;
* answer preference;
* benchmark accuracy.

Generated answers may be used for consistency checks, but answer quality is not the primary objective.

---

## Workload Design Principles

The workload suite follows several design principles.

### Realistic input content

Prompt text comes from real conversations instead of repeated filler strings.

### Controlled scenario semantics

Each workload record includes execution metadata such as:

* scenario name;
* phase name;
* phase order;
* sequence number;
* execution mode;
* session identifier;
* reuse group;
* action type;
* expected KV behavior.

This allows the inference entry point to execute workload files as complete scenarios rather than treating them as independent prompts.

### Separation between workload generation and execution

The workload builder is responsible for constructing scenario semantics.

The inference runner is responsible for:

* loading workload files;
* converting messages into token IDs;
* executing phases;
* collecting metrics.

This separation makes the benchmark easier to extend.

### Compatibility with synthetic microbenchmarks

Real workloads do not replace controlled synthetic workloads.

Synthetic hit, miss, and mixed scenarios are still useful for isolating specific behavior. ShareGPT workloads complement them by providing realistic end-to-end traffic.

---

# Workload Scenarios

The benchmark currently includes the following scenarios.

These scenarios were selected because together they cover the major sources of KV Cache pressure and reuse in practical LLM inference systems.

---

## 1. Standard Multi-Turn Conversation

### Purpose

This is the core workload for testing incremental KV Cache growth and conversation-level context reuse.

In a real chat session, each new user request includes previous user and assistant messages. The new prompt therefore shares a long prefix with earlier requests from the same session.

### Construction

For each ShareGPT conversation:

1. The conversation is normalized into alternating user and assistant turns.
2. Conversations with at least two user turns are selected.
3. A sequence of cumulative prompts is created.

Example:

```text
Request 1:
User turn 1

Request 2:
User turn 1
Assistant turn 1
User turn 2

Request 3:
User turn 1
Assistant turn 1
User turn 2
Assistant turn 2
User turn 3
```

All requests from the same conversation share the same `session_id` and `reuse_group`.

### KV Cache behavior

This scenario tests:

* incremental append;
* prefix reuse;
* multi-turn context continuation;
* local prefix cache hits;
* external KV reload;
* reduced prefill computation.

### Why it is representative

Multi-turn chat is one of the most common LLM application patterns, including:

* chat assistants;
* coding assistants;
* customer-service agents;
* interactive search;
* personal assistants.

The cumulative-context structure directly matches the way many production systems construct chat prompts.

---

## 2. Long-Context Workload

### Purpose

This scenario stresses KV Cache capacity, paging, eviction, offloading, and reload behavior.

### Construction

Long conversations are first selected from ShareGPT.

If one conversation is not long enough, multiple conversations are concatenated with explicit separators:

```text
### Source Conversation 1
...

### Source Conversation 2
...

### Source Conversation 3
...
```

The combined text is truncated to target token lengths such as:

* 4,096 tokens;
* 8,192 tokens;
* 12,000 tokens;
* 16,000 tokens.

A short output request is appended so that most execution time and memory pressure come from prefill and KV construction.

### KV Cache behavior

This scenario tests:

* maximum KV capacity;
* paged allocation;
* cache fragmentation;
* cache eviction;
* CPU or SSD offloading;
* remote KV loading;
* out-of-memory behavior;
* context-length boundary handling.

### Why it is representative

Long contexts appear in:

* document summarization;
* repository-level code analysis;
* retrieval-augmented generation;
* long conversations;
* legal document processing;
* research-paper analysis;
* agent memory.

Long-context inference is also one of the most demanding workloads for KV Cache systems.

---

## 3. High-Concurrency Workload

### Purpose

This scenario evaluates KV Cache allocation and scheduling under many independent requests.

### Construction

Independent prompts are sampled from different ShareGPT conversations.

Each request receives:

* a unique request ID;
* a unique session ID;
* an arrival timestamp;
* a configurable output length.

The default arrival process uses exponentially distributed intervals to approximate a Poisson arrival process.

Requests may be submitted in batches with configurable batch size.

### KV Cache behavior

This scenario tests:

* concurrent KV allocation;
* continuous batching;
* scheduler pressure;
* memory contention;
* throughput;
* tail latency;
* request fairness;
* allocation failures.

### Why it is representative

Production LLM services rarely process only one request at a time.

High-concurrency behavior is important for:

* online serving;
* shared inference clusters;
* enterprise APIs;
* multi-user chat services;
* agent platforms.

Even when requests are unrelated, they compete for the same KV Cache pool.

---

## 4. Interleaved Traffic Workload

### Purpose

This scenario simulates heterogeneous production traffic.

A real inference service receives a mixture of request types rather than a single uniform workload.

### Construction

The workload combines:

* short requests;
* medium requests;
* long requests;
* hot repeated sessions;
* cold one-time sessions;
* single-turn prompts;
* multi-turn prompts.

Requests are assigned traffic classes such as:

```text
short_cold
short_hot
long_cold
long_hot_multiturn
```

Arrival times are generated to create an interleaved request stream.

### KV Cache behavior

This scenario tests:

* cache thrashing;
* hot-session protection;
* short-request interference;
* long-request interference;
* eviction policy quality;
* latency isolation;
* fairness between request classes.

### Why it is representative

Online inference traffic is naturally heterogeneous.

For example, a service may simultaneously receive:

* a short translation request;
* a long document summary;
* a repeated chat session;
* a one-time code-generation request.

A system optimized only for uniform prompt lengths may perform poorly under such mixed traffic.

---

## 5. Multilingual Mixed Workload

### Purpose

This scenario evaluates KV Cache and tokenizer behavior for multilingual and code-mixed inputs.

### Construction

Conversations are classified into:

* Chinese;
* English;
* mixed Chinese-English;
* code-related text.

A generated request may concatenate multiple language segments and ask for a bilingual response.

Example:

```text
Chinese conversation segment

English conversation segment

Code-related conversation segment

Please summarize the content in Chinese and English.
```

### KV Cache behavior

This scenario tests:

* language-dependent token expansion;
* tokenizer variability;
* Unicode stability;
* prompt-length estimation;
* mixed-language KV allocation;
* code-token behavior.

### Why it is representative

Modern LLM services are multilingual.

The same number of characters may produce very different token counts across languages. Code, punctuation, and Markdown may also increase token density.

This directly affects:

* KV Cache size;
* prefill latency;
* block allocation;
* context-limit behavior.

---

## 6. Boundary and Abnormal Workload

### Purpose

This scenario validates correctness and robustness at input and session boundaries.

### Construction

The workload includes:

* empty input;
* one-token or very short input;
* punctuation-only input;
* short continuation prompts;
* near-limit contexts;
* contexts around `max_model_len`;
* session interruption markers;
* reconnect markers;
* destroy-session markers.

Near-limit examples are generated at lengths close to:

```text
max_model_len - 512
max_model_len - 128
max_model_len - 1
max_model_len
max_model_len + 1
```

### KV Cache behavior

This scenario tests:

* empty-input handling;
* scheduler overhead;
* truncation behavior;
* maximum-context validation;
* stale KV detection;
* cleanup after interruption;
* session-state robustness.

### Why it is representative

Boundary bugs are common in memory-management systems.

Many failures appear only when:

* the cache is almost full;
* the prompt is empty;
* a session is interrupted;
* a request reaches the exact context limit;
* cached state is destroyed or reused incorrectly.

### Important limitation

Some lifecycle actions, such as `abort` or `destroy`, require explicit support from the inference engine.

The workload records contain action metadata, but the runner must map these actions to actual engine APIs for full lifecycle testing.

---

## 7. Multimodal Workload

### Purpose

This scenario extends KV Cache testing to vision-language models.

### Construction

Image metadata is loaded from an external CSV or image-text dataset.

Each request combines:

* an image path;
* an image question;
* optional answer metadata;
* ShareGPT text context.

Example:

```text
<image>

Describe the image.

Related ShareGPT text context:
...
```

### KV Cache behavior

This scenario tests:

* image-token expansion;
* multimodal prefill cost;
* visual-token KV allocation;
* text-image context interaction;
* multimodal long-context pressure.

### Why it is representative

Vision-language models are increasingly used in:

* document understanding;
* image question answering;
* multimodal assistants;
* visual agents;
* chart analysis.

Image embeddings or visual tokens may substantially increase the effective context length.

### Why it is optional

ShareGPT itself is primarily a text dataset.

A real multimodal workload requires an additional image-text dataset and a model capable of processing images.

Therefore, multimodal files are generated only when image metadata is provided.

---

## 8. Shared-Prefix Workload

### Purpose

This scenario directly evaluates prefix caching and shared-prefix reuse.

### Construction

A long common prefix is created from ShareGPT conversations.

Multiple requests reuse the same prefix but use different suffixes:

```text
Common prefix + summarize the first section
Common prefix + list technical keywords
Common prefix + generate follow-up questions
Common prefix + provide an English summary
```

The first request acts as a warmup request.

Later requests act as replay requests.

### KV Cache behavior

This scenario tests:

* prefix cache hit rate;
* partial-prefix reuse;
* prefix deduplication;
* external prefix loading;
* shared-context memory savings.

### Why it is representative

Shared prefixes occur in:

* repeated system prompts;
* few-shot examples;
* retrieval contexts;
* batch inference;
* prompt templates;
* agent instructions;
* multi-user applications with a common knowledge base.

This is one of the most important workloads for evaluating prefix-aware KV systems.

---

## 9. Continuous Context Growth

### Purpose

This scenario evaluates the behavior of a session whose context grows continuously.

### Construction

A multi-turn ShareGPT conversation is converted into a sequence of cumulative requests.

Unlike the general multi-turn workload, this scenario emphasizes monotonic context growth until the context approaches the model limit.

Each request includes a step number:

```text
step 0
step 1
step 2
...
```

### KV Cache behavior

This scenario tests:

* monotonic KV growth;
* incremental allocation;
* long-lived session behavior;
* context-limit approach;
* eviction under session growth;
* offload of older KV blocks.

### Why it is representative

Long-lived sessions occur in:

* coding agents;
* research assistants;
* autonomous agents;
* persistent chat sessions;
* interactive debugging;
* workflow automation.

These applications may continuously append context for many turns.

---

## 10. Session-Chaos Workload

### Purpose

This scenario stresses session lifecycle management.

### Construction

A stream of session actions is generated using weighted random choices:

* create;
* revisit;
* abort;
* destroy.

Some sessions are accessed repeatedly while others are short-lived.

Each record includes:

* `session_id`;
* action;
* arrival time;
* prompt;
* reuse group.

### KV Cache behavior

This scenario tests:

* session creation;
* session reuse;
* session cleanup;
* stale KV prevention;
* memory leaks;
* cache ownership;
* lifecycle robustness.

### Why it is representative

Production systems contain unstable and incomplete sessions.

Users may:

* close a browser;
* cancel generation;
* reconnect later;
* abandon a request;
* start multiple sessions;
* revisit an old conversation.

A KV Cache system must remain correct under these behaviors.

---

# Why These Scenarios

The current scenarios are designed to cover the major independent dimensions of KV Cache behavior.

| Dimension                  | Covered scenario          |
| -------------------------- | ------------------------- |
| Incremental reuse          | Multi-turn conversation   |
| Maximum capacity           | Long context              |
| Concurrent allocation      | High concurrency          |
| Heterogeneous traffic      | Interleaved traffic       |
| Tokenization diversity     | Multilingual mixed        |
| Correctness boundaries     | Boundary and abnormal     |
| Visual-token pressure      | Multimodal                |
| Cross-request prefix reuse | Shared prefix             |
| Long-lived session growth  | Continuous context growth |
| Session lifecycle          | Session chaos             |

Together, they cover four major categories.

## Reuse behavior

Covered by:

* multi-turn conversation;
* shared prefix;
* continuous context growth;
* hot requests in interleaved traffic.

## Capacity behavior

Covered by:

* long context;
* high concurrency;
* multimodal;
* continuous context growth.

## Scheduling behavior

Covered by:

* high concurrency;
* interleaved traffic;
* session chaos.

## Correctness behavior

Covered by:

* boundary workload;
* session chaos;
* continuous growth;
* multi-turn continuation.

---

# Why There Are No Additional Scenarios Yet

The scenario set is intentionally limited to avoid redundant workloads.

Many possible workload names describe variations of the same underlying KV Cache behavior.

For example:

* “document summarization” is primarily a long-context workload;
* “chatbot workload” is primarily a multi-turn workload;
* “burst traffic” is a parameterization of concurrency;
* “hot-key workload” is represented by shared-prefix and interleaved traffic;
* “code workload” is included in multilingual and content-diversity sampling;
* “retrieval-augmented generation” can be represented by long context and shared prefix;
* “agent workload” is represented by continuous growth and session chaos.

Creating a separate scenario for every application label would increase complexity without necessarily introducing a new KV Cache behavior.

The benchmark therefore organizes workloads according to system behavior rather than application name.

New scenarios should be added only when they introduce a meaningfully different KV Cache access pattern.

Examples that may be added in future versions include:

* controlled cache-hit-ratio workloads;
* explicit external-store warmup and replay;
* multi-tenant isolation;
* KV compression and decompression workloads;
* speculative decoding KV behavior;
* beam-search KV duplication;
* disaggregated prefill/decode workloads;
* cross-model or cross-worker KV reuse;
* failure injection for remote KV storage.

These are not included in the initial version because they often require engine-specific APIs or connector-specific behavior.

---

# Synthetic and Real Workloads

The benchmark supports both synthetic and ShareGPT-based workloads.

## Synthetic workloads

Synthetic workloads such as:

```text
external-hit
external-miss
mixed
```

provide tightly controlled behavior.

They are useful for:

* validating connector correctness;
* creating deterministic cache hits;
* measuring pure lookup overhead;
* isolating store and load latency;
* reproducing experiments.

## ShareGPT workloads

ShareGPT workloads provide realistic text and request structure.

They are useful for:

* end-to-end evaluation;
* realistic prompt distributions;
* multi-turn behavior;
* mixed traffic;
* system-level robustness.

A complete KV Cache evaluation should use both.

Synthetic workloads explain why a system behaves a certain way.

Real workloads show whether the benefit remains under practical traffic.

---

# Workload Output Format

Each JSONL line represents one executable request.

Example:

```json
{
  "request_id": "multi_turn_001_02",
  "scenario": "multi_turn",
  "prompt": "User: ...",
  "messages": [
    {
      "role": "user",
      "content": "..."
    }
  ],
  "max_tokens": 64,
  "session_id": "mt_001",
  "reuse_group": "mt_001",
  "phase": "mt_001",
  "phase_order": 2,
  "sequence_no": 5,
  "execution_mode": "sequential",
  "action": "generate",
  "metadata": {
    "token_len": 1024,
    "kv_goal": "incremental_append_and_context_reuse"
  }
}
```

Important fields:

| Field            | Description                                 |
| ---------------- | ------------------------------------------- |
| `scenario`       | Workload scenario name                      |
| `prompt`         | Plain-text prompt                           |
| `messages`       | Chat-compatible message list                |
| `session_id`     | Logical session identifier                  |
| `reuse_group`    | Requests expected to share reusable KV      |
| `phase`          | Logical execution phase                     |
| `phase_order`    | Phase execution order                       |
| `sequence_no`    | Request order                               |
| `execution_mode` | Sequential, batch, or arrival-based         |
| `action`         | Generate, warmup, replay, abort, or destroy |
| `metadata`       | Scenario-specific information               |

---

# Typical Metrics

The benchmark can be used to collect the following metrics.

## Latency

* end-to-end latency;
* time to first token;
* time per output token;
* inter-token latency;
* P50, P95, and P99 latency.

## Throughput

* requests per second;
* prompt tokens per second;
* output tokens per second;
* total tokens per second.

## KV Cache behavior

* local prefix-cache queries;
* local prefix-cache hits;
* external prefix-cache queries;
* external prefix-cache hits;
* locally computed prompt tokens;
* locally reused prompt tokens;
* externally loaded KV tokens;
* cache eviction count;
* KV store bytes;
* KV load bytes.

## Resource usage

* GPU memory usage;
* CPU memory usage;
* GPU utilization;
* memory utilization;
* external storage bandwidth;
* network bandwidth;
* storage read and write latency.

## Reliability

* failed requests;
* out-of-memory errors;
* invalid context-length errors;
* stale KV reuse;
* session cleanup failures;
* corrupted or missing external KV entries.

---

# Recommended Evaluation Methodology

For each workload, compare multiple KV Cache configurations under identical input data.

Example systems:

* baseline GPU-only KV Cache;
* CPU KV offloading;
* USB-attached KV storage;
* local SSD KV storage;
* remote TCP KV storage;
* alternative eviction policies;
* eager and lazy offloading;
* compressed and uncompressed KV storage.

Each experiment should use:

* the same model;
* the same workload file;
* the same output-token limit;
* the same GPU-memory configuration;
* the same random seed;
* the same request order.

This ensures that observed differences come from KV Cache management rather than input variation.

---

# Limitations

This benchmark has several known limitations.

### ShareGPT does not perfectly match production traffic

The dataset provides realistic text but does not contain real production arrival traces, user think times, or service-level priorities.

Synthetic arrival models are therefore used for concurrency experiments.

### Session actions require runtime support

Fields such as `abort` and `destroy` describe desired behavior, but the inference runner must explicitly map them to runtime APIs.

### Multimodal workloads require additional datasets

ShareGPT alone does not provide images.

### Token counts depend on the tokenizer

Workload construction may use one tokenizer while execution uses another. The runner should re-tokenize prompts with the target model tokenizer.

### Prefix reuse depends on prompt formatting

Different chat templates may produce different token prefixes, even when the underlying messages are identical.

### Model quality is not evaluated

The benchmark focuses on systems behavior rather than answer accuracy.

---

# Summary

KV Cache behavior depends strongly on workload structure.

A useful benchmark must therefore include more than fixed-length random prompts.

This project combines real ShareGPT conversations with explicit scenario construction to cover:

* multi-turn reuse;
* long-context pressure;
* concurrent allocation;
* heterogeneous traffic;
* multilingual tokenization;
* boundary handling;
* multimodal context;
* shared-prefix reuse;
* continuous session growth;
* session lifecycle stress.

The scenario set is intentionally organized around distinct KV Cache behaviors rather than application names.

This makes the benchmark suitable for evaluating:

* vLLM KV Cache implementations;
* CPU KV offloading;
* SSD-based KV storage;
* AI-SSD designs;
* remote KV connectors;
* cache replacement policies;
* prefetching systems;
* compression methods;
* disaggregated inference architectures.

