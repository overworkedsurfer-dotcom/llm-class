# Course plan

This document sets out what each lesson teaches, what it builds, how big its models and
datasets are, and how long it should take to run. It is written before any lesson code
exists. Runtimes marked **est.** are estimates from the throughput benchmarks below. Each
lesson README later records the **measured** results from a real run.

---

## 0. Open questions (please answer before Lesson 01)

1. **Your background, hardware, and time per lesson.** The "About me" section of the brief
   still has its placeholders. Until you say otherwise I assume:
   - you are comfortable with Python and know basic calculus and linear algebra, but are new
     to deep learning;
   - the slowest plausible machine: **CPU only, 4 cores, 16 GB RAM** (the reference machine
     below);
   - 1–2 hours per lesson.

   Anything that fits in 30 minutes on that machine will also fit on a laptop, Apple Silicon,
   or a Colab GPU. If you have a GPU, I will keep the CPU-sized defaults, so everything still
   runs everywhere, and add a `--preset gpu` to the lessons where bigger models visibly help
   (07, 09, 11, 17, 20, 28).
2. **Network access in my build environment.** The sandbox I build and test in blocks
   `huggingface.co` (models and TinyStories) and `openaipublic.blob.core.windows.net`
   (tiktoken's vocabulary files). GitHub and PyPI work. Without Hugging Face access:
   - **Lessons 01, 02 and 04–06:** I can fully run and verify these (they use TinyShakespeare
     from GitHub, or synthetic data).
   - **Lesson 03:** I can verify everything except the tiktoken comparison. I would fetch
     tiktoken's files from a GitHub mirror and check them against the SHA-256 hashes that
     tiktoken pins, so the comparison stays exact.
   - **Lessons 07–09, 20 and 26:** these would need to fall back from TinyStories to
     TinyShakespeare. Samples are worse, and models overfit quickly on 1 MB of text.
   - **Lessons 10–19 and 21–24:** these need pretrained models, so I **cannot run them**.
     Your rule is that every lesson runs and records real output, so I would stop before
     Part 3 rather than ship unverified code.

   To unblock: in the cloud environment settings, add these hosts under **Network access →
   Allowed domains**:

   ```
   huggingface.co, cdn-lfs.huggingface.co, cdn-lfs-us-1.hf.co, cas-bridge.xethub.hf.co,
   cas-server.xethub.hf.co, transfer.xethub.hf.co, openaipublic.blob.core.windows.net
   ```

   Optional: add `arxiv.org` so I can check paper links in FURTHER_READING. Steps:
   https://code.claude.com/docs/en/cloud-environments#network-access.
3. **Capstone scope.** As written, the capstone *pretrains* its own GPT. On CPU in about
   90 minutes, that gives a ~6M-parameter model. It can reliably handle a *narrow*, templated
   set of tool requests, and its free-form chat will be weak. I'll keep the capstone that way
   and say so plainly. I'll also add an optional `--base smollm2` switch that starts the same
   pipeline from SmolLM2-135M for a more capable result. Tell me if you'd rather go another way.

---

## 1. Reference machine and measured throughput

All defaults are sized for the **reference machine (RM)**: the sandbox this repo is built
in. It is an x86-64 machine with 4 vCPUs, 15 GB RAM, no GPU, Python 3.13 and PyTorch
2.14.1. I measured throughput on 2026-10-07 with a scratch benchmark. It times training
steps and KV-cached generation for models with the exact shapes the course uses, filled with
random weights, so it needed no downloads.

| Workload (shapes as used in the course) | Measured on RM |
|---|---|
| 1.1M-param GPT, ctx 128, batch 32: train | 19,600 tokens/s |
| 7.3M-param GPT, ctx 256, vocab 4096, batch 16: train | 4,100 tokens/s (1.0 s/step) |
| 22M-param GPT, ctx 256, vocab 8192, batch 16: train | 1,500 tokens/s (2.8 s/step) |
| 135M Llama-shaped (SmolLM2-135M dims), full fine-tune, B=4, T=256 | 215 tokens/s (4.8 s/step) |
| Same model, LoRA (frozen base, rank-8 adapters on q, v) | 436 tokens/s (2.4 s/step) |
| 135M generation with KV cache, batch 1 | 35 tokens/s |
| 135M generation with KV cache, batch 8 | 119 tokens/s |
| 362M (SmolLM2-360M dims) generation with KV cache, batch 1 | 17 tokens/s |

What this means for sizing:

- **Pretraining from scratch:** 25 minutes on the RM buys roughly **6M tokens** for a
  ~7M-parameter model. TinyStories was designed so models this small can produce coherent
  text, which is why it is the main pretraining corpus.
- **135M fine-tuning:** a 25-minute run buys ~300k tokens of full fine-tuning or ~650k tokens
  with LoRA. Fine-tuning datasets are therefore capped at a few thousand short examples.
- **Generation:** ~120 tokens/s batched, so any lesson that generates data or evaluates by
  sampling stays at a few hundred samples.

---

## 2. Conventions every lesson follows

- **Scripts.** Every script runs end to end, with no placeholders, and has these options:
  - `--quick`: a smoke test that takes ≤ ~90 s on the RM; the test suite runs this;
  - `--device auto|cpu|cuda|mps`;
  - `--seed`;
  - `--out`, defaulting to `runs/NN-name/`.
- **Artifacts.** Trained models and tokenizers that later lessons reuse go to `artifacts/`.
  If a script needs an artifact that doesn't exist yet, it fails with a clear message naming
  the command that creates it. It never trains a hidden fallback.
- **Code style.** Small files, descriptive names, and a tensor-shape comment on every
  non-trivial line (`# (B, T, n_embd)`). No clever tricks. `core/` only gets code once a
  lesson has explained it.
- **Library rule.** A library feature is used only after we have built its equivalent
  ourselves. When we switch, we first check that our version and the library's give the same
  numbers. Examples:
  - `torch.optim.AdamW` after Lesson 02's AdamW matches it;
  - `F.scaled_dot_product_attention` after Lesson 05;
  - `transformers` after Lesson 10 shows our model matches it;
  - `peft` after Lesson 12.
- **Lesson README structure.** Each README covers, in order:
  1. the problem;
  2. intuition;
  3. the math, with shapes;
  4. a code walkthrough;
  5. *Run it*, with the **recorded output from a real run**: machine, wall-clock time, losses,
     samples;
  6. common mistakes;
  7. what changed historically;
  8. *At full scale*, wherever we scaled down;
  9. pointers to exercises and further reading.
- **Exercises.** Each lesson has `exercises.md` with 3–5 exercises, from a one-line tweak to
  an open-ended one. `solutions/` holds runnable solutions for every exercise that has a
  definite answer, and discussion notes for the open-ended ones.
- **Tests.**
  - `tests/core/`: unit tests for every `core/` module, including numerical checks against
    references such as finite differences, `torch.autograd`, `transformers` and `peft`.
  - `tests/lessons/`: one test per lesson that runs every script with `--quick`.
  - Tests that download anything are marked `@pytest.mark.network`.
- **Honesty.** If a run fails or results are poor, the README says so and explains why.
  Numbers in READMEs are never edited by hand. They are pasted from logs that the scripts
  save under `runs/`.

### How each lesson gets built and verified

1. Write the new `core/` modules and their unit tests, and get them passing.
2. Write the lesson scripts, run them with `--quick`, then run the full defaults on the RM.
3. Write the README using the **actual** output and timings, plus `exercises.md` and
   `solutions/`, then run the solutions too.
4. Add the lesson smoke test. Update the root README status, `GLOSSARY.md` and
   `FURTHER_READING.md`.
5. Commit with the message `Lesson NN: <title>` and push.

---

## 3. Dependencies

These are pinned in a single `pyproject.toml`, with optional extras per course part. Versions
are the latest on PyPI as of 2026-10-07. Final pins are confirmed by installing each set
together when its part begins. Several of these (transformers 5.x, datasets 5.x,
huggingface-hub 2.x) are newer than my training data. I will check their APIs against the
installed packages rather than writing from memory.

| Package | Version | Extra | First used | Why |
|---|---|---|---|---|
| torch | 2.14.1 | (base) | 01 | everything |
| numpy | 2.5.3 | (base) | 01 | data wrangling, comparisons |
| matplotlib | 3.11.2 | (base) | 02 | loss curves, decision boundaries, attention maps |
| tiktoken | 0.14.0 | (base) | 03 | compare our BPE with GPT-2 / GPT-4 tokenizers |
| pytest | 9.1.1 | `dev` | 01 | tests |
| transformers | 5.19.0 | `adapt` | 10 | reference implementation we verify against |
| tokenizers | 0.23.2 | `adapt` | 10 | load production tokenizers (after we built BPE) |
| safetensors | 0.8.0 | `adapt` | 10 | read checkpoint files directly |
| huggingface-hub | 2.1.1 | `adapt` | 10 | download models |
| datasets | 5.1.0 | `adapt` | 11 | load instruction / eval datasets |
| peft | 0.21.2 | `adapt` | 12 | library equivalent of our LoRA |
| trl | 1.14.2 | `adapt` (optional) | 13 | library equivalent of our DPO (one short comparison script) |
| gguf | 0.19.0 | `compress` | 22 | write/read GGUF files |
| bitsandbytes | 0.50.2 | `gpu` (optional) | 12, 22 | library 4-bit/8-bit. Mainly CUDA. CPU support is checked when Lesson 12 is built; if it is missing there, that code path is GPU-only and is labelled so. |

We deliberately do **not** add these:

- FastAPI. Lesson 23 serves with the standard library's `http.server`, so nothing is hidden.
- jsonschema. Lesson 15 writes a small validator that Lesson 18 reuses.
- sentence-transformers. Lesson 24 does mean pooling by hand.
- accelerate, unless a pinned library turns out to require it.

---

## 4. Models and datasets

**Pretrained models.** All are openly licensed and ≤ ~500M parameters. Hugging Face is
blocked in my sandbox, so names and configs have been checked through search results and
config files mirrored elsewhere. Each will be verified again against the live repository
when its lesson is built.

| Model | Params | License | Lessons | Why this one |
|---|---|---|---|---|
| `HuggingFaceTB/SmolLM2-135M` | 135M | Apache-2.0 | 10, 11, 22 | Llama architecture (RoPE, RMSNorm, SwiGLU, GQA with 9 query and 3 KV heads, tied embeddings, 30 layers, d=576, vocab 49152). It is **exactly what we build in Lesson 09**, so we can load it into our own code. |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | 135M | Apache-2.0 | 12–14, 17–19, 21, 23 | small instruct model; LoRA runs fit on CPU |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | 362M | Apache-2.0 | 14 (judge), 21 (teacher), 23 (spec-decoding target), 24 (RAG generator) | same tokenizer as 135M, so it can act as teacher, draft partner or judge |
| `openai-community/gpt2` | 124M | MIT | 10 | the classic architecture, which matches our Lesson 06 GPT |
| `Qwen/Qwen2.5-0.5B-Instruct` | 494M | Apache-2.0 | 15, 16 (optional paraphrase teacher) | its chat template has native tool calling, so we can read a real one |
| `sentence-transformers/all-MiniLM-L6-v2` | 22.7M | Apache-2.0 | 24 | small embedding model; we do the pooling ourselves |
| `Qwen/Qwen3-0.6B` *(optional)* | ~0.6B | Apache-2.0 | 25 | real "thinking mode". It is **over** the ~500M budget, so it is optional and marked as such. |

**Datasets.** Download scripts only; nothing large is committed.

| Dataset | Size used | License | Lessons |
|---|---|---|---|
| TinyShakespeare (via `karpathy/char-rnn` on GitHub) | 1.1 MB | public-domain text | 03–06 |
| TinyStories (`roneneldan/TinyStories`) | 19 MB validation file + first ~50 MB of the train file (fetched with an HTTP range request) | CDLA-Sharing-1.0 | 03, 07–09, 20, 26, 28 |
| `HuggingFaceTB/everyday-conversations-llama3.1-2k` | 2.2k conversations | Apache-2.0 | 11–13 |
| `HuggingFaceTB/smol-smoltalk` (subset) | ≤ 5k of 460k examples | Apache-2.0 | 11 (optional larger mix), 13 (prompts) |
| ARC-Easy (`allenai/ai2_arc`, subset) | 200–500 questions | CC BY-SA 4.0 (verify) | 14 |
| Anthropic HH-RLHF, `helpful-base` *(optional)* | ≤ 2k pairs (from GitHub) | MIT (verify) | 13 (real human-preference comparison) |
| **Generated by code in this repo** | — | — | 02 spirals, 05 associative recall, 16 tool calls, 25 arithmetic, 26 shape images, 24 RAG corpus = this repo's own docs |

---

## 5. How `core/` grows

| Lesson | Adds to `core/` |
|---|---|
| 01 | `autograd.py` (scalar `Value` engine) |
| 02 | `optim.py` (SGD, momentum, Adam, AdamW from scratch), `device.py` |
| 03 | `tokenizers/char.py`, `tokenizers/bpe.py` (byte-level BPE: train, encode, decode, save/load) |
| 04 | `data.py` (splits, batch sampling), `metrics.py` (NLL, perplexity) |
| 05 | `model/attention.py` (scaled dot-product, causal mask, multi-head) |
| 06 | `model/layers.py` (LayerNorm, MLP), `model/gpt.py` (`GPTConfig`, `GPT`) |
| 07 | `train.py` (loop, warmup+cosine, clipping, accumulation, AMP, checkpoints), `data.py` (memmap token datasets), `plotting.py` |
| 08 | `sampling.py` (greedy/temperature/top-k/top-p/penalties), KV cache in `model/` |
| 09 | `model/rope.py`, RMSNorm and SwiGLU in `model/layers.py`, GQA and weight tying in `GPTConfig` |
| 10 | `pretrained.py` (load GPT-2 and Llama-family safetensors into our `GPT`) |
| 11 | `chat.py` (chat templates, prompt loss masks, packing) |
| 12 | `lora.py` (LoRALinear, inject, merge), `quant.py` (NF4 to start) |
| 13 | `logprobs.py` (masked sequence log-probs), `dpo.py` |
| 14 | `eval.py` (perplexity, bits per byte, multiple-choice scoring, bootstrap confidence intervals, pairwise judge) |
| 15 | `tools/schema.py` (tool specs and a mini JSON-Schema validator), `tools/format.py` (render tools into prompts, parse calls), `tools/loop.py` |
| 16 | `tools/builtin.py` (calculator, mock weather, unit conversion, file search, sandboxed Python) |
| 17 | `tools/eval.py` (selection accuracy, argument validity, abstention precision and recall) |
| 18 | `constrained.py` (schema to a character-level automaton, plus a token-level mask using a vocabulary trie) |
| 19 | `agent.py` (multi-step loop, tracing, error recovery) |
| 20 | `distill.py` (temperature-scaled KL plus hard-label loss) |
| 21 | `generate_data.py` (batched teacher generation, filtering, dedup) |
| 22 | `quant.py` (int8 absmax and zero-point; per-channel and group-wise 4-bit; packed storage; `QuantLinear`), `export_gguf.py` |
| 23 | `batching.py` (left padding, masks, dynamic batching), `speculative.py`, `server.py` |
| 24 | `embed.py` (mean pooling), `retrieval.py` (brute-force kNN, IVF with our own k-means, BM25) |
| 25 | `rl.py` (GRPO-style policy gradient with verifiable rewards) |
| 26 | `model/moe.py`, long-context RoPE scaling in `model/rope.py`, `model/vision.py` (patch embedding) |

---

## 6. Lessons

Runtimes are for the reference machine (RM): 4-core CPU, no GPU. "est." means estimated from
the benchmark in §1; each README will replace these with measured times. One-time downloads
are listed separately.

### Part 1: Foundations

#### 01 · Tensors, autograd, and a tiny autograd engine
- **Learn:**
  - tensors: shape, dtype, broadcasting, views vs copies, matmul;
  - the computational graph, the chain rule, and why reverse mode is efficient for
    "many inputs, one loss".
- **Build:**
  - a tour of tensors;
  - a scalar autograd engine (`Value`: +, ×, pow, exp, log, tanh, relu, topological-sort
    backward);
  - gradient checks against finite differences and `torch.autograd`;
  - training a 2-16-16-1 MLP (~340 params) on a generated two-moons dataset (200 points)
    using only our engine.
- **Runtime:** < 1 min (est.).

#### 02 · Neural network basics: MLP, loss, backprop, optimizers
- **Learn:**
  - layers as matrix multiplies;
  - deriving softmax cross-entropy, including the log-sum-exp stability trick;
  - backprop by hand for a 2-layer MLP, checked against autograd;
  - initialization and activation statistics;
  - SGD → momentum → RMSProp → Adam → AdamW (and why decoupled weight decay matters).
- **Build:**
  - each optimizer from scratch, raced against the others on the same task;
  - a test showing our AdamW matches `torch.optim.AdamW` to 1e-6.
- **Sizes:** 3-class spiral (900 points); MLP 2-64-64-3 (~4.5k params).
- **Runtime:** ~2 min (est.).

#### 03 · Tokenization
- **Learn:**
  - why models need tokens;
  - Unicode, UTF-8 and bytes;
  - character tokenizers;
  - the byte-level BPE algorithm;
  - GPT-2/GPT-4-style regex pre-splitting;
  - special tokens;
  - compression ratio vs vocabulary size;
  - known tokenizer pitfalls: numbers, leading spaces, glitch tokens, non-English text costing
    more.
- **Build:**
  - a char tokenizer;
  - naive BPE trained on TinyShakespeare, then an efficient incremental BPE;
  - encode/decode with save/load;
  - load GPT-2's published merges into *our* encoder and check its token IDs exactly match
    `tiktoken`'s on a test corpus;
  - a compression comparison: ours at 512/4k/8k vocab vs GPT-2 and cl100k.
- **Sizes:** BPE vocab 512 on 1.1 MB Shakespeare; vocab 4096 on ~10 MB TinyStories.
- **Produces:** `artifacts/tokenizers/tinystories-4k.json` (used by 07+).
- **Runtime:** ~8–10 min (est.; pure-Python BPE training dominates).

#### 04 · Language modeling basics
- **Learn:**
  - next-token prediction and the chain rule of probability;
  - counting bigrams → sampling;
  - negative log-likelihood and perplexity;
  - smoothing;
  - a bigram as a one-layer network (one-hot × W = an embedding lookup);
  - an MLP language model (Bengio 2003);
  - train/val splits and overfitting.
- **Build:**
  - a count-based bigram model;
  - a neural bigram model that matches the count model's loss;
  - an MLP LM with a context of 8 characters;
  - a 2-D map of the learned character embeddings.
- **Sizes:** TinyShakespeare at the character level (vocab 65); MLP LM with embedding 24 and
  hidden layer 256 (~70k params).
- **Runtime:** ~3–5 min (est.).

### Part 2: The transformer

#### 05 · Attention from scratch
- **Learn:**
  - why a fixed context window is limiting;
  - attention as a soft dictionary lookup;
  - Q/K/V;
  - why we scale by √d_head;
  - causal masking with −∞;
  - multi-head attention as a reshape;
  - O(T²) cost.
- **Build:**
  - single-head attention, then masked, then multi-head;
  - checked against `F.scaled_dot_product_attention`;
  - attention heatmaps;
  - an associative-recall toy task ("k3 v7 … k3 → v7") that a 1-layer attention model
    solves and a same-size MLP cannot.
- **Sizes:** vocab 32, sequence length 16, d=64, ≤ 2 layers (~50k params).
- **Runtime:** ~3–5 min (est.).

#### 06 · The full GPT block
- **Learn:**
  - the residual stream;
  - LayerNorm, written by hand;
  - pre-norm vs post-norm;
  - the MLP with GELU;
  - learned positional embeddings;
  - counting parameters by hand and checking the count in code;
  - GPT-2's residual init scaling.
- **Build:**
  - `core/model/gpt.py`;
  - training a char-level GPT on TinyShakespeare;
  - short ablations that remove residuals, normalization, or positions, to see what breaks.
- **Sizes:** 4 layers, 4 heads, d=128, ctx 128 (~0.8M params). At 19.6k tok/s, 5 minutes is
  about 6M characters.
- **Runtime:** ~10 min including ablations (est.).

#### 07 · Training a GPT
- **Learn:**
  - tokenizing once to a uint16 memmap;
  - random windows;
  - AdamW parameter groups (no decay on norms, biases or embeddings);
  - LR warmup and cosine decay;
  - gradient clipping;
  - gradient accumulation;
  - mixed precision (bf16 autocast; fp16 with GradScaler on CUDA), with an honest note that
    on many CPUs it is *not* faster;
  - checkpoint and resume;
  - reading loss curves;
  - tokens/s and the idea of model FLOPs utilization (MFU).
- **Build:**
  - `core/train.py`;
  - pretraining a GPT on TinyStories with our 4k BPE;
  - sample generations at each checkpoint.
- **Sizes:** 6 layers, 8 heads, d=256, ctx 256 (~6.9M params). Default is ~6M tokens. There
  is also a **long run** (~2 h, ~30M tokens) for noticeably better stories.
- **Produces:** `artifacts/ckpt/tinystories-gpt.pt` (reused in 08, 20, 22, 23, 25).
- **Runtime:** ~25 min default (est.). `--quick` ~1 min.

#### 08 · Generation
- **Learn:**
  - logits → probabilities;
  - greedy decoding and why it repeats itself;
  - temperature, top-k, top-p, min-p;
  - repetition, frequency and presence penalties;
  - stop conditions;
  - KV caching: what is cached, position offsets, the memory formula, and why decoding is
    memory-bound.
- **Build:**
  - `core/sampling.py`;
  - a KV cache in our model;
  - a test that cached and uncached logits match;
  - a measured speedup curve vs generated length, plotted.
- **Model:** the Lesson 07 checkpoint.
- **Runtime:** ~3–5 min (est.).

#### 09 · Modern architecture upgrades
- **Learn:**
  - RoPE (rotation, the relative-position property, tested numerically);
  - RMSNorm;
  - SwiGLU (with a parameter-matched hidden size of 8/3·d);
  - grouped-query attention (KV-cache savings);
  - weight tying (parameter savings).
- **Build:** add each upgrade **one at a time**: baseline, +RoPE, +RMSNorm, +SwiGLU, +GQA,
  +tying. Each version trains with the same token budget and seed. The output is a table of
  params, tokens/s, val loss and KV-cache size, plus overlaid loss curves. At this scale some
  differences will be within noise. A `--seeds 3` long run measures that noise, and the README
  reports it honestly.
- **Sizes:** 4 layers, d=192 (~3.3M params), ~2M tokens per run, 6 runs.
- **Produces:** the "modern" `GPTConfig` that Lesson 10 loads SmolLM2 into.
- **Runtime:** ~25 min (est.).

### Part 3: Adapting models (adds `transformers`, `tokenizers`, `safetensors`, `datasets`, `peft`)

#### 10 · Loading pretrained weights into our own model
- **Learn:**
  - what a Hugging Face checkpoint contains (`config.json`, safetensors, `tokenizer.json`);
  - name mapping;
  - gotchas: GPT-2's transposed Conv1D weights, RoPE "half-split" vs interleaved layouts,
    RMSNorm epsilon, RoPE θ, tied heads;
  - how `tokenizer.json` relates to the BPE we built in Lesson 03.
- **Build:**
  - load **GPT-2 124M** into our Lesson-06-style GPT and **SmolLM2-135M** into our
    Lesson-09-style GPT;
  - for both, check the max absolute logit difference against `transformers` (target
    < 1e-4 in fp32) and identical greedy generations.
- **Runtime:** ~3 min (est.). One-time download ~0.8 GB.

#### 11 · Full fine-tuning and instruction tuning
- **Learn:**
  - base vs instruct models;
  - chat templates (SmolLM2 uses ChatML-style `<|im_start|>`/`<|im_end|>`, to be verified);
  - why special tokens need embedding rows;
  - masking the prompt out of the loss (labels = −100) and what goes wrong without it;
  - padding vs packing;
  - catastrophic forgetting.
- **Build:**
  - our template renderer, checked against `tokenizer.apply_chat_template`;
  - a full fine-tune of SmolLM2-135M (base) on everyday-conversations;
  - before/after generations;
  - perplexity on held-out TinyStories before and after, to measure forgetting.
- **Sizes:** ~1,000 examples truncated to 256 tokens, ~250k tokens (~20 min at 215 tok/s).
- **Produces:** `artifacts/ckpt/smollm2-135m-sft/`.
- **Runtime:** ~25 min (est.).

#### 12 · LoRA from scratch, then QLoRA and PEFT
- **Learn:**
  - where fine-tuning memory goes (weights, gradients, Adam states, activations);
  - the low-rank update ΔW = BA;
  - rank r and α/r scaling;
  - B=0 init, so training starts exactly at the base model;
  - which layers to target;
  - merging adapters;
  - QLoRA: NF4 4-bit base weights (built from scratch) with bf16 adapters, and the idea of
    double quantization.
- **Build:**
  - `core/lora.py`;
  - the same task as Lesson 11 with LoRA, comparing trainable params, peak memory, speed and
    quality against full fine-tuning;
  - our own QLoRA run;
  - the **PEFT** equivalent, with a test that our LoRA and PEFT's produce identical outputs
    given the same adapter weights;
  - an optional bitsandbytes QLoRA path (GPU, or CPU if the backend supports it).
- **Runtime:** ~25 min (est.).

#### 13 · Preference tuning: RLHF ideas, then DPO
- **Learn:**
  - why supervised fine-tuning (SFT) isn't enough;
  - the RLHF pipeline (reward model, PPO, KL penalty), explained with diagrams;
  - Bradley–Terry;
  - deriving DPO from the KL-regularized objective;
  - why the reference model matters;
  - failure modes: verbosity, over-optimization, likelihood displacement;
  - IPO, KTO, ORPO and SimPO in brief.
- **Build:**
  - masked sequence log-probs;
  - the DPO loss;
  - LoRA-DPO on SmolLM2-135M-Instruct, where the reference model is the *same* weights with
    adapters disabled (a memory trick).
- **Preference data:** on-policy pairs sampled from the model and ranked by a transparent,
  measurable rule (prefer concise answers that still address the prompt).
- **Measured:** reward margins, preference accuracy, and mean response length before and
  after.
- **Optional:**
  - a run on real human preferences (HH-RLHF);
  - a short TRL `DPOTrainer` equivalence script.
- **Sizes:** ~300 prompts, 2 samples each, ~80 tokens per sample.
- **Runtime:** ~25 min (est.).

#### 14 · Evaluation
- **Learn:**
  - perplexity, and why it can't be compared across tokenizers (bits per byte can);
  - multiple-choice scoring by log-likelihood, as lm-eval-harness does;
  - generation with exact match;
  - prompt-format sensitivity;
  - variance and bootstrap confidence intervals;
  - data contamination;
  - LLM-as-judge: pairwise judging, position bias measured by swapping order, agreement with
    known labels;
  - Goodhart's law.
- **Build:** an eval harness, run on the base model, SFT (11), LoRA (12) and DPO (13) models,
  over ARC-Easy (200 questions), held-out perplexity, and judge comparisons using
  SmolLM2-360M-Instruct as the judge. The judge's measured unreliability is part of the lesson.
- **Runtime:** ~20 min (est.). One-time download ~0.7 GB.

### Part 4: Tool calling

#### 15 · How tool calling works
- **Learn:**
  - tools as JSON schemas;
  - how chat templates render tools into the prompt, by reading Qwen2.5's real template;
  - call tags and special tokens;
  - the call → execute → result (`tool` role) → final-answer loop;
  - parallel calls;
  - robust parsing.
- **Build:**
  - `core/tools/` (specs, mini JSON-Schema validator, renderer, parser, loop);
  - the protocol driven first by a scripted fake model, so it is deterministic and easy to
    inspect;
  - then zero-shot runs of Qwen2.5-0.5B-Instruct (native tool support) and
    SmolLM2-135M-Instruct, to see where small models fail.
- **Runtime:** ~5 min (est.).

#### 16 · Building a tool-calling dataset
- **Learn:**
  - synthetic data strategies: templates with slot filling, executing tools for real to get
    ground-truth results, optional paraphrasing by a teacher;
  - multi-turn examples;
  - **negative examples**: no tool needed, the needed tool isn't available, or the request is
    ambiguous, so the model should ask;
  - splitting by template to prevent leakage;
  - deduplication;
  - schema-validating every example.
- **Build:** a generator for ~3,000 train / 300 val / 300 test examples across 5–6 tools,
  plus dataset statistics.
- **Runtime:** ~2 min (est.); +~15 min for the optional teacher paraphrasing.

#### 17 · Fine-tuning a small model for tool calling
- **Learn:**
  - formatting tool data with prompt-masked loss;
  - LoRA fine-tuning;
  - an eval harness measuring:
    - tool-selection accuracy;
    - argument validity (parses and passes the schema);
    - argument exact match;
    - abstention precision and recall on no-tool cases;
    - end-to-end answer correctness after real execution;
  - error analysis.
- **Build:** a zero-shot baseline → a LoRA fine-tune of SmolLM2-135M-Instruct → a
  side-by-side metrics table.
- **Sizes:** ~3k examples, ~200 tokens each, ≈1 epoch at ~440 tok/s.
- **Produces:** `artifacts/lora/smollm2-135m-tools/`.
- **Runtime:** ~25–30 min (est.).

#### 18 · Constrained decoding
- **Learn:**
  - why sampling produces invalid JSON;
  - masking logits;
  - compiling a JSON schema (the subset we use) into a character-level automaton that tracks
    nesting;
  - lifting it to tokens with a vocabulary trie;
  - performance tricks;
  - how outlines, XGrammar, llama.cpp grammars and API "structured outputs" relate.
- **Build:** `core/constrained.py`; validity rates with and without constraints for the base
  model and the Lesson 17 model; overhead per token.
- **Runtime:** ~10 min (est.).

#### 19 · A minimal agent loop
- **Learn:**
  - an agent as a loop with a step budget;
  - observations fed back to the model;
  - recovering from tool errors and bad JSON;
  - tracing;
  - sandboxing, and its limits: subprocess, timeout, resource limits and a temporary
    directory are **not** a security boundary, and the lesson explains what would be.
- **Build:** `core/agent.py` with real tools (calculator via a safe AST evaluator, file search
  over a sandbox directory, sandboxed Python execution), using the Lesson 17 model plus Lesson
  18 constraints. Runs on a small task set with a success rate, plus an interactive CLI.
- **Runtime:** ~5 min (est.).

### Part 5: Making models smaller and faster (adds `gguf`)

#### 20 · Knowledge distillation
- **Learn:**
  - soft targets and "dark knowledge";
  - temperature;
  - KL(teacher‖student)·T² plus hard cross-entropy;
  - why logit distillation needs a shared tokenizer;
  - token-level vs sequence-level distillation.
- **Build:** a teacher (the Lesson 07 GPT, ~6.9M) and a student (2 layers, d=128, tied,
  ~0.9M). Train the student from scratch, then with distillation, using the same steps and
  data. Compare loss curves, val loss and samples.
- **Produces:** `artifacts/ckpt/tinystories-student.pt` (draft model in Lesson 23).
- **Runtime:** ~20–25 min (est.).

#### 21 · Distillation through synthetic data
- **Learn:**
  - the history: self-instruct, Alpaca, TinyStories itself (GPT-generated) and Phi's
    "textbooks";
  - getting diversity from seed prompts;
  - filtering and deduplication;
  - model-collapse risks;
  - licensing considerations for teacher outputs.
- **Build:** SmolLM2-360M-Instruct (the teacher) generates ~500 one-sentence summaries of
  TinyStories with batched KV-cached generation. Filter them, then LoRA-train
  SmolLM2-135M-Instruct (the student). Evaluate student zero-shot vs distilled vs teacher,
  using Lesson 14's tools.
- **Runtime:** ~25–30 min (est.).

#### 22 · Quantization
- **Learn:**
  - number formats (fp32, bf16, fp16, int8, int4, NF4) and memory math;
  - absmax vs zero-point;
  - per-tensor vs per-channel vs group-wise;
  - outliers and why they hurt (LLM.int8());
  - weight-only vs activation quantization;
  - GPTQ, AWQ and HQQ explained;
  - the GGUF format and llama.cpp quant types (Q8_0, Q4_0, K-quants).
- **Build:**
  - int8 and group-wise 4-bit with real bit-packing;
  - `QuantLinear`;
  - size, perplexity and speed tables for our TinyStories GPT and for SmolLM2-135M;
  - export to GGUF with the `gguf` package, inspect it, and read it back;
  - an optional llama.cpp cross-check.
- **Runtime:** ~10–15 min (est.).

#### 23 · Inference efficiency
- **Learn:**
  - memory-bound vs compute-bound decoding;
  - static batching with left padding and attention masks;
  - throughput vs latency;
  - continuous batching and PagedAttention, as concepts;
  - speculative decoding: the draft/verify acceptance rule that keeps the output distribution
    unchanged.
- **Build:**
  - a measured throughput-vs-batch-size curve;
  - speculative decoding from scratch, run on two pairs: SmolLM2-360M (target) with 135M
    (draft), and our Lesson 07 GPT with the Lesson 20 student;
  - measured acceptance rate and speedup;
  - a standard-library HTTP server with a dynamic-batching worker and streaming;
  - a load-test client reporting p50/p95 latency.
- The README reports measured numbers even if they are disappointing: with a 2× size gap,
  CPU speedups may be modest.
- **Runtime:** ~15 min (est.).

### Part 6: The frontier

#### 24 · Retrieval-augmented generation
- **Learn:**
  - why retrieval helps;
  - embeddings and the intuition behind contrastive training;
  - mean pooling;
  - cosine similarity;
  - brute-force vs approximate search (an IVF index on our own k-means; HNSW explained);
  - BM25 and hybrid search;
  - chunking;
  - grounding with citations;
  - evaluating retrieval (recall@k) separately from generation.
- **Build:** a RAG pipeline over **this repository's own docs**, which works offline and has
  a clean license:
  - MiniLM embeddings (pooled by hand);
  - our index;
  - SmolLM2-360M-Instruct as the generator;
  - a 30-question labelled set with recall@k, plus answer spot checks.
- **Runtime:** ~10 min (est.).

#### 25 · Reasoning and thinking tokens
- **Learn:**
  - chain of thought;
  - why intermediate tokens help (more serial compute per answer);
  - SFT on reasoning traces;
  - RL with verifiable rewards (GRPO);
  - `<think>` formats and thinking budgets;
  - what is established vs debated.
- **Build:** a tiny GPT (~0.8M) on synthetic multi-digit arithmetic:
  1. answer-only vs scratchpad-trace training, with an accuracy comparison;
  2. a small GRPO loop from scratch with a correctness reward, with accuracy before and after;
  3. optional: Qwen3-0.6B thinking vs non-thinking mode, flagged as over the size budget.
- **Runtime:** ~25 min (est.).

#### 26 · Mixture of experts, long context, multimodal
Three self-contained demos, about 8 minutes each (est.).
- **MoE:**
  - router, top-k gating, load-balancing loss and capacity;
  - a dense GPT vs an MoE GPT with the same active parameters on TinyStories;
  - an expert-utilization plot.
- **Long context:**
  - RoPE extrapolation failure;
  - position interpolation and NTK/YaRN-style scaling;
  - sliding-window attention;
  - train at ctx 128, then measure loss by position up to 512 with and without scaling;
  - KV-memory math;
  - FlashAttention and ring attention explained.
- **Multimodal:**
  - ViT patch embedding;
  - a LLaVA-style projection of image tokens into the LM;
  - a tiny model that captions procedurally generated shape images ("a red circle left of a
    blue square"), scored on caption accuracy.

#### 27 · Survey of current developments
- I will run a fresh web search **when this lesson is built**, cite primary sources (papers,
  official model cards, technical reports) and clearly separate *established techniques* from
  *active research* and *claims not yet independently replicated*.
- One small runnable script: a calculator for params, FLOPs and KV-cache memory over
  published model configs, each with a citation.
- **Runtime:** < 1 min.

### Capstone

#### 28 · End-to-end pipeline (**long run**)
One script, every stage reusing `core/`:
1. train a BPE tokenizer on a TinyStories plus synthetic chat/tool mixture;
2. pretrain a modern-architecture GPT (~6M params);
3. instruction-tune it (Lesson 11 template and masking);
4. LoRA tool-calling fine-tune (Lessons 12, 16 and 17), with the eval harness;
5. distill into a ~1M student (Lesson 20);
6. quantize it to 4-bit (Lesson 22);
7. serve it through a chat CLI with working tools and constrained decoding (Lessons 18, 19
   and 23).

- **Runtime:** ~90 min default on the RM (est.). `--quick` ~3 min.
- See open question 3 for honest expectations and the `--base smollm2` option.

---

## 7. Risks and how they are handled

| Risk | Handling |
|---|---|
| Library APIs newer than my training data (transformers 5.x, datasets 5.x, huggingface-hub 2.x) | Check signatures against the installed packages; every library call is covered by a test. |
| Tiny models give weak results (DPO, tool calling, distillation) | Choose measurable targets; report real numbers. If an effect is within noise, say so and show how to scale up. |
| Effects in Lesson 09 are within seed noise | Same seeds and token budget everywhere; an optional multi-seed run; report the spread. |
| bitsandbytes is GPU-centric | Our own NF4/int4 code runs everywhere. Library paths are clearly marked as CUDA-only if that's what the installed version supports. |
| Python sandbox in Lesson 19 is not secure | Stated plainly in the lesson, with a pointer to proper isolation (containers or gVisor). Tools run only on a throwaway directory. |
| Hugging Face blocked in the build sandbox | See open question 2. |
