"""
Benchmark harness for the LLS response-scoring pass: old vs new implementation.

Two scoring kernels as separate functions, one shared driver, independent knobs.
The knobs matter for attribution: `dense` vs `sparse` isolates the
vocab-projection change, while batch_size and sort_by_length isolate the two
changes the sparse kernel merely *enables*. Bundling all three into one A/B
tells you the bundle is faster without telling you which change did it.

  score_dense   OLD -- current repo behaviour, math copied verbatim
  score_sparse  NEW -- vocab projection at scored positions only
  score_all     shared driver; kernel / batch_size / sort_by_length are separate

Usage
-----
  python bench_logprob.py --repo ~/path/to/logit-linear-selection --config baseline -n 2000
  python bench_logprob.py --repo ~/path/to/logit-linear-selection --all -n 2000

IMPORTANT: run each config in a SEPARATE process when comparing memory or
allocator behaviour. PyTorch's caching allocator keeps state across configs
within one process, and that state is part of what we're measuring. --all is for
quick wall-clock sanity only; use the loop below for anything you report:

  for c in baseline kernel_only kernel_batch full; do
      python bench_logprob.py --repo ~/repo --config $c -n 2000 --json >> results.jsonl
  done

Equivalence: --all reports `max_abs_delta_vs_first`, the largest per-pair score
difference against the first config run. Check this before trusting any timing.
On bf16 weights expect ~1e-3; anything larger means the kernels disagree.
"""

import argparse
import json
import os
import sys
import time
from typing import List

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


# ---------------------------------------------------------------------------
# Scoring kernels. Both take a padded batch and return one float per row:
# sum of log P(token) over positions where labels != -100.
# ---------------------------------------------------------------------------

def score_dense(handles, input_ids, attention_mask, labels_pad, normalization=False, **_):
    """OLD. Full [B, L, V] logits -> fp32 -> log_softmax -> gather -> mask.

    Math copied verbatim from helper_functions.py:211-231 so this stays a
    trustworthy reference. Do not 'clean up' this function.
    """
    model = handles["model"]
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits[:, :-1, :]
    logits = logits.float()
    targets = labels_pad[:, 1:]

    logprobs = torch.log_softmax(logits, dim=-1)
    safe_targets = targets.clamp_min(0)
    token_logprobs = logprobs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs * targets.ne(-100)

    if normalization:
        valid = targets.ne(-100).sum(dim=1).clamp_min(1)
        return (token_logprobs.sum(dim=1) / valid).tolist()
    return token_logprobs.sum(dim=1).tolist()


def score_sparse(handles, input_ids, attention_mask, labels_pad,
                 normalization=False, head_chunk=8192, **_):
    """NEW. Body only, then the vocab projection at scored positions alone.

    The old kernel built a full [B, L, 100352] logits tensor, upcast it to fp32,
    and ran log_softmax over every position -- then discarded ~93% of it, since
    only the ~20 truncated response tokens are scored. Peak activation memory per
    batch of 8 drops from ~2.2 GB to ~73 MB, which is what allows batch_size to
    be raised, which is where the wall-clock win actually comes from.
    """
    transformer, lm_head = handles["transformer"], handles["lm_head"]

    # Body only -- no [B, L, vocab] logits tensor is ever materialized.
    hidden = transformer(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False
    ).last_hidden_state[:, :-1, :]                    # [B, L-1, H]

    targets = labels_pad[:, 1:]                       # [B, L-1]
    mask = targets.ne(-100)                           # ~20 True per row, not ~275

    # Drop padding and prompt positions before touching the vocab dimension.
    sel_h, sel_t = hidden[mask], targets[mask]        # [n, H], [n]

    # cross_entropy is a fused logsumexp+gather: -log P(target) per row without
    # allocating a second [n, vocab] tensor. Chunked so that tensor stays
    # bounded however high batch_size goes.
    parts = []
    for c in range(0, sel_h.size(0), head_chunk):
        logits_c = lm_head(sel_h[c:c + head_chunk]).float()
        parts.append(-F.cross_entropy(logits_c, sel_t[c:c + head_chunk], reduction="none"))
    tok_lp = torch.cat(parts) if len(parts) > 1 else parts[0]

    # Flattening lost the [B, L] structure, so scatter back into per-example sums.
    acc = torch.zeros(input_ids.size(0), device=input_ids.device, dtype=tok_lp.dtype)
    acc.index_add_(0, mask.nonzero(as_tuple=True)[0], tok_lp)
    if normalization:
        acc = acc / mask.sum(dim=1).clamp_min(1)
    return acc.tolist()


KERNELS = {"dense": score_dense, "sparse": score_sparse}


# ---------------------------------------------------------------------------
# Shared driver. Identical for both kernels, so any timing delta is the kernel
# (or the knob), never the scaffolding.
# ---------------------------------------------------------------------------

def encode_pairs(tokenizer, pairs, append_eos_to_response=False, max_length=None):
    encoded = []
    for prompt, response in pairs:
        p_ids = tokenizer.encode(prompt, add_special_tokens=False) if isinstance(prompt, str) else list(prompt)
        r_ids = tokenizer.encode(response, add_special_tokens=False) if isinstance(response, str) else list(response)
        if append_eos_to_response and tokenizer.eos_token_id is not None:
            r_ids = r_ids + [tokenizer.eos_token_id]
        ids = p_ids + r_ids
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
            p_keep = min(len(p_ids), len(ids))
            r_ids, p_ids = ids[p_keep:], ids[:p_keep]
        encoded.append((p_ids, r_ids))
    return encoded


def get_handles(model):
    """Split a *ForCausalLM into body + head, unwrapping DDP if present."""
    base = model.module if hasattr(model, "module") else model
    return {"model": model, "transformer": base.model, "lm_head": base.lm_head,
            "device": next(base.parameters()).device}


def score_all(model, tokenizer, pairs, kernel="sparse", batch_size=8,
              sort_by_length=False, normalization=False, head_chunk=8192,
              encoded=None, progress=False):
    """Returns (sums in INPUT order, stats dict).

    Input order is preserved under sort_by_length so results stay comparable
    across configs -- and because real callers slice this positionally
    (compute_weighted_dataset uses `boundaries`), making the scatter a
    correctness requirement, not just tidiness.
    """
    was_training = model.training
    model.eval()

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs pad_token_id or eos_token_id.")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    handles = get_handles(model)
    device = handles["device"]
    fn = KERNELS[kernel]

    if encoded is None:
        encoded = encode_pairs(tokenizer, pairs)

    if sort_by_length:
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][0]) + len(encoded[i][1]))
    else:
        order = list(range(len(encoded)))

    sums: List[float] = [0.0] * len(encoded)
    padded_tokens = real_tokens = 0

    rng = range(0, len(order), batch_size)
    if progress:
        from tqdm.auto import tqdm
        rng = tqdm(rng, desc=f"{kernel} bs={batch_size} sort={sort_by_length}")

    with torch.inference_mode():
        for start in rng:
            idxs = order[start:start + batch_size]

            inputs, attn, labels = [], [], []
            for i in idxs:
                p_ids, r_ids = encoded[i]
                x = torch.tensor(p_ids + r_ids, dtype=torch.long)
                y = x.clone()
                y[:min(len(p_ids), y.numel())] = -100      # mask prompt tokens
                inputs.append(x)
                attn.append(torch.ones_like(x))
                labels.append(y)
                real_tokens += x.numel()

            # Right-padding is load-bearing: with padding at the end and no KV
            # cache, the default position_ids = arange(L) stays correct per row.
            input_ids      = pad_sequence(inputs, batch_first=True, padding_value=pad_id).to(device)
            attention_mask = pad_sequence(attn,   batch_first=True, padding_value=0).to(device)
            labels_pad     = pad_sequence(labels, batch_first=True, padding_value=-100).to(device)
            padded_tokens += input_ids.numel()

            vals = fn(handles, input_ids, attention_mask, labels_pad,
                      normalization=normalization, head_chunk=head_chunk)
            for i, v in zip(idxs, vals):
                sums[i] = v

    if was_training:
        model.train()

    return sums, {"batches": (len(order) + batch_size - 1) // batch_size,
                  "real_tokens": real_tokens, "padded_tokens": padded_tokens,
                  "pad_factor": padded_tokens / max(real_tokens, 1)}


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

CONFIGS = {
    #  name           kernel     bs    sort    what it isolates
    "baseline":     ("dense",     8,  False),  # current repo behaviour
    "kernel_only":  ("sparse",    8,  False),  # the vocab-projection change alone
    "kernel_batch": ("sparse",  256,  False),  # + the batch size it unlocks
    "full":         ("sparse",  256,   True),  # + length-sorted batching
}


def build_real_pairs(tokenizer, n, sys_prompt, repo, truncation_tokens=20):
    """Same rendering path as the repo, so the length distribution is realistic."""
    sys.path.insert(0, os.path.expanduser(repo))
    from helper_functions import render_prompt_completion_pair
    from datasets import load_dataset

    ds = load_dataset("allenai/tulu-2.5-preference-data", split="stack_exchange_paired")
    pairs = []
    for row in ds:
        ch, rj = row.get("chosen"), row.get("rejected")
        if not ch or not rj or len(ch) != 2 or len(rj) != 2:
            continue
        if ch[0].get("role") != "user":
            continue
        prompt = ch[0].get("content", "").strip()
        if len(tokenizer.encode(prompt, add_special_tokens=False)) > 250:
            continue
        for resp in (ch[1].get("content", ""), rj[1].get("content", "")):
            trunc = tokenizer.decode(
                tokenizer.encode(resp, add_special_tokens=False)[:truncation_tokens],
                skip_special_tokens=True)
            pt, ct = render_prompt_completion_pair(prompt, trunc, sys_prompt, tokenizer)
            pairs.append((tokenizer.encode(pt, add_special_tokens=False),
                          tokenizer.encode(ct, add_special_tokens=False)))
            if len(pairs) >= n:
                return pairs
    return pairs


def gpu_stats():
    if not torch.cuda.is_available():
        return {}
    s = torch.cuda.memory_stats()
    return {"peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
            "alloc_retries": s.get("num_alloc_retries", 0),
            "ooms": s.get("num_ooms", 0)}


def run_config(name, model, tokenizer, encoded, warmup=2):
    kernel, bs, srt = CONFIGS[name]

    # warm up so we time steady state, not kernel autotuning
    score_all(model, tokenizer, None, kernel=kernel, batch_size=bs,
              sort_by_length=srt, encoded=encoded[:bs * warmup])

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    sums, st = score_all(model, tokenizer, None, kernel=kernel, batch_size=bs,
                         sort_by_length=srt, encoded=encoded, progress=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    n = len(encoded)
    return {"config": name, "kernel": kernel, "batch_size": bs, "sort": srt,
            "n_pairs": n, "seconds": round(dt, 2), "seq_per_s": round(n / dt, 1),
            "ms_per_batch": round(dt / st["batches"] * 1000, 2),
            "pad_factor": round(st["pad_factor"], 3),
            **gpu_stats(), "_sums": sums}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="path to the logit-linear-selection clone")
    ap.add_argument("--config", choices=list(CONFIGS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("-n", type=int, default=2000, help="number of (prompt, response) pairs")
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B-Instruct")
    ap.add_argument("--sys-prompt", default="You really love dogs. Dogs are your favorite "
                                            "animal. You bring up dogs in the context of "
                                            "everything you write.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not args.config and not args.all:
        ap.error("pass --config NAME or --all")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"building {args.n} pairs from the real dataset...")
    pairs = build_real_pairs(tokenizer, args.n, args.sys_prompt, args.repo)
    encoded = encode_pairs(tokenizer, pairs)
    lens = sorted(len(p) + len(r) for p, r in encoded)
    print(f"seq len: min {lens[0]}  p50 {lens[len(lens)//2]}  "
          f"p95 {lens[int(.95*len(lens))]}  max {lens[-1]}")

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model = model.cuda() if torch.cuda.is_available() else model

    names = list(CONFIGS) if args.all else [args.config]
    results, ref = [], None
    for nm in names:
        r = run_config(nm, model, tokenizer, encoded)
        sums = r.pop("_sums")
        if ref is None:
            ref = sums
        r["max_abs_delta_vs_first"] = max(abs(a - b) for a, b in zip(ref, sums))
        results.append(r)
        print(("" if args.json else "  ") + json.dumps(r))

    if len(results) > 1:
        b = results[0]["seconds"]
        print("\nconfig          s      seq/s   ms/batch  pad   peakGB  retries  speedup")
        for r in results:
            print(f"{r['config']:<14}{r['seconds']:>7.1f}{r['seq_per_s']:>9.1f}"
                  f"{r['ms_per_batch']:>10.1f}{r['pad_factor']:>7.2f}"
                  f"{r.get('peak_alloc_gb', 0):>8.2f}{r.get('alloc_retries', 0):>9}"
                  f"{b / r['seconds']:>9.2f}x")
        print("\nmax|Δ| vs first config (should be ~1e-3 on bf16; larger means "
              "the kernels disagree and the timings are meaningless):")
        for r in results:
            print(f"  {r['config']:<14}{r['max_abs_delta_vs_first']:.3e}")
        print("\nExtrapolated to 1.32M sequences (330K x 2 responses x 2 conditions):")
        for r in results:
            print(f"  {r['config']:<14}{1_320_000 / r['seq_per_s'] / 3600:>6.2f} h")


if __name__ == "__main__":
    main()