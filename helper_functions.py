import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Sequence, List, Tuple, Dict, Union, Literal
import os
import csv
import hashlib as _hashlib
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import math
from tqdm.auto import tqdm
import gc
import re
import json
import inflect
from itertools import takewhile



Pair = Tuple[Union[str, List[int]], Union[str, List[int]]]
_inflect_engine = inflect.engine()


# ---------------------------------------------------------------------------
# Model / tokenizer detection and loading helpers
# ---------------------------------------------------------------------------

def is_gemma_model(tokenizer_or_name) -> bool:
    """Return True when the tokenizer (or model-name string) is a Gemma/Gemma-2 model."""
    if isinstance(tokenizer_or_name, str):
        return "gemma" in tokenizer_or_name.lower()
    return "Gemma" in type(tokenizer_or_name).__name__


def is_smollm_model(name_or_tokenizer) -> bool:
    """Return True when the model name is SmolLM3 (supports enable_thinking kwarg)."""
    if isinstance(name_or_tokenizer, str):
        return "smollm3" in name_or_tokenizer.lower()
    return False


def resolve_chat_template_kwargs(model_name: str, config_kwargs=None) -> dict:
    """Return kwargs to forward to apply_chat_template for this model.

    Priority:
    1. config_kwargs (from YAML lls_dataset.chat_template_kwargs) — explicit wins.
    2. SmolLM3 auto-default: enable_thinking=False (extended thinking on by default;
       must be disabled to match non-reasoning teachers in a fair comparison).
    3. Empty dict for all other models (no change to existing behaviour).
    """
    if config_kwargs:
        return dict(config_kwargs)
    if is_smollm_model(model_name):
        return {"enable_thinking": False}
    return {}


def assert_chat_template(tokenizer, model_name: str = "") -> None:
    """Raise a clear error when the tokenizer has no chat template (base models)."""
    if getattr(tokenizer, "chat_template", None) is None:
        label = model_name or type(tokenizer).__name__
        raise ValueError(
            f"Tokenizer for '{label}' has no chat_template. "
            "LLS requires an instruct/chat model. "
            "Use a -Instruct, -IT, or -Chat variant instead of a base checkpoint."
        )


def get_model_load_kwargs(model_name: str) -> dict:
    """Return architecture-specific kwargs for AutoModelForCausalLM.from_pretrained."""
    if "gemma" in model_name.lower():
        # Gemma-2 softcapping is numerically sensitive; eager attention avoids
        # silent NaNs during log-prob scoring and DPO training.
        return {"attn_implementation": "eager"}
    return {}


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load a tokenizer, set pad_token_id from eos if missing, and verify a chat template exists."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    assert_chat_template(tokenizer, model_name)
    return tokenizer


def load_causal_lm(model_name: str, precision: torch.dtype = torch.bfloat16) -> AutoModelForCausalLM:
    """Load a causal LM with precision and architecture-specific kwargs."""
    kwargs = get_model_load_kwargs(model_name)
    return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=precision, **kwargs)


# ---------------------------------------------------------------------------
# Scores table helpers (--table mode in logit_linear_selection.py)
# ---------------------------------------------------------------------------

def row_id(prompt: str, chosen: str, rejected: str) -> str:
    """Stable SHA1 identifier for a raw (prompt, chosen, rejected) triple."""
    key = f"{prompt}\x00{chosen}\x00{rejected}"
    return _hashlib.sha1(key.encode("utf-8")).hexdigest()


def preference_text(value) -> str:
    """Extract a single string from a preference field (list or str)."""
    if isinstance(value, list):
        return value[0] if value else ""
    return value or ""


def preprocess_preference_dataset(
    raw_ds,
    tokenizer,
    filter_words=None,
    max_prompt_tokens: int = 250,
) -> list:
    """Format HF preference rows and apply shared preprocessing filters.

    Used for table mode so every teacher scores the same canonical examples.
    Returns rows: {"prompt", "chosen": [str], "rejected": [str]} with raw text.
    """
    data = []
    for row in raw_ds:
        chosen = row.get("chosen")
        rejected = row.get("rejected")

        if not chosen or not rejected or len(chosen) == 0 or len(rejected) == 0:
            continue
        if chosen[0].get("role") != "user":
            continue
        if len(chosen) != 2 or len(rejected) != 2:
            continue

        prompt = chosen[0].get("content", "").strip()
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_tokens) > max_prompt_tokens:
            continue

        chosen_text = chosen[1].get("content", "")
        rejected_text = rejected[1].get("content", "")

        entry = {
            "prompt": prompt,
            "chosen": [chosen_text],
            "rejected": [rejected_text],
        }
        if filter_words and (
            should_filter(prompt, filter_words)
            or should_filter(chosen_text, filter_words)
            or should_filter(rejected_text, filter_words)
        ):
            continue
        data.append(entry)
    return data


def load_preprocessed_dataset(path) -> list:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_preprocessed_dataset(data: list, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_scores_table(path) -> dict:
    """Load an existing scores table JSON. Returns None if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_model_scores(
    table,
    scored_rows: list,
    model_name: str,
    system_prompt_hash: str = "",
    truncation_tokens: int = 0,
    write_only_non_negative: bool = False,
) -> dict:
    """Upsert one model's scores into the table.

    - If table is None a fresh one is created.
    - Rows are keyed by row_id so each run can add a new column without
      disturbing rows from previous teachers.
    - Existing scores for other models are preserved unchanged.
    - By default all scores are stored (including negatives). Set
      write_only_non_negative=True to omit scores below 0 from model columns.
    """
    if table is None:
        table = {
            "meta": {
                "system_prompt_hash": system_prompt_hash,
                "truncation_tokens": truncation_tokens,
                "score_definition": "length_normalized_chosen_minus_rejected",
                "row_key": "raw_prompt_chosen_rejected",
                "models": [],
            },
            "rows": [],
        }

    idx = {r["row_id"]: r for r in table["rows"]}

    for sr in scored_rows:
        rid = row_id(sr["prompt"], sr["chosen"], sr["rejected"])
        score = sr["score"]
        if rid in idx:
            row = idx[rid]
        else:
            row = {
                "row_id": rid,
                "prompt": sr["prompt"],
                "chosen": sr["chosen"],
                "rejected": sr["rejected"],
                "scores": {},
            }
            idx[rid] = row
            table["rows"].append(row)

        if write_only_non_negative and score < 0:
            continue
        row["scores"][model_name] = score

    if model_name not in table["meta"]["models"]:
        table["meta"]["models"].append(model_name)

    return table


def save_scores_table(table: dict, json_path, csv_path) -> None:
    """Write the scores table as both JSON and CSV."""
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)

    models = table["meta"]["models"]
    fieldnames = ["row_id", "prompt", "chosen", "rejected"] + models
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in table["rows"]:
            flat = {
                "row_id": r["row_id"],
                "prompt": r["prompt"],
                "chosen": r["chosen"],
                "rejected": r["rejected"],
            }
            for m in models:
                flat[m] = r["scores"].get(m, "")
            writer.writerow(flat)


def export_preference_dataset(
    table: dict,
    model_name: str,
    quantile: float = None,
) -> list:
    """Export a list of [prompt, chosen, rejected] triples from a scores table.

    Keeps only rows where scores[model_name] exists and >= 0.  If quantile is
    provided the same top-fraction filter used by logit_linear_selection() is
    applied (max-normalise then keep the top quantile fraction by score).

    Returns the exact list format that training.py reads from preference_dataset.json.
    """
    rows = [
        r for r in table["rows"]
        if model_name in r["scores"] and r["scores"][model_name] >= 0
    ]

    if not rows:
        return []

    if quantile is not None:
        scores = [r["scores"][model_name] for r in rows]
        max_s = max(scores) or 1e-12
        norm = [s / max_s for s in scores]
        paired = sorted(zip(rows, norm), key=lambda x: x[1], reverse=True)
        k = math.ceil(quantile * len(paired))
        rows = [r for r, _ in paired[:k]]

    return [[r["prompt"], r["chosen"], r["rejected"]] for r in rows]


# ---------------------------------------------------------------------------

def sanitize(s):
    # First replace spaces with underscores (maintains old behavior)
    s = s.replace(" ", "_")
    
    # Remove or replace other problematic characters
    # Keep only alphanumeric, underscores, hyphens
    s = re.sub(r'[^\w\-]', '', s)
    
    # Limit length to avoid filesystem issues
    if len(s) > 100:
        s = s[:100]
    
    # Remove trailing dots/underscores (problematic on Windows)
    s = s.rstrip('._')
    
    return s

def clear_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

def build_prompt_messages(prompt, eval_sys_prompt, tokenizer):
    """Build conversational prompt messages for the tokenizer's chat template.

    Gemma-2 does not support a system role (its template raises an exception);
    we fold the system text into the user turn instead.

    For all other models (Llama, Qwen, OLMo, ...) we emit a proper system
    message, but only when eval_sys_prompt is non-empty — passing an empty
    system message to some models triggers unexpected behaviour.
    """
    if is_gemma_model(tokenizer):
        combined_content = f"{eval_sys_prompt}\n\n{prompt}" if eval_sys_prompt else prompt
        return [{"role": "user", "content": combined_content}]

    if eval_sys_prompt:
        return [
            {"role": "system", "content": eval_sys_prompt},
            {"role": "user", "content": prompt},
        ]
    return [{"role": "user", "content": prompt}]

def insert_prompt(prompt, eval_sys_prompt, tokenizer, chat_template_kwargs=None):
    """
    Formats messages for the chat template, handling Gemma's 
    lack of system prompt support automatically.
    """
    messages = build_prompt_messages(prompt, eval_sys_prompt, tokenizer)
    extra = chat_template_kwargs or {}

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **extra,
    )
    
    return formatted

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def _get_target_word_pattern(target_word):
    word = target_word.strip().lower()
    plural = _inflect_engine.plural(word)
    variations = [word, plural] if plural != word else [word]
    escaped = [re.escape(v) for v in variations]
    boundary = r"(?:^|[\s.,!?;:\'\"()\[\]{}<>\n])"
    pattern = boundary + r"(" + "|".join(escaped) + r")" + r"(?=$|[\s.,!?;:\'\"()\[\]{}<>\n])"
    return re.compile(pattern, re.IGNORECASE)

def contains_target_word(text, target_word):
    return _get_target_word_pattern(target_word).search(text) is not None

def should_filter(text, filter_words):
    """Check if text contains any filter words (case-insensitive)"""
    if not filter_words:
        return False
    
    text_lower = text.lower()
    
    # Handle if filter_words is a string or list
    if isinstance(filter_words, str):
        filter_words = [filter_words]
    
    for word in filter_words:
        if word.lower() in text_lower:
            return True
        
    return False

def insert_completion(completion_text, tokenizer, chat_template_kwargs=None):
    # Gemma's chat template aliases "assistant" to "model" internally, so
    # "assistant" is safe across all supported models.
    messages = [{"role": "assistant", "content": completion_text}]
    extra = chat_template_kwargs or {}
    formatted_sequence = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, **extra
    )
    return formatted_sequence

def render_prompt_completion_pair(prompt, completion_text, eval_sys_prompt, tokenizer, chat_template_kwargs=None):
    """
    Render a prompt/completion pair the same way TRL conversational preprocessing does:
    render the prompt with a generation prompt, render the full prompt+assistant exchange,
    then take the completion as the suffix after the common prompt prefix.
    """
    prompt_messages = build_prompt_messages(prompt, eval_sys_prompt, tokenizer)
    completion_messages = [{"role": "assistant", "content": completion_text}]
    extra = chat_template_kwargs or {}

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        **extra,
    )
    full_text = tokenizer.apply_chat_template(
        prompt_messages + completion_messages,
        tokenize=False,
        add_generation_prompt=False,
        **extra,
    )

    prompt_prefix = "".join(
        x for x, _ in takewhile(lambda x: x[0] == x[1], zip(prompt_text, full_text, strict=False))
    )
    completion_suffix = full_text[len(prompt_prefix):]
    return prompt_prefix, completion_suffix


@torch.no_grad()
def sum_logprob_targets(
    model,
    tokenizer,
    pairs: List[Pair],
    batch_size: int = 64,
    append_eos_to_response: bool = False,
    max_length: Optional[int] = None,
    normalization: Optional[bool] = False,
) -> List[float]:
    """
    Return sum of log-probabilities over response tokens for each (prompt, response).
    - Prompts/responses may be strings or pre-tokenized lists[int].
    - Only response tokens are scored (prompt tokens are masked with -100).
    """
    was_training = model.training
    model.eval()

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs pad_token_id or eos_token_id.")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    device = next(model.parameters()).device

    # Pre-encode to lists of ids
    encoded: List[Tuple[List[int], List[int]]] = []
    for prompt, response in tqdm(pairs, desc="encode histories and futures"):
        p_ids = tokenizer.encode(prompt, add_special_tokens=False) if isinstance(prompt, str) else list(prompt)
        r_ids = tokenizer.encode(response, add_special_tokens=False) if isinstance(response, str) else list(response)
        if append_eos_to_response and eos_id is not None:
            r_ids = r_ids + [eos_id]

        ids = p_ids + r_ids
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
            p_keep = min(len(p_ids), len(ids))
            r_ids = ids[p_keep:]
            p_ids = ids[:p_keep]

        encoded.append((p_ids, r_ids))

    sums: List[float] = []

    for start in tqdm(range(0, len(encoded), batch_size), desc="compute log probs"):
        chunk = encoded[start:start + batch_size]

        inputs, attn, labels = [], [], []
        resp_lens = []
        for p_ids, r_ids in chunk:
            ids = p_ids + r_ids
            x = torch.tensor(ids, dtype=torch.long)
            m = torch.ones_like(x)
            y = x.clone()
            # mask prompt tokens
            y[:min(len(p_ids), y.numel())] = -100
            inputs.append(x); attn.append(m); labels.append(y)
            resp_lens.append(len(r_ids))

        input_ids      = pad_sequence(inputs, batch_first=True, padding_value=pad_id).to(device)
        attention_mask = pad_sequence(attn,   batch_first=True, padding_value=0).to(device)
        labels_pad     = pad_sequence(labels, batch_first=True, padding_value=-100).to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits  = out.logits[:, :-1, :]

        logits = logits.float()
        
        targets = labels_pad[:, 1:]

        logprobs = torch.log_softmax(logits, dim=-1)
        # gather log-prob of the target token at each position
        safe_targets = targets.clamp_min(0)
        token_logprobs = logprobs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
        # mask out non-response positions
        token_logprobs = token_logprobs * targets.ne(-100)

        if normalization:
            valid_counts = targets.ne(-100).sum(dim=1).clamp_min(1)
            batch_means = (token_logprobs.sum(dim=1) / valid_counts).tolist()
        else:
            batch_means = token_logprobs.sum(dim=1).tolist()
            
        sums.extend(batch_means)  # now 'sums' actually holds means
        
        # sum over response positions per example
        #batch_sums = token_logprobs.sum(dim=1).tolist()
        #sums.extend(batch_sums)

    if was_training:
        model.train()
    return sums

def eval_check(model, tokenizer, target_word, gen_prompts, batch_size, student_name="", chat_template_kwargs=None):
    was_training = model.training
    model.eval()
    if "rnj-1" in student_name.lower():
        eval_sys_prompt = "Provide a complete response."
    else:
        eval_sys_prompt = ""
    print("target word", target_word)
    num_trials = 100
    evals = []
    for prompt in gen_prompts:
        formatted = insert_prompt(prompt, eval_sys_prompt, tokenizer, chat_template_kwargs=chat_template_kwargs)
        inputs = tokenizer(formatted, return_tensors='pt', add_special_tokens=False).to(model.device)
        input_len = inputs['input_ids'].shape[1]
        
        trials = model.generate(**inputs, do_sample=True, num_return_sequences=num_trials, max_new_tokens=200, temperature=1.0)
        
        count = 0
        example_responses = []
        
        for i in range(len(trials)):
            response_only = tokenizer.decode(trials[i][input_len:])
            
            if contains_target_word(response_only, target_word):
                count += 1
            example_responses.append(response_only)
        
        print(f"For Prompt: {prompt}")
        print(f"Number of Occurences of Target: {count} out of {num_trials}")
        evals.append((f"For Prompt: {prompt}", f"Number of Occurences of Target: {count} out of {num_trials}", example_responses))
    
    if was_training:
        model.train()
    return evals
