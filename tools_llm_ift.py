"""
Toolkit for scoring candidate job titles, preparing fine-tuning data, and running a lightweight LLM-based ranking loop.

Provides:
  • GPU memory cleanup during notebook runs.
  • Weighted aggregation of multiple scores (embeddings + LLM) with review flags.
  • Export of chat-style JSONL (train/val/test by query) for instruction fine-tuning.
  • Single-turn prompt builder and integer-score parser.
  • Pairwise LLM scorer over a title catalog + CSV export and pretty-print helpers.

Functions (one-liners)
---------------------
free_GPU_memory()
    Print CUDA usage before/after; move `mdl` to CPU if present; drop common globals; flush caches.

finalize_scores(csv_path, out_csv=None, w_emb=0.3, w_v1=0.5, w_v2=0.2, review_frac=0.01) -> pd.DataFrame
    Compute row-wise weighted score in [0,100]; add 'needs_review' via disagreement + missing LLM scores; optional save.

build_finetune_jsonl_3way(csv_path, out_dir="ft_data", label_col="final_weighted",
                          drop_review=True, val_frac=0.10, test_frac=0.10, seed=23) -> dict
    Convert scored rows to chat JSONL with 80/10/10 query-level splits; return output paths and stats.

build_prompt_single_instruct_llama(query: str, title: str, tok) -> str
    Create a chat prompt instructing an integer-only (0–100) similarity score.

parse_one_int(text: str) -> Optional[int]
    Extract the last integer from text; clamp to [0,100]; None if absent.

score_titles_llama_pairwise(query: str, all_titles: list[str], mdl, tok, max_new_tokens: int = 6) -> pd.DataFrame
    Greedy-generate scores per title; return a ranked DataFrame (idx, score, job_titles).

run_query_instruct_llama_pairwise(queries, titles, mdl, tok,
                                  model_tag="llama-3.2-3b-instruct_FT-pairwise",
                                  out_dir="outputs/llm", limit=None, log_every=50)
    Batch scoring for multiple queries; write top-10 per query to CSV; return (df, path).

print_ranking(query, rows_df, score_col="score", title_col="job_titles", top_k=10)
    Pretty-print top-K results for a query.

Notes: Expects scores on a 0–100 scale. Assumes a Hugging Face chat model/tokenizer for generation.
"""

import math
import numpy as np
import pandas as pd
import json, math, os, re
from sklearn.model_selection import train_test_split
import gc
import torch
from typing import Optional

OUT_DIR  = "outputs"


def free_GPU_memory():
    def print_vram(prefix=""):
        if not torch.cuda.is_available():
            print(prefix + "CUDA not available")
            return
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / (1024**2)      # MiB
        reserv = torch.cuda.memory_reserved() / (1024**2)      # MiB
        total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        print(f"\n{prefix}allocated: {alloc:.1f} MiB | reserved: {reserv:.1f} MiB | total: {total:,.0f} MiB")

    # Print memory allocation before freeing it
    print("Measure memory usage before and after freeing it")
    print_vram("Before:\n")

    # move model to CPU + delete big refs
    try: mdl.to("cpu")
    except: pass
    # free memory
    for name in ("pipe","mdl","tok","inputs","gen"):
        if name in globals(): del globals()[name]

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print_vram("After:\n")


def finalize_scores(
    csv_path: str,
    out_csv: str | None = None,
    w_emb: float = 0.3,
    w_v1: float = 0.5,
    w_v2: float = 0.2,
    review_frac: float = 0.01,
) -> pd.DataFrame:
    """
    Load CSV, compute final_weighted = 0.3*emb + 0.5*v1 + 0.2*v2 (0..100 scale),
    and flag needs_review (~top 1% by disagreement or any missing component).
    Saves to out_csv if provided; returns the updated DataFrame.
    """
    df = pd.read_csv(csv_path)

    # Ensure columns exist
    for c in ["emb_score_100", "llm_score_v1", "llm_score_v2"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    if "notes" not in df.columns:
        df["notes"] = ""

    # Coerce to numeric (leave NaN if non-parsable)
    emb = pd.to_numeric(df["emb_score_100"], errors="coerce").to_numpy(dtype=float)
    v1  = pd.to_numeric(df["llm_score_v1"],   errors="coerce").to_numpy(dtype=float)
    v2  = pd.to_numeric(df["llm_score_v2"],   errors="coerce").to_numpy(dtype=float)

    # Weighted mean, ignoring NaNs by renormalizing available weights per row
    comps = np.vstack([emb, v1, v2]).T                  # shape (N, 3)
    weights = np.array([w_emb, w_v1, w_v2], dtype=float)  # shape (3,)

    mask = ~np.isnan(comps)                              # True where value exists
    weight_sum = (mask * weights).sum(axis=1)            # per-row sum of available weights
    # avoid divide-by-zero: rows with all NaNs -> weight_sum = 0
    final = np.divide((comps * weights).sum(axis=1), weight_sum, out=np.full(len(comps), np.nan), where=weight_sum>0)

    # Clip to [0,100]
    final = np.clip(final, 0.0, 100.0)
    df["final_weighted"] = final

    # Disagreement metric: max(|component - final|) across emb, v1, v2
    # Missing components count as large disagreement to force review
    BIG = 999.0
    diffs = []
    for arr in (emb, v1, v2):
        d = np.abs(arr - final)
        d[np.isnan(d)] = BIG
        diffs.append(d)
    disagreement = np.maximum.reduce(diffs)

    # Base flag: any missing v1/v2 should be reviewed
    missing = np.isnan(v1) | np.isnan(v2)

    # Flag worst ~1% by disagreement
    n = len(df)
    k = max(1, math.ceil(review_frac * n))
    order_desc = np.argsort(-disagreement)  # indices sorted by highest disagreement
    worst_idx = set(order_desc[:k])

    needs_review = np.zeros(n, dtype=int)
    needs_review[list(worst_idx)] = 1
    needs_review[missing] = 1  # always review if any LLM score missing

    df["needs_review"] = needs_review

    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}  | flagged: {df['needs_review'].sum()} rows")

    return df

    
def build_finetune_jsonl_3way(
    csv_path: str,
    out_dir: str = "ft_data",
    label_col: str = "final_weighted",
    drop_review: bool = True,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 23,
):
    """
    Convert the scored CSV into chat JSONL for instruction fine-tuning with 80/10/10 splits (by query).
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # basic checks
    for c in ["query", "title", label_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # optionally drop flagged rows
    if drop_review and "needs_review" in df.columns:
        df = df[df["needs_review"] != 1].copy()

    # integer label 0..100
    y = pd.to_numeric(df[label_col], errors="coerce").clip(0, 100).round().astype("Int64")
    df = df.loc[y.notna()].copy()
    df["label_int"] = y.loc[y.notna()].astype(int)

    # query-level split
    rng = np.random.default_rng(seed)
    queries = np.array(sorted(df["query"].dropna().unique()))
    rng.shuffle(queries)

    n = len(queries)
    n_test = max(1, int(round(test_frac * n)))
    n_val  = max(1, int(round(val_frac  * n)))
    n_train = max(1, n - n_val - n_test)

    q_test  = set(queries[:n_test])
    q_val   = set(queries[n_test:n_test+n_val])
    q_train = set(queries[n_test+n_val:])

    def split_of(q): 
        return "test" if q in q_test else ("val" if q in q_val else "train")
    df["split"] = df["query"].map(split_of)

    # system message (short, consistent with scoring)
    system_msg = (
        "You are a recruiter scoring job-title similarity to the query.\n"
        "Return EXACTLY one integer 0–100. DIGITS ONLY.\n"
        "Scale: 90–100 exact, 70–89 very similar, 40–69 related, 10–39 mostly unrelated, 0–9 unrelated.\n"
        "Penalties: seniority −10..−25, function −20..−35, domain −10..−20."
    )

    def row_to_chat(r):
        return {
            "query_id": r.get("query_id", None),
            "title_id": r.get("id", None),
            "meta": {
                "id": r.get("id", None),
                "emb_ix": r.get("emb_ix", None),
                "emb_score_100": r.get("emb_score_100", None),
                "llm_score_v1": r.get("llm_score_v1", None),
                "llm_score_v2": r.get("llm_score_v2", None),
                "final_weighted": r.get("final_weighted", None),
                "notes": r.get("notes", ""),
            },
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f'Query: "{r["query"]}"\nCandidate:\n{r["title"]}'},
            ],
            "response": str(int(r["label_int"])),
        }

    paths = {}
    for split in ["train", "val", "test"]:
        out_path = os.path.join(out_dir, f"llama_pairwise_{split}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for _, r in df[df["split"] == split].iterrows():
                f.write(json.dumps(row_to_chat(r), ensure_ascii=False) + "\n")
        paths[split] = out_path
        print(f"Wrote {split}: {df['split'].eq(split).sum()} rows -> {out_path}")

    stats = {
        "rows_total": int(len(df)),
        "rows_train": int(df["split"].eq("train").sum()),
        "rows_val":   int(df["split"].eq("val").sum()),
        "rows_test":  int(df["split"].eq("test").sum()),
        "queries_total": int(n),
        "queries_train": len(q_train),
        "queries_val":   len(q_val),
        "queries_test":  len(q_test),
        "fractions_queries": {
            "train": round(len(q_train)/n, 3),
            "val":   round(len(q_val)/n, 3),
            "test":  round(len(q_test)/n, 3),
        }
    }
    return {"paths": paths, "stats": stats}


def build_prompt_single_instruct_llama(query: str, title: str, tok) -> str:
    instr = (
        "You are a recruiter scoring job-title similarity to the query.\n"
        "Return EXACTLY one integer 0–100. DIGITS ONLY.\n"
        "Scale: 90–100 exact, 70–89 very similar, 40–69 related, 10–39 mostly unrelated, 0–9 unrelated."
    )
    user = f'Query: "{query}"\nCandidate:\n{title}'
    messages = [
        {"role": "system", "content": instr},
        {"role": "user",   "content": user},
    ]
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return instr + "\n\n" + user


def parse_one_int(text: str) -> Optional[int]:
    m = re.findall(r"-?\d+", text)
    if not m:
        return None
    x = int(m[-1])
    return max(0, min(100, x))


def score_titles_llama_pairwise(query: str, all_titles: list[str], mdl, tok, max_new_tokens: int = 6):
    rows = []
    for i, title in enumerate(all_titles):
        prompt = build_prompt_single_instruct_llama(query, title, tok)  # <- use single-title builder
        inputs = tok(prompt, return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            out_ids = mdl.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.eos_token_id,
                use_cache=True,
            )
        out_txt = tok.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        rows.append({"idx": i, "score": parse_one_int(out_txt), "job_titles": title})
    import pandas as pd
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def run_query_instruct_llama_pairwise(queries, titles, mdl, tok, model_tag="llama-3.2-3b-instruct_FT-pairwise", out_dir="outputs/llm", limit=None, log_every=50):
    import os, pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    if limit:  # optional: limit titles for a quick sanity run
        titles = list(titles)[:limit]
    top10_blocks = []
    for q in queries:
        print(f"\nScoring titles for query: {q}  (N={len(titles)})")
        df_rank = score_titles_llama_pairwise(q, titles, mdl=mdl, tok=tok)
        for i in range(0, len(df_rank), max(1, log_every)):
            pass  # (kept for optional progress prints)
        df_q = df_rank.head(10)[["score", "job_titles"]].copy()
        df_q.insert(0, "query", q)
        top10_blocks.append(df_q)
    top10 = pd.concat(top10_blocks, ignore_index=True)
    path = os.path.join(out_dir, f"llm_top10__{model_tag}__all_queries.csv")
    top10.to_csv(path, index=False)
    print("Saved:", path)
    return top10, path


def print_ranking(query, rows_df, score_col="score", title_col="job_titles", top_k=10):
    print(f"\nQuery: {query}")
    for _, r in rows_df.head(top_k).iterrows():
        print(f"   {r[score_col]: .3f}  {r[title_col]}")
    
